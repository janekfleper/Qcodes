"""
The measurement module provides a context manager for registering parameters
to measure and storing results. The user is expected to mainly interact with it
using the :class:`.Measurement` class.
"""


import json
import logging
from time import monotonic
from collections import OrderedDict
from typing import (Callable, Union, Dict, Tuple, List, Sequence, cast, Set,
                    MutableMapping, MutableSequence, Optional, Any, TypeVar)
from inspect import signature
from numbers import Number
from copy import deepcopy

import numpy as np

import qcodes as qc
from qcodes import Station
from qcodes.instrument.parameter import ArrayParameter, _BaseParameter, \
    Parameter, MultiParameter, ParameterWithSetpoints
from qcodes.dataset.experiment_container import Experiment
from qcodes.dataset.param_spec import ParamSpec, ParamSpecBase
from qcodes.dataset.dependencies import (InterDependencies_,
                                         DependencyError, InferenceError)
from qcodes.dataset.data_set import DataSet, VALUE
from qcodes.utils.helpers import NumpyJSONEncoder
import qcodes.config

log = logging.getLogger(__name__)

array_like_types = (tuple, list, np.ndarray)
scalar_res_types = Union[str, int, float, np.dtype]
values_type = Union[scalar_res_types, np.ndarray,
                    Sequence[scalar_res_types]]
res_type = Tuple[Union[_BaseParameter, str],
                 Union[scalar_res_types, np.ndarray,
                       Sequence[scalar_res_types]]]
setpoints_type = Sequence[Union[str, _BaseParameter]]
numeric_types = Union[float, int]


class ParameterTypeError(Exception):
    pass


def is_number(thing: Any) -> bool:
    """
    Test if an object can be converted to a number UNLESS it is a string
    """
    if isinstance(thing, str):
        return False
    try:
        float(thing)
        return True
    except (ValueError, TypeError):
        return False


class DataSaver:
    """
    The class used by the Runner context manager to handle the datasaving to
    the database.
    """

    default_callback: Optional[dict] = None

    def __init__(self, dataset: DataSet,
                 write_period: numeric_types,
                 interdeps: InterDependencies_) -> None:
        self._dataset = dataset
        if DataSaver.default_callback is not None \
                and 'run_tables_subscription_callback' \
                    in DataSaver.default_callback:
            callback = DataSaver.default_callback[
                'run_tables_subscription_callback']
            min_wait = DataSaver.default_callback[
                'run_tables_subscription_min_wait']
            min_count = DataSaver.default_callback[
                'run_tables_subscription_min_count']
            snapshot = dataset.get_metadata('snapshot')
            self._dataset.subscribe(callback,
                                    min_wait=min_wait,
                                    min_count=min_count,
                                    state={},
                                    callback_kwargs={'run_id':
                                                         self._dataset.run_id,
                                                     'snapshot': snapshot})
        default_subscribers = qcodes.config.subscription.default_subscribers
        for subscriber in default_subscribers:
            self._dataset.subscribe_from_config(subscriber)

        self._interdeps = interdeps
        self.write_period = float(write_period)
        self._results: List[Dict[str, VALUE]] = []  # will be filled by addResult
        self._last_save_time = monotonic()
        self._known_dependencies: Dict[str, List[str]] = {}

    def add_result(self, *res_tuple: res_type) -> None:
        """
        Add a result to the measurement results. Represents a measurement
        point in the space of measurement parameters, e.g. in an experiment
        varying two voltages and measuring two currents, a measurement point
        is the four dimensional (v1, v2, c1, c2). The corresponding call
        to this function would be (e.g.)
        >> datasaver.add_result((v1, 0.1), (v2, 0.2), (c1, 5), (c2, -2.1))

        For better performance, this function does not immediately write to
        the database, but keeps the results in memory. Writing happens every
        `write_period` seconds and during the __exit__ method if this class.


        Args:
            res_tuple: a tuple with the first element being the parameter name
                and the second element is the corresponding value(s) at this
                measurement point. The function takes as many tuples as there
                are results.

        Raises:
            ValueError: if a parameter name not registered in the parent
                Measurement object is encountered.
            ValueError: if the shapes of parameters do not match, i.e. if
                a parameter gets values of a different shape than its setpoints
                 (the exception being that setpoints can always be scalar)
            ParameterTypeError: if a parameter is given a value not matching
                its type.
        """

        # we iterate through the input twice. First we find any array and
        # multiparameters that need to be unbundled and collect the names
        # of all parameters. This also allows users to call
        # add_result with the arguments in any particular order, i.e. NOT
        # enforcing that setpoints come before dependent variables.

        results_dict: Dict[ParamSpecBase, Any] = {}

        for partial_result in res_tuple:
            parameter = partial_result[0]
            if isinstance(parameter, ArrayParameter):
                results_dict.update(self._unpack_arrayparameter(partial_result))
            elif isinstance(parameter, MultiParameter):
                results_dict.update(self._unpack_multiparameter(partial_result))
            else:
                results_dict.update(self._unpack_partial_result(partial_result))

        self._validate_result_deps(results_dict)
        self._validate_result_shapes(results_dict)
        self._validate_result_types(results_dict)

        self._enqueue_results(results_dict)

        if monotonic() - self._last_save_time > self.write_period:
            self.flush_data_to_database()
            self._last_save_time = monotonic()

    def _unpack_partial_result(
        self, partial_result: res_type) -> Dict[ParamSpecBase, values_type]:
        """
        Unpack a partial result (not containing ArrayParameters or
        MultiParameters) into a standard results dict form and return that
        dict
        """
        param, values = partial_result
        try:
            parameter = self._interdeps._id_to_paramspec[str(param)]
        except KeyError:
            raise ValueError('Can not add result for parameter '
                             f'{param}, no such parameter registered '
                             'with this measurement.')
        return {parameter: values}

    def _unpack_arrayparameter(
        self, partial_result: Tuple[ArrayParameter, values_type]
        ) -> Dict[ParamSpecBase, values_type]:
        """
        Unpack a partial result containing an arrayparameter into a standard
        results dict form and return that dict
        """
        array_param, values_array = partial_result

        if array_param.setpoints is None:
            raise RuntimeError(f"{array_param.full_name} is an "
                               f"{type(array_param)} "
                               f"without setpoints. Cannot handle this.")
        try:
            main_parameter = self._interdeps._id_to_paramspec[str(array_param)]
        except KeyError:
            raise ValueError('Can not add result for parameter '
                             f'{array_param}, no such parameter registered '
                             'with this measurement.')

        res_dict = {main_parameter: values_array}

        sp_names = array_param.setpoint_full_names
        fallback_sp_name = f"{array_param.full_name}_setpoint"

        res_dict.update(
            self._unpack_setpoints_from_parameter(
                array_param, array_param.setpoints,
                sp_names, fallback_sp_name))

        return res_dict

    def _unpack_multiparameter(
        self, partial_result: res_type) -> Dict[ParamSpecBase, Any]:
        """
        Unpack the subarrays and setpoints from a MultiParameter and
        into a standard results dict form and return that
        dict

        Args:
            parameter: The MultiParameter to extract from
            data: The acquired data for this parameter
        """

        parameter, data = partial_result

        result_dict = {}

        if parameter.setpoints is None:
            raise RuntimeError(f"{parameter.full_name} is an {type(parameter)} "
                               f"without setpoints. Cannot handle this.")
        for i in range(len(parameter.shapes)):
            shape = parameter.shapes[i]

            try:
                paramspec = self._interdeps._id_to_paramspec[parameter.names[i]]
            except KeyError:
                raise ValueError('Can not add result for parameter '
                                 f'{parameter.names[i]}, '
                                 'no such parameter registered '
                                 'with this measurement.')

            result_dict.update({paramspec: data[i]})
            if shape != ():
                # array parameter like part of the multiparameter
                # need to find setpoints too
                fallback_sp_name = f'{parameter.full_names[i]}_setpoint'

                if parameter.setpoint_full_names[i] is not None:
                    sp_names = parameter.setpoint_full_names[i]
                else:
                    sp_names = None

                result_dict.update(
                    self._unpack_setpoints_from_parameter(
                        parameter,
                        parameter.setpoints[i],
                        sp_names,
                        fallback_sp_name))

        return result_dict

    def _unpack_setpoints_from_parameter(
        self, parameter: _BaseParameter, setpoints: Sequence,
        sp_names: Sequence[str], fallback_sp_name: str
        ) -> Dict[ParamSpecBase, values_type]:
        """
        Unpack the setpoints and their values from a parameter with setpoints
        into a standard results dict form and return that dict
        """
        setpoint_axes = []
        setpoint_parameters: List[ParamSpecBase] = []

        for i, sps in enumerate(setpoints):
            if sp_names is not None:
                spname = sp_names[i]
            else:
                spname = f'{fallback_sp_name}_{i}'

            try:
                setpoint_parameter = self._interdeps[spname]
            except KeyError:
                raise RuntimeError('No setpoints registered for '
                                   f'{type(parameter)} {parameter.full_name}!')
            sps = np.array(sps)
            while sps.ndim > 1:
                # The outermost setpoint axis or an nD param is nD
                # but the innermost is 1D. In all cases we just need
                # the axis along one dim, the innermost one.
                sps = sps[0]

            setpoint_parameters.append(setpoint_parameter)
            setpoint_axes.append(sps)

        output_grids = np.meshgrid(*setpoint_axes, indexing='ij')
        result_dict = {}
        for grid, param in zip(output_grids, setpoint_parameters):
            result_dict.update({param: grid})

        return result_dict

    def _validate_result_deps(
        self, results_dict: Dict[ParamSpecBase, values_type]) -> None:
        """
        Validate that the dependencies of the results_dict are met, meaning
        that (some) values for all required setpoints and inferences are
        present
        """
        try:
            self._interdeps.validate_subset(results_dict.keys())
        except (DependencyError, InferenceError) as err:
            raise ValueError('Can not add result, some required parameters '
                             'are missing.') from err

    def _validate_result_shapes(
        self, results_dict: Dict[ParamSpecBase, values_type]) -> None:
        """
        Validate that all sizes of the results_dict are consistent. This means
        that array-values of parameters and their setpoints are of the
        same size, whereas parameters with no setpoint relation to each other
        can have different sizes.
        """
        toplevel_params = (set(self._interdeps.dependencies)
                           .intersection(set(results_dict)))
        for toplevel_param in toplevel_params:
            required_shape = np.shape(results_dict[toplevel_param])
            for setpoint in self._interdeps.dependencies[toplevel_param]:
                # a setpoint is allowed to be a scalar; shape is then ()
                setpoint_shape = np.shape(results_dict[setpoint])
                if setpoint_shape not in [(), required_shape]:
                    raise ValueError(f'Incompatible shapes. Parameter '
                                     f"{toplevel_param.name} has shape "
                                     f"{required_shape}, but its setpoint "
                                     f"{setpoint.name} has shape "
                                     f"{setpoint_shape}.")

    def _validate_result_types(
        self, results_dict: Dict[ParamSpecBase, values_type]) -> None:
        """
        Validate the type of the results
        """

        def basic_type_validator(
            ps_name: str, vals: Any, expec_type: str) -> None:

            if type(vals) not in allowed_types[expec_type]:
                raise ValueError(f'Parameter {ps_name} is of type '
                                 f'"{expec_type}", but got a result of '
                                 f'type {type(vals)} ({vals}).')

        def array_type_validator(
            ps_name: str,
            vals: Union[np.ndarray, tuple, list],
            expec_dtype: str) -> None:

            if isinstance(vals, np.ndarray):
                if vals.dtype not in allowed_types[expec_dtype]:
                    raise ValueError(f'Parameter {ps_name} expects values of '
                                    f'type "{expec_dtype}", but got a result '
                                    f'of type {vals.dtype}.')
            else:
                seq_types = list(isinstance(val, allowed_types[expec_dtype])
                                 for val in vals)
                if not all(seq_types):
                    wrong_val = seq_types.index(False)
                    raise ValueError(f'Parameter {ps_name} expects values of '
                                    f'type "{expec_dtype}", but got a result '
                                    f'of type {type(wrong_val)}.')

        # Note that we allow 'numeric' input to be of type np.ndarray, if
        # the shape of the input is ()
        allowed_types = {'numeric': (int, float, np.int, np.int8,
                                     np.int16, np.int32, np.int64,
                                     np.float, np.float16, np.float32,
                                     np.float64, np.ndarray),
                         'text': (str,),
                         'array': (np.ndarray, tuple , list)}
        for ps, vals in results_dict.items():
            # we allow for 'numeric' and 'text' parameters to get results of
            # Sequence[scalar_type] or Sequence[str], so we must handle that as
            # well as results with the "correct" type
            if ps.type == 'numeric':
                if np.shape(vals) == ():
                    basic_type_validator(ps.name, vals, 'numeric')
                else:
                    array_type_validator(ps.name, np.array(vals), 'numeric')
            elif ps.type == 'text':
                if isinstance(vals, (list, tuple)):
                    array_type_validator(ps.name, vals, 'text')
                else:
                    basic_type_validator(ps.name, vals, 'text')
            elif ps.type == 'array':
                basic_type_validator(ps.name, vals, 'array')
                array_type_validator(ps.name, vals, 'numeric')

    def _enqueue_results(
        self, result_dict: Dict[ParamSpecBase, values_type]) -> None:
        """
        Enqueue the results into self._results

        Before we can enqueue the results, all values of the results dict
        must have the same length. We enqueue each parameter tree seperately,
        effectively mimicking making one call to add_result per parameter
        tree.

        Deal with 'numeric' type parameters. If a 'numeric' top level parameter
        has non-scalar shape, it must be unrolled into a list of
        """

        interdeps = self._interdeps

        toplevel_params = (set(interdeps.dependencies)
                           .intersection(set(result_dict)))
        for toplevel_param in toplevel_params:
            inff_params = set(interdeps.inferences.get(toplevel_param, ()))
            deps_params = set(interdeps.dependencies.get(toplevel_param, ()))
            all_params = (inff_params
                          .union(deps_params)
                          .union({toplevel_param}))
            res_dict: Dict[str, VALUE] = {}  # the dict to append to _results
            if toplevel_param.type == 'array':
                res_list = self._finalize_res_dict_array(
                    result_dict, all_params)
            elif toplevel_param.type == 'numeric':
                res_list = self._finalize_res_dict_numeric(
                               result_dict, toplevel_param,
                               inff_params, deps_params)
            else:
                res_dict = {ps.name: result_dict[ps] for ps in all_params}
                res_list = [res_dict]
            self._results += res_list

        # Finally, handle standalone parameters

        standalones = (set(interdeps.standalones)
                       .intersection(set(result_dict)))

        if standalones:
            stdln_dict = {st: result_dict[st] for st in standalones}
            self._results += self._finalize_res_dict_standalones(stdln_dict)

    @staticmethod
    def _finalize_res_dict_array(
        result_dict: Dict[ParamSpecBase, values_type],
        all_params: Set[ParamSpecBase]) -> List[Dict[str, VALUE]]:
        """
        Make a list of res_dicts out of the results for a 'array' type
        parameter. The results are assumed to already have been validated for
        type and shape
        """
        def reshaper(val):
            if np.shape(val) == () and isinstance(val, np.ndarray):
                return np.reshape(val, (1,))
            elif isinstance(val, (tuple, list)):
                return np.array(val)
            else:
                return val

        res_dict = {ps.name: reshaper(result_dict[ps]) for ps in all_params}

        return [res_dict]

    @staticmethod
    def _finalize_res_dict_numeric(
        result_dict: Dict[ParamSpecBase, values_type],
        toplevel_param: ParamSpecBase,
        inff_params: Set[ParamSpecBase],
        deps_params: Set[ParamSpecBase]) -> List[Dict[str, VALUE]]:
        """
        Make a res_dict in the format expected by DataSet.add_results out
        of the results for a 'numeric' type parameter. This includes
        replicating and unrolling values as needed and also handling the corner
        case of np.array(1) kind of values
        """

        def array_to_scalar(val):
            if isinstance(val, np.ndarray):
                return float(val)
            else:
                return val

        res_list: List[Dict[str, VALUE]] = []
        all_params = inff_params.union(deps_params).union({toplevel_param})

        toplevel_shape = np.shape(result_dict[toplevel_param])
        if toplevel_shape == ():
            # In the case of a scalar, life is reasonably simple
            res_list = [{ps.name: array_to_scalar(result_dict[ps])
                         for ps in all_params}]
        else:
            # We first massage all values into np.arrays of the same
            # shape
            flat_results = {}
            flat_results[toplevel_param] = np.array(result_dict[toplevel_param]).ravel()
            N = len(flat_results[toplevel_param])
            for dep in deps_params:
                if np.shape(result_dict[dep]) == ():
                    flat_results[dep] = np.repeat(result_dict[dep], N)
                else:
                    flat_results[dep] = np.array(result_dict[dep]).ravel()
            for inff in inff_params:
                if np.shape(result_dict[inff]) == ():
                    flat_results[inff] = np.repeat(result_dict[dep], N)
                else:
                    flat_results[inff] = np.array(result_dict[inff]).ravel()

            # And then put everything into the list

            res_list = [{p.name: flat_results[p][ind] for p in all_params}
                        for ind in range(N)]

        return res_list

    @staticmethod
    def _finalize_res_dict_standalones(
        result_dict: Dict[ParamSpecBase, values_type]
        ) -> List[Dict[str, VALUE]]:
        """
        Massage all standalone parameters into the correct shape
        """
        res_list = []
        for param, value in result_dict.items():
            if param.type == 'text':
                if isinstance(value, (tuple, list)):
                    res_list += [{param.name: string} for string in value]
                else:
                    res_list += [{param.name: value}]
            else:
                res_list += [{param.name: value}]

        return res_list

    def flush_data_to_database(self) -> None:
        """
        Write the in-memory results to the database.
        """
        log.debug('Flushing to database')
        if self._results != []:
            try:
                write_point = self._dataset.add_results(self._results)
                log.debug(f'Successfully wrote from index {write_point}')
                self._results = []
            except Exception as e:
                log.warning(f'Could not commit to database; {e}')
        else:
            log.debug('No results to flush')

    @property
    def run_id(self) -> int:
        return self._dataset.run_id

    @property
    def points_written(self) -> int:
        return self._dataset.number_of_results

    @property
    def dataset(self):
        return self._dataset


class Runner:
    """
    Context manager for the measurement.

    Lives inside a Measurement and should never be instantiated
    outside a Measurement.

    This context manager handles all the dirty business of writing data
    to the database. Additionally, it may perform experiment bootstrapping
    and clean-up after the measurement.
    """

    def __init__(
            self, enteractions: List, exitactions: List,
            experiment: Experiment = None, station: Station = None,
            write_period: numeric_types = None,
            interdeps: InterDependencies_ = InterDependencies_(),
            name: str = '',
            subscribers: Sequence[Tuple[Callable,
                                        Union[MutableSequence,
                                              MutableMapping]]] = None) -> None:

        self.enteractions = enteractions
        self.exitactions = exitactions
        self.subscribers: Sequence[Tuple[Callable,
                                         Union[MutableSequence,
                                               MutableMapping]]]
        if subscribers is None:
            self.subscribers = []
        else:
            self.subscribers = subscribers
        self.experiment = experiment
        self.station = station
        self._interdependencies = interdeps
        # here we use 5 s as a sane default, but that value should perhaps
        # be read from some config file
        self.write_period = float(write_period) \
            if write_period is not None else 5.0
        self.name = name if name else 'results'

    def __enter__(self) -> DataSaver:
        # TODO: should user actions really precede the dataset?
        # first do whatever bootstrapping the user specified
        for func, args in self.enteractions:
            func(*args)

        # next set up the "datasaver"
        if self.experiment is not None:
            self.ds = qc.new_data_set(
                self.name, self.experiment.exp_id, conn=self.experiment.conn
            )
        else:
            self.ds = qc.new_data_set(self.name)

        # .. and give the dataset a snapshot as metadata
        if self.station is None:
            station = qc.Station.default
        else:
            station = self.station

        if station:
            self.ds.add_snapshot(json.dumps({'station': station.snapshot()},
                                            cls=NumpyJSONEncoder))

        if self._interdependencies == InterDependencies_():
            raise RuntimeError("No parameters supplied")
        else:
            self.ds.set_interdependencies(self._interdependencies)

        self.ds.mark_started()

        # register all subscribers
        for (callble, state) in self.subscribers:
            # We register with minimal waiting time.
            # That should make all subscribers be called when data is flushed
            # to the database
            log.debug(f'Subscribing callable {callble} with state {state}')
            self.ds.subscribe(callble, min_wait=0, min_count=1, state=state)

        print(f'Starting experimental run with id: {self.ds.run_id}')

        self.datasaver = DataSaver(dataset=self.ds,
                                   write_period=self.write_period,
                                   interdeps=self._interdependencies)

        return self.datasaver

    def __exit__(self, exception_type, exception_value, traceback) -> None:

        self.datasaver.flush_data_to_database()

        # perform the "teardown" events
        for func, args in self.exitactions:
            func(*args)

        # and finally mark the dataset as closed, thus
        # finishing the measurement
        self.ds.mark_completed()

        self.ds.unsubscribe_all()



T = TypeVar('T', bound='Measurement')
class Measurement:
    """
    Measurement procedure container

    Args:
        exp: Specify the experiment to use. If not given
            the default one is used.
        station: The QCoDeS station to snapshot. If not given, the
            default one is used.
    """

    def __init__(self, exp: Optional[Experiment] = None,
                 station: Optional[qc.Station] = None) -> None:
        self.exitactions: List[Tuple[Callable, Sequence]] = []
        self.enteractions: List[Tuple[Callable, Sequence]] = []
        self.subscribers: List[Tuple[Callable, Union[MutableSequence,
                                                     MutableMapping]]] = []
        self.experiment = exp
        self.station = station
        # self.parameters: Dict[str, ParamSpec] = OrderedDict()
        self._write_period: Optional[float] = None
        self.name = ''
        self._interdeps = InterDependencies_()

    @property
    def parameters(self) -> Dict[str, ParamSpecBase]:
        return deepcopy(self._interdeps._id_to_paramspec)

    @property
    def write_period(self) -> Optional[float]:
        return self._write_period

    @write_period.setter
    def write_period(self, wp: numeric_types) -> None:
        if not isinstance(wp, Number):
            raise ValueError('The write period must be a number (of seconds).')
        wp_float = float(wp)
        if wp_float < 1e-3:
            raise ValueError('The write period must be at least 1 ms.')
        self._write_period = wp_float

    def _paramspecbase_from_strings(
            self, name: str, setpoints: Sequence[str] = None,
            basis: Sequence[str] = None
            ) -> Tuple[Tuple[ParamSpecBase, ...], Tuple[ParamSpecBase, ...]]:
        """
        Helper function to look up and get ParamSpecBases and to give a nice
        error message if the user tries to register a parameter with reference
        (setpoints, basis) to a parameter not registered with this measurement

        Called by _register_parameter only.

        Args:
            name: Name of the parameter to register
            setpoints: name(s) of the setpoint parameter(s)
            basis: name(s) of the parameter(s) that this parameter is
                inferred from
        """

        idps = self._interdeps

        # now handle setpoints
        depends_on = []
        if setpoints:
            for sp in setpoints:
                try:
                    sp_psb = idps._id_to_paramspec[sp]
                    depends_on.append(sp_psb)
                except KeyError:
                    raise ValueError(f'Unknown setpoint: {sp}.'
                                     ' Please register that parameter first.')

        # now handle inferred parameters
        inf_from = []
        if basis:
            for inff in basis:
                try:
                    inff_psb = idps._id_to_paramspec[inff]
                    inf_from.append(inff_psb)
                except KeyError:
                    raise ValueError(f'Unknown basis parameter: {inff}.'
                                     ' Please register that parameter first.')

        return tuple(depends_on), tuple(inf_from)

    def register_parameter(
            self: T, parameter: _BaseParameter,
            setpoints: setpoints_type = None,
            basis: setpoints_type = None,
            paramtype: str = 'numeric') -> T:
        """
        Add QCoDeS Parameter to the dataset produced by running this
        measurement.

        Args:
            parameter: The parameter to add
            setpoints: The Parameter representing the setpoints for this
                parameter. If this parameter is a setpoint,
                it should be left blank
            basis: The parameters that this parameter is inferred from. If
                this parameter is not inferred from any other parameters,
                this should be left blank.
            paramtype: type of the parameter, i.e. the SQL storage class
        """
        # input validation
        if paramtype not in ParamSpec.allowed_types:
            raise RuntimeError("Trying to register a parameter with type "
                               f"{paramtype}. However, only "
                               f"{ParamSpec.allowed_types} are supported.")
        if not isinstance(parameter, _BaseParameter):
            raise ValueError('Can not register object of type {}. Can only '
                             'register a QCoDeS Parameter.'
                             ''.format(type(parameter)))
        # perhaps users will want a different name? But the name must be unique
        # on a per-run basis
        # we also use the name below, but perhaps is is better to have
        # a more robust Parameter2String function?
        name = str(parameter)
        if isinstance(parameter, ArrayParameter):
            self._register_arrayparameter(parameter,
                                          setpoints,
                                          basis,
                                          paramtype)
        elif isinstance(parameter, ParameterWithSetpoints):
            self._register_parameter_with_setpoints(parameter,
                                                    setpoints,
                                                    basis,
                                                    paramtype)
        elif isinstance(parameter, MultiParameter):
            self._register_multiparameter(parameter,
                                          setpoints,
                                          basis,
                                          paramtype,
                                          )
        elif isinstance(parameter, Parameter):
            self._register_parameter(name,
                                     parameter.label,
                                     parameter.unit,
                                     setpoints,
                                     basis, paramtype)
        else:
            raise RuntimeError("Does not know how to register a parameter"
                               f"of type {type(parameter)}")

        return self

    def _register_parameter(self: T, name: str,
                            label: Optional[str],
                            unit: Optional[str],
                            setpoints: Optional[setpoints_type],
                            basis: Optional[setpoints_type],
                            paramtype: str) -> T:
        """
        Update the interdependencies object with a new group
        """

        try:
            parameter = self._interdeps[name]
        except KeyError:
            parameter = None
            pass

        paramspec = ParamSpecBase(name=name,
                                  paramtype=paramtype,
                                  label=label,
                                  unit=unit)

        # We want to allow the registration of the exact same parameter twice,
        # the reason being that e.g. two ArrayParameters could share the same
        # setpoint parameter, which would then be registered along with each
        # dependent (array)parameter

        if parameter is not None and parameter != paramspec:
            raise ValueError("Parameter already registered "
                             "in this Measurement.")

        if setpoints is not None:
            sp_strings = [str(sp) for sp in setpoints]
        else:
            sp_strings = []

        if basis is not None:
            bs_strings = [str(bs) for bs in basis]
        else:
            bs_strings = []

        # get the ParamSpecBases
        depends_on, inf_from = self._paramspecbase_from_strings(name,
                                                                sp_strings,
                                                                bs_strings)

        if depends_on:
            self._interdeps = self._interdeps.extend(
                                  dependencies={paramspec: depends_on})
        if inf_from:
            self._interdeps = self._interdeps.extend(
                                  inferences={paramspec: inf_from})
        if not(depends_on or inf_from):
            self._interdeps = self._interdeps.extend(standalones=(paramspec,))

        log.info(f'Registered {name} in the Measurement.')

        return self

    def _register_arrayparameter(self,
                                 parameter: ArrayParameter,
                                 setpoints: Optional[setpoints_type],
                                 basis: Optional[setpoints_type],
                                 paramtype: str, ) -> None:
        """
        Register an ArrayParameter and the setpoints belonging to that
        ArrayParameter
        """
        name = str(parameter)
        my_setpoints = list(setpoints) if setpoints else []
        for i in range(len(parameter.shape)):
            if parameter.setpoint_full_names is not None and \
                    parameter.setpoint_full_names[i] is not None:
                spname = parameter.setpoint_full_names[i]
            else:
                spname = f'{name}_setpoint_{i}'
            if parameter.setpoint_labels:
                splabel = parameter.setpoint_labels[i]
            else:
                splabel = ''
            if parameter.setpoint_units:
                spunit = parameter.setpoint_units[i]
            else:
                spunit = ''

            self._register_parameter(name=spname,
                                     paramtype=paramtype,
                                     label=splabel,
                                     unit=spunit,
                                     setpoints=None,
                                     basis=None)

            my_setpoints += [spname]

        self._register_parameter(name,
                                 parameter.label,
                                 parameter.unit,
                                 my_setpoints,
                                 basis,
                                 paramtype)

    def _register_parameter_with_setpoints(self,
                                           parameter: ParameterWithSetpoints,
                                           setpoints: Optional[setpoints_type],
                                           basis: Optional[setpoints_type],
                                           paramtype: str) -> None:
        """
        Register an ParameterWithSetpoints and the setpoints belonging to the
        Parameter
        """
        name = str(parameter)
        my_setpoints = list(setpoints) if setpoints else []
        for sp in parameter.setpoints:
            if not isinstance(sp, Parameter):
                raise RuntimeError("The setpoints of a "
                                   "ParameterWithSetpoints "
                                   "must be a Parameter")
            spname = sp.full_name
            splabel = sp.label
            spunit = sp.unit

            self._register_parameter(name=spname,
                                     paramtype=paramtype,
                                     label=splabel,
                                     unit=spunit,
                                     setpoints=None,
                                     basis=None)

            my_setpoints.append(spname)

        self._register_parameter(name,
                                 parameter.label,
                                 parameter.unit,
                                 my_setpoints,
                                 basis,
                                 paramtype)

    def _register_multiparameter(self,
                                 multiparameter: MultiParameter,
                                 setpoints: Optional[setpoints_type],
                                 basis: Optional[setpoints_type],
                                 paramtype: str) -> None:
        """
        Find the individual multiparameter components and their setpoints
        and register those as individual parameters
        """
        setpoints_lists = []
        for i in range(len(multiparameter.shapes)):
            shape = multiparameter.shapes[i]
            name = multiparameter.full_names[i]
            if shape is ():
                my_setpoints = setpoints
            else:
                my_setpoints = list(setpoints) if setpoints else []
                for j in range(len(shape)):
                    if multiparameter.setpoint_full_names is not None and \
                            multiparameter.setpoint_full_names[i] is not None:
                        spname = multiparameter.setpoint_full_names[i][j]
                    else:
                        spname = f'{name}_setpoint_{j}'
                    if multiparameter.setpoint_labels is not None and \
                            multiparameter.setpoint_labels[i] is not None:
                        splabel = multiparameter.setpoint_labels[i][j]
                    else:
                        splabel = ''
                    if multiparameter.setpoint_units is not None and \
                            multiparameter.setpoint_units[i] is not None:
                        spunit = multiparameter.setpoint_units[i][j]
                    else:
                        spunit = ''

                    self._register_parameter(name=spname,
                                             paramtype=paramtype,
                                             label=splabel,
                                             unit=spunit,
                                             setpoints=None,
                                             basis=None)

                    my_setpoints += [spname]

            setpoints_lists.append(my_setpoints)

        for i, setpoints in enumerate(setpoints_lists):
            self._register_parameter(multiparameter.names[i],
                                     multiparameter.labels[i],
                                     multiparameter.units[i],
                                     setpoints,
                                     basis,
                                     paramtype)

    def register_custom_parameter(
            self : T, name: str,
            label: str = None, unit: str = None,
            basis: setpoints_type = None,
            setpoints: setpoints_type = None,
            paramtype: str = 'numeric') -> T:
        """
        Register a custom parameter with this measurement

        Args:
            name: The name that this parameter will have in the dataset. Must
                be unique (will overwrite an existing parameter with the same
                name!)
            label: The label
            unit: The unit
            basis: A list of either QCoDeS Parameters or the names
                of parameters already registered in the measurement that
                this parameter is inferred from
            setpoints: A list of either QCoDeS Parameters or the names of
                of parameters already registered in the measurement that
                are the setpoints of this parameter
            paramtype: type of the parameter, i.e. the SQL storage class
        """
        return self._register_parameter(name,
                                        label,
                                        unit,
                                        setpoints,
                                        basis,
                                        paramtype)

    def unregister_parameter(self,
                             parameter: setpoints_type) -> None:
        """
        Remove a custom/QCoDeS parameter from the dataset produced by
        running this measurement
        """
        if isinstance(parameter, _BaseParameter):
            param = str(parameter)
        elif isinstance(parameter, str):
            param = parameter
        else:
            raise ValueError('Wrong input type. Must be a QCoDeS parameter or'
                             ' the name (a string) of a parameter.')

        try:
            paramspec: ParamSpecBase = self._interdeps[param]
        except KeyError:
            return

        self._interdeps = self._interdeps.remove(paramspec)

        log.info(f'Removed {param} from Measurement.')

    def add_before_run(self : T, func: Callable, args: tuple) -> T:
        """
        Add an action to be performed before the measurement.

        Args:
            func: Function to be performed
            args: The arguments to said function
        """
        # some tentative cheap checking
        nargs = len(signature(func).parameters)
        if len(args) != nargs:
            raise ValueError('Mismatch between function call signature and '
                             'the provided arguments.')

        self.enteractions.append((func, args))

        return self

    def add_after_run(self : T, func: Callable, args: tuple) -> T:
        """
        Add an action to be performed after the measurement.

        Args:
            func: Function to be performed
            args: The arguments to said function
        """
        # some tentative cheap checking
        nargs = len(signature(func).parameters)
        if len(args) != nargs:
            raise ValueError('Mismatch between function call signature and '
                             'the provided arguments.')

        self.exitactions.append((func, args))

        return self

    def add_subscriber(self : T,
                       func: Callable,
                       state: Union[MutableSequence, MutableMapping]) -> T:
        """
        Add a subscriber to the dataset of the measurement.

        Args:
            func: A function taking three positional arguments: a list of
                tuples of parameter values, an integer, a mutable variable
                (list or dict) to hold state/writes updates to.
            state: The variable to hold the state.
        """
        self.subscribers.append((func, state))

        return self

    def run(self) -> Runner:
        """
        Returns the context manager for the experimental run
        """
        return Runner(self.enteractions, self.exitactions,
                      self.experiment, station=self.station,
                      write_period=self._write_period,
                      interdeps=self._interdeps,
                      name=self.name,
                      subscribers=self.subscribers)
