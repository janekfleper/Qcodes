from typing import Sequence, Tuple, Any, Optional

import numpy as np

from qcodes import validators as vals
from qcodes.instrument import VisaInstrument
from qcodes.instrument.channel import ChannelList, InstrumentChannel, InstrumentModule
from qcodes.instrument.function import Function
from qcodes.instrument.parameter import Parameter
from qcodes.utils.helpers import create_on_off_val_mapping

CHANNEL_COUNT = {
    'EDUX1052A': 2, 'EDUX1052G': 2,
    'DSOX1202A': 2, 'DSOX1202G': 2,
    'DSOX1204A': 4, 'DSOX1204G': 4,
}

WAVEFORM_FORMAT = {0:'byte', 1:'word', 4:'ascii'}
ACQUISITION_TYPE = {0:'normal', 1:'peak', 2:'average', 3:'hresolution'}

def interpret_preamble(preamble):
    args = preamble.split(',')
    return {
        'waveform_format': WAVEFORM_FORMAT[int(args[0])],
        'acquisition_type': ACQUISITION_TYPE[int(args[1])],
        'points': int(args[2]), 'averages': int(args[3]),
        'dt': float(args[4]), 't0': float(args[5]), 'tref': int(args[6]),
        'dy': float(args[7]), 'y0': float(args[8]), 'yref': int(args[9]),
    }

class Acquire(InstrumentModule):
    def __init__(self, parent: 'InfiniiVision', name: str, **kwargs: Any):
        super().__init__(parent, name, **kwargs)

        self.complete = Parameter(
            name="complete",
            instrument=self,
            label="Acquisition completion status",
            set_cmd=False,
            get_cmd=":acquire:complete?",
            get_parser=lambda x: bool(int(x)//100),
        )

        self.count = Parameter(
            name="count",
            instrument=self,
            label="Acquisition count in average mode",
            set_cmd=f":acquire:count {{}}",
            get_cmd=":acquire:count?",
            vals=vals.Ints(2, 65536),
            get_parser=int,
        )

        self.mode = Parameter(
            name="mode",
            instrument=self,
            label="Acquisition mode",
            set_cmd=f":acquire:mode {{}}",
            get_cmd=":acquire:mode?",
            vals=vals.Enum("rtime", "segmented"),
        )

        self.points = Parameter(
            name="points",
            instrument=self,
            label="Number of acquired points",
            set_cmd=None,
            get_cmd=":acquire:points?",
            get_parser=int,
        )

        self.srate = Parameter(
            name="srate",
            instrument=self,
            label="Sampling rate",
            set_cmd=None,
            get_cmd=":acquire:srate?",
            get_parser=float,
        )

        self.type = Parameter(
            name="type",
            instrument=self,
            label="Acquisition type",
            set_cmd=f":acquire:type {{}}",
            get_cmd=":acquire:type?",
            vals=vals.Enum("normal", "average", "hresolution", "peak"),
        )

class Waveform(InstrumentModule):
    def __init__(self, parent: 'InfiniiVision', name: str, **kwargs: Any):
        super().__init__(parent, name, **kwargs)

        self.byteorder = Parameter(
            name="byteorder",
            instrument=self,
            label="Byte order of WORD data",
            set_cmd=f":waveform:byteorder {{}}",
            get_cmd=":waveform:byteorder?",
            vals=vals.Enum("lsbfirst", "msbfirst"),
        )

        self.count = Parameter(
            name="count",
            instrument=self,
            label="Count used for the acquired waveform",
            set_cmd=None,
            get_cmd=":waveform:count?",
            get_parser=int,
        )

        self.data = Function(
            name="data",
            instrument=self,
            call_cmd=":waveform:data?"
        )

        self.format = Parameter(
            name="format",
            instrument=self,
            label="Data transmission mode for waveform data points",
            set_cmd=f":waveform:format {{}}",
            get_cmd=":waveform:format?",
            vals=vals.Enum("word", "byte", "ascii"),
        )

        self.points = Parameter(
            name="points",
            instrument=self,
            label="Number of points of the waveform",
            set_cmd=f":waveform:points {{}}",
            get_cmd=":waveform:points?",
            vals=vals.Ints(100, 2000000),
            get_parser=int,
        )

        self.points_mode = Parameter(
            name="points_mode",
            instrument=self,
            label="???",
            set_cmd=f":waveform:points:mode {{}}",
            get_cmd=":waveform:points:mode?",
            vals=vals.Enum("normal", "maximum", "raw"),
        )

        self.preamble = Parameter(
            name="preamble",
            instrument=self,
            label="Read the waveform properties",
            set_cmd=False,
            get_cmd=":waveform:preamble?",
        )

        self.source = Parameter(
            name="source",
            instrument=self,
            label="Source for the waveform",
            set_cmd=f":waveform:source {{}}",
            get_cmd=":waveform:source?",
            vals=vals.Strings()
        )

    def header(self):
        """
        Interpret the preamble holding the waveform properties

        Returns
        -------
        dict
        """
        return interpret_preamble(self.preamble())

class Channel(InstrumentChannel):
    def __init__(self, parent: 'InfiniiVision', name: str, channel: int, **kwargs: Any):
        self._channel = channel
        super().__init__(parent, name, **kwargs)

        self.bandwidth = Parameter(
            name="bandwidth",
            instrument=self,
            label=f"Channel {channel} bandwidth",
            set_cmd=f":channel{channel}:bandwidth {{}}",
            get_cmd=f":channel{channel}:bandwidth?",
            vals=vals.Enum(25e6),
            get_parser=float,
        )

        self.bwlimit = Parameter(
            name="bwlimit",
            instrument=self,
            label=f"Enable channel {channel} bandwidth limit",
            set_cmd=f":channel{channel}:bwlimit {{}}",
            get_cmd=f":channel{channel}:bwlimit?",
            val_mapping=create_on_off_val_mapping(on_val=1, off_val=0),
        )

        self.coupling = Parameter(
            name="coupling",
            instrument=self,
            label=f"Channel {channel} input coupling",
            set_cmd=f":channel{channel}:coupling {{}}",
            get_cmd=f":channel{channel}:coupling?",
            vals=vals.Enum("AC", "DC"),
        )

        self.display = Parameter(
            name="display",
            instrument=self,
            label=f"Display channel {channel}",
            set_cmd=f":channel{channel}:display {{}}",
            get_cmd=f":channel{channel}:display?",
            val_mapping=create_on_off_val_mapping(on_val=1, off_val=0),
        )

        self.impedance = Parameter(
            name="impedance",
            instrument=self,
            label=f"Channel {channel} input impedance",
            set_cmd=f":channel{channel}:impedance {{}}",
            get_cmd=f":channel{channel}:impedance?",
            vals=vals.Enum("onemeg"),
        )

        self.invert = Parameter(
            name="invert",
            instrument=self,
            label=f"Invert channel {channel}",
            set_cmd=f":channel{channel}:invert {{}}",
            get_cmd=f":channel{channel}:invert?",
            val_mapping=create_on_off_val_mapping(on_val=1, off_val=0),
        )

        self.label = Parameter(
            name="label",
            instrument=self,
            label=f"Channel {channel} label",
            set_cmd=f":channel{channel}:label {{}}",
            get_cmd=f":channel{channel}:label?",
            vals=vals.Strings(max_length=10),
        )

        self.offset = Parameter(
            name="offset",
            instrument=self,
            label=f"Channel {channel} offset",
            set_cmd=f":channel{channel}:offset {{}}",
            get_cmd=f":channel{channel}:offset?",
            vals=vals.MultiType(vals.Numbers(), vals.Strings()),
            get_parser=float,
        )

        self.range = Parameter(
            name="range",
            instrument=self,
            label=f"Channel {channel} full-scale range",
            set_cmd=f":channel{channel}:range {{}}",
            get_cmd=f":channel{channel}:range?",
            vals=vals.MultiType(vals.Numbers(), vals.Strings()),
            get_parser=float,
        )

        self.scale = Parameter(
            name="scale",
            instrument=self,
            label=f"Channel {channel} vertical scale (unit per division)",
            set_cmd=f":channel{channel}:scale {{}}",
            get_cmd=f":channel{channel}:scale?",
            vals=vals.MultiType(vals.Numbers(), vals.Strings()),
            get_parser=float,
        )

        self.unit = Parameter(
            name="unit",
            instrument=self,
            label=f"Channel {channel} measurement unit",
            set_cmd=f":channel{channel}:unit {{}}",
            get_cmd=f":channel{channel}:unit",
            vals=vals.Enum("volt", "ampere"),
        )

        self.vernier = Parameter(
            name="vernier",
            instrument=self,
            label=f"Channel {channel} vernier (fine adjustment)",
            set_cmd=f":channel{channel}:vernier {{}}",
            get_cmd=f":channel{channel}:vernier?",
            val_mapping=create_on_off_val_mapping(on_val=1, off_val=0),
        )

    def read(self, raw=False):
        complete = self.parent.acquire.complete()
        if not complete:
            return None

        self.parent.waveform.source(f"channel{self._channel}")
        self.parent.waveform.data()
        params = dict(datatype='H', is_big_endian=True, container=np.ndarray,
                      header_fmt='ieee', expect_termination=True)
        data = self.root_instrument.visa_handle.read_binary_values(**params).astype(np.int32)
        if raw:
            return data

        header = self.parent.waveform.header()
        points = header['points']
        x = np.linspace(0, points*header['dt'], points, endpoint=False) + header['t0']
        y = (data - header['yref']) * header['dy'] + header['y0']
        return y, x, header


class InfiniiVision(VisaInstrument):
    """
    This is the QCoDeS driver for the Keysight InfiniiVision oscilloscopes
    """

    def __init__(
        self,
        name: str,
        address: str,
        timeout: float = 20,
        silence_pyvisapy_warning: bool = False,
        **kwargs: Any,
    ):
        """
        Initialises the oscilloscope.
        Args:
            name: Name of the instrument used by QCoDeS
            address: Instrument address as used by VISA
            timeout: Visa timeout, in secs.
            channels: The number of channels on the scope.
            silence_pyvisapy_warning: Don't warn about pyvisa-py at startup
        """
        super().__init__(name, address, timeout=timeout, terminator="\n", **kwargs)
        self.connect_message()

        self.model = self.IDN()['model']
        self.n_channels = CHANNEL_COUNT[self.model]

        self.digitize = Function(
            name="digitize",
            instrument=self,
            call_cmd=":digitize"
        )

        self.run = Function(
            name="run",
            instrument=self,
            call_cmd=":run"
        )

        self.single = Function(
            name="single",
            instrument=self,
            call_cmd=":single"
        )

        self.stop = Function(
            name="stop",
            instrument=self,
            call_cmd=":stop"
        )

        self.recall = Parameter(
            name="recall",
            instrument=self,
            label=f"Restore a saved instrument state",
            set_cmd="*rcl {}",
            get_cmd=False,
            vals=vals.Ints(0, 9),
        )

        self.save = Parameter(
            name="save",
            instrument=self,
            label=f"Save the current state of the instrument",
            set_cmd="*sav {}",
            get_cmd=False,
            vals=vals.Ints(0, 9),
        )

        _channels = ChannelList(self, "channels", Channel, snapshotable=False)
        for i in range(1, self.n_channels + 1):
            channel = Channel(self, f"channel{i}", i)
            _channels.append(channel)
            self.add_submodule(f"ch{i}", channel)
        self.add_submodule("channels", _channels.to_channel_tuple())

        self.add_submodule("acquire", Acquire(self, "acquire"))
        self.add_submodule("waveform", Waveform(self, "waveform"))