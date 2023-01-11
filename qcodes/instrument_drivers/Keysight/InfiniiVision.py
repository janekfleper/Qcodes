from typing import Any

import numpy as np

from qcodes import validators as vals
from qcodes.instrument import VisaInstrument
from qcodes.instrument.channel import ChannelList, InstrumentChannel, InstrumentModule
from qcodes.instrument.function import Function
from qcodes.instrument.parameter import Parameter
from qcodes.utils.helpers import create_on_off_val_mapping

CHANNEL_COUNT = {
    "EDUX1052A": 2,
    "EDUX1052G": 2,
    "DSOX1202A": 2,
    "DSOX1202G": 2,
    "DSOX1204A": 4,
    "DSOX1204G": 4,
}

WAVEFORM_FORMAT = {0: "byte", 1: "word", 4: "ascii"}
ACQUISITION_TYPE = {0: "normal", 1: "peak", 2: "average", 3: "hresolution"}


def interpret_preamble(preamble: str):
    args = preamble.split(",")
    return {
        "waveform_format": WAVEFORM_FORMAT[int(args[0])],
        "acquisition_type": ACQUISITION_TYPE[int(args[1])],
        "points": int(args[2]),
        "averages": int(args[3]),
        "dt": float(args[4]),
        "t0": float(args[5]),
        "tref": int(args[6]),
        "dy": float(args[7]),
        "y0": float(args[8]),
        "yref": int(args[9]),
    }


class Acquire(InstrumentModule):
    def __init__(self, parent: "InfiniiVision", name: str, **kwargs: Any):
        super().__init__(parent, name, **kwargs)

        self.complete = Parameter(
            name="complete",
            instrument=self,
            set_cmd=False,
            get_cmd=":acquire:complete?",
            get_parser=lambda x: bool(int(x) // 100),
            docstring="Acquisition completion status",
        )

        self.count = Parameter(
            name="count",
            instrument=self,
            set_cmd=f":acquire:count {{}}",
            get_cmd=":acquire:count?",
            vals=vals.Ints(2, 65536),
            get_parser=int,
            docstring="Acquisition count in average mode",
        )

        self.mode = Parameter(
            name="mode",
            instrument=self,
            set_cmd=f":acquire:mode {{}}",
            get_cmd=":acquire:mode?",
            vals=vals.Enum("rtime", "segmented"),
            docstring="Acquisition mode",
        )

        self.points = Parameter(
            name="points",
            instrument=self,
            set_cmd=None,
            get_cmd=":acquire:points?",
            get_parser=int,
            docstring="Number of acquired points",
        )

        self.srate = Parameter(
            name="srate",
            instrument=self,
            set_cmd=None,
            get_cmd=":acquire:srate?",
            get_parser=float,
            docstring="Sampling rate",
        )

        self.type = Parameter(
            name="type",
            instrument=self,
            set_cmd=f":acquire:type {{}}",
            get_cmd=":acquire:type?",
            vals=vals.Enum("normal", "average", "hresolution", "peak"),
            docstring="Acquisition type",
        )


class Waveform(InstrumentModule):
    def __init__(self, parent: "InfiniiVision", name: str, **kwargs: Any):
        super().__init__(parent, name, **kwargs)

        self.byteorder = Parameter(
            name="byteorder",
            instrument=self,
            set_cmd=f":waveform:byteorder {{}}",
            get_cmd=":waveform:byteorder?",
            vals=vals.Enum("lsbfirst", "msbfirst"),
            docstring="Byte order of WORD data",
        )

        self.count = Parameter(
            name="count",
            instrument=self,
            set_cmd=None,
            get_cmd=":waveform:count?",
            get_parser=int,
            docstring="Count used for the acquired waveform",
        )

        self.data = Function(name="data", instrument=self, call_cmd=":waveform:data?")

        self.format = Parameter(
            name="format",
            instrument=self,
            set_cmd=f":waveform:format {{}}",
            get_cmd=":waveform:format?",
            vals=vals.Enum("word", "byte", "ascii"),
            docstring="Data transmission mode for waveform data points",
        )

        self.points = Parameter(
            name="points",
            instrument=self,
            set_cmd=f":waveform:points {{}}",
            get_cmd=":waveform:points?",
            vals=vals.Ints(100, 2000000),
            get_parser=int,
            docstring="Number of points of the waveform",
        )

        self.points_mode = Parameter(
            name="points_mode",
            instrument=self,
            set_cmd=f":waveform:points:mode {{}}",
            get_cmd=":waveform:points:mode?",
            vals=vals.Enum("normal", "maximum", "raw"),
        )

        self.preamble = Parameter(
            name="preamble",
            instrument=self,
            set_cmd=False,
            get_cmd=":waveform:preamble?",
            docstring="Read the waveform properties",
        )

        self.source = Parameter(
            name="source",
            instrument=self,
            set_cmd=f":waveform:source {{}}",
            get_cmd=":waveform:source?",
            vals=vals.Strings(),
            docstring="Source for the waveform",
        )


class Channel(InstrumentChannel):
    def __init__(self, parent: "InfiniiVision", name: str, channel: int, **kwargs: Any):
        self._channel = channel
        super().__init__(parent, name, **kwargs)

        self.bandwidth = Parameter(
            name="bandwidth",
            instrument=self,
            set_cmd=f":channel{channel}:bandwidth {{}}",
            get_cmd=f":channel{channel}:bandwidth?",
            vals=vals.Enum(25e6),
            get_parser=float,
            docstring=f"Channel {channel} bandwidth",
        )

        self.bwlimit = Parameter(
            name="bwlimit",
            instrument=self,
            set_cmd=f":channel{channel}:bwlimit {{}}",
            get_cmd=f":channel{channel}:bwlimit?",
            val_mapping=create_on_off_val_mapping(on_val=1, off_val=0),
            docstring=f"Enable channel {channel} bandwidth limit",
        )

        self.coupling = Parameter(
            name="coupling",
            instrument=self,
            set_cmd=f":channel{channel}:coupling {{}}",
            get_cmd=f":channel{channel}:coupling?",
            vals=vals.Enum("AC", "DC"),
            docstring=f"Channel {channel} input coupling",
        )

        self.display = Parameter(
            name="display",
            instrument=self,
            set_cmd=f":channel{channel}:display {{}}",
            get_cmd=f":channel{channel}:display?",
            val_mapping=create_on_off_val_mapping(on_val=1, off_val=0),
            docstring=f"Enable/disable display of channel {channel}",
        )

        self.impedance = Parameter(
            name="impedance",
            instrument=self,
            set_cmd=f":channel{channel}:impedance {{}}",
            get_cmd=f":channel{channel}:impedance?",
            vals=vals.Enum("onemeg"),
            docstring=f"Channel {channel} input impedance",
        )

        self.invert = Parameter(
            name="invert",
            instrument=self,
            set_cmd=f":channel{channel}:invert {{}}",
            get_cmd=f":channel{channel}:invert?",
            val_mapping=create_on_off_val_mapping(on_val=1, off_val=0),
            docstring=f"Invert channel {channel}",
        )

        self.label = Parameter(
            name="label",
            instrument=self,
            set_cmd=f":channel{channel}:label {{}}",
            get_cmd=f":channel{channel}:label?",
            vals=vals.Strings(max_length=10),
            docstring=f"Channel {channel} label",
        )

        self.offset = Parameter(
            name="offset",
            instrument=self,
            set_cmd=f":channel{channel}:offset {{}}",
            get_cmd=f":channel{channel}:offset?",
            vals=vals.MultiType(vals.Numbers(), vals.Strings()),
            get_parser=float,
            docstring=f"Channel {channel} voltage offset",
        )

        self.range = Parameter(
            name="range",
            instrument=self,
            set_cmd=f":channel{channel}:range {{}}",
            get_cmd=f":channel{channel}:range?",
            vals=vals.MultiType(vals.Numbers(), vals.Strings()),
            get_parser=float,
            docstring=f"Channel {channel} full-scale range",
        )

        self.scale = Parameter(
            name="scale",
            instrument=self,
            set_cmd=f":channel{channel}:scale {{}}",
            get_cmd=f":channel{channel}:scale?",
            vals=vals.MultiType(vals.Numbers(), vals.Strings()),
            get_parser=float,
            docstring=f"Set the vertical scale (unit per division)",
        )

        self.unit = Parameter(
            name="unit",
            instrument=self,
            set_cmd=f":channel{channel}:unit {{}}",
            get_cmd=f":channel{channel}:unit",
            vals=vals.Enum("volt", "ampere"),
            label=f"Select the measurement unit for the connected probe",
        )

        self.vernier = Parameter(
            name="vernier",
            instrument=self,
            set_cmd=f":channel{channel}:vernier {{}}",
            get_cmd=f":channel{channel}:vernier?",
            val_mapping=create_on_off_val_mapping(on_val=1, off_val=0),
            docstring=f"Enable/disable the vernier (fine vertical adjustment)",
        )

    def get_data(self):
        """
        Read the time axis and trace from the oscilloscope

        The acquisition must be completed before running this method.
        """
        waveform = self.root_instrument.waveform
        waveform.source(f"channel{self._channel}")
        waveform.data()

        data = self.root_instrument.visa_handle.read_binary_values(
            datatype="H",
            is_big_endian=True,
            container=np.ndarray,
            header_fmt="ieee",
            expect_termination=True,
        ).astype(np.int32)

        header = interpret_preamble(waveform.preamble())
        n_points = header["points"]
        t = np.linspace(
            header["t0"],
            header["t0"] + n_points * header["dt"],
            n_points,
            endpoint=False,
        )
        y = (data - header["yref"]) * header["dy"] + header["y0"]
        return t, y


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

        self.model = self.IDN()["model"]
        self.n_channels = CHANNEL_COUNT[self.model]

        self.digitize = Function(name="digitize", instrument=self, call_cmd=":digitize")

        self.run = Function(name="run", instrument=self, call_cmd=":run")

        self.single = Function(name="single", instrument=self, call_cmd=":single")

        self.stop = Function(name="stop", instrument=self, call_cmd=":stop")

        self.recall = Parameter(
            name="recall",
            instrument=self,
            set_cmd="*rcl {}",
            get_cmd=False,
            vals=vals.Ints(0, 9),
            docstring=f"Restore a saved instrument state",
        )

        self.save = Parameter(
            name="save",
            instrument=self,
            set_cmd="*sav {}",
            get_cmd=False,
            vals=vals.Ints(0, 9),
            docstring=f"Save the current state of the instrument",
        )

        _channels = ChannelList(self, "channels", Channel, snapshotable=False)
        for i in range(1, self.n_channels + 1):
            channel = Channel(self, f"channel{i}", i)
            _channels.append(channel)
            self.add_submodule(f"ch{i}", channel)
        self.add_submodule("channels", _channels.to_channel_tuple())

        self.add_submodule("acquire", Acquire(self, "acquire"))
        self.add_submodule("waveform", Waveform(self, "waveform"))
