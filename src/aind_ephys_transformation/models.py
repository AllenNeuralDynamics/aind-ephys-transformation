"""Helpful models used in the ephys compression job"""
from typing import Union
from pathlib import Path
from enum import Enum

from numcodecs import Blosc


PathLike = Union[str, Path]


class ReaderName(str, Enum):
    """Enum for readers"""

    OPENEPHYS = "openephysbinary"
    CHRONIC = "chronic"


class CompressorName(str, Enum):
    """Enum for compression algorithms a user can select"""

    BLOSC = Blosc.codec_id
    WAVPACK = "wavpack"


class RecordingBlockPrefixes(str, Enum):
    """Enum for types of recording blocks."""

    neuropix = "Neuropix"
    nidaq = "NI-DAQ"
