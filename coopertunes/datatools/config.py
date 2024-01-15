from enum import Enum


class DataType(Enum):
    AUDIO = "audio"
    MIDI = "midi"


DATA_NAMES = {
    DataType.MIDI: ["classic_piano"],
    DataType.AUDIO: [],
}
