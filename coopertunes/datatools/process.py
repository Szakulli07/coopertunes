import os
import sys
import hashlib
from concurrent.futures import ProcessPoolExecutor

import torch
from progress.bar import Bar

from coopertunes.datatools.miditools import NoteSeq, EventSeq, ControlSeq
from coopertunes.utils import find_files_by_extensions

# pylint: disable=W0718

def get_preprocessing(name):
    """
    Returns processing funtion for dataset.
    Name should be in DATA_NAMES values.
    """
    downloaders = {"classic_piano": preprocess_classic_piano}
    return downloaders[name]


def preprocess_wav2spectrogram():
    """
    Preprocess single wav under given path to spectrogram
    """


def preprocess_midi2sequence(path):
    """
    Preprocess single midi under given path to event sequence.
    """
    note_seq = NoteSeq.from_midi_file(path)
    note_seq.adjust_time(-note_seq.notes[0].start)
    event_seq = EventSeq.from_note_seq(note_seq)
    control_seq = ControlSeq.from_event_seq(event_seq)
    return event_seq.to_array(), control_seq.to_compressed_array()


def preprocess_classic_piano(midi_root, save_dir, num_workers):
    midi_paths = list(find_files_by_extensions(midi_root, [".mid", ".midi"]))
    os.makedirs(save_dir, exist_ok=True)
    out_fmt = "{}-{}.data"
    faulty_data_counter = 0

    results = []
    executor = ProcessPoolExecutor(num_workers)

    for path in midi_paths:
        try:
            results.append((path, executor.submit(preprocess_midi2sequence, path)))
        except KeyboardInterrupt:
            print(" Abort")
            return
        except Exception:  # noqa
            print(" Error")
            continue

    for path, future in Bar("Processing").iter(results):
        name = os.path.basename(path)
        code = hashlib.md5(path.encode()).hexdigest()
        save_path = os.path.join(save_dir, out_fmt.format(name, code))
        try:
            torch.save(future.result(), save_path)
        except OSError:
            faulty_data_counter += 1
    print(faulty_data_counter)


if __name__ == "__main__":
    preprocess_classic_piano(
        midi_root=sys.argv[1], save_dir=sys.argv[2], num_workers=int(sys.argv[3])
    )
