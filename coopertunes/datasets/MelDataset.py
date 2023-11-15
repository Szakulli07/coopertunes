from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset

from ..hparams import HParams
from ..utils import (AUDIO_EXTENSIONS, convert_audios2mels, log_info,
                     normalize_audio)


class MelDataset(Dataset):
    def __init__(self, hparams: HParams, data_dirs: list[Path] | Path):
        super().__init__()
        self.hparams = hparams
        if isinstance(data_dirs, Path):
            data_dirs = [data_dirs]
        self.data_dirs = data_dirs

        assert all(map(Path.exists, self.data_dirs))

        self.filepaths = self._load_paths()

        log_info("Prepared dataset, loaded %d filepaths", len(self.filepaths))

    def _load_paths(self):
        filepaths = []
        for data_dir in self.data_dirs:
            filepaths.extend(
                [fp for ext in AUDIO_EXTENSIONS for fp in data_dir.rglob(f"*{ext}")]
            )
        return filepaths

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]

        audio, sample_rate = torchaudio.load(filepath)
        audio = normalize_audio(audio, sample_rate, self.hparams.sample_rate)
        mels = convert_audios2mels(
            audio,
            self.hparams.sample_rate,
            self.hparams.n_fft,
            self.hparams.win_length,
            self.hparams.hop_length,
            self.hparams.n_mels,
            self.hparams.normalized
        )

        mels = self._get_segment(mels)

        return {"mels": mels, "filepath": str(filepath.absolute())}

    def __len__(self):
        return len(self.filepaths)

    def _get_segment(self, mels):
        if self.hparams.segment_len is None:
            return mels

        if mels.shape[-1] > self.hparams.segment_len:
            max_start = mels.shape[-1] - self.hparams.segment_len
            start = torch.randint(0, max_start, (1,))
            mels = mels[:, :, start: start + self.hparams.segment_len]
        else:
            mels = F.pad(
                mels, (0, self.hparams.segment_len - mels.size(1)), "constant"
            )
        return mels
