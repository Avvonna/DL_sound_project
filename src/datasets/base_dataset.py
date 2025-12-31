import logging
import random
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional
from torch.utils.data import Dataset

from src.text_encoder import CTCTextEncoder

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    """
    Base class for ASR datasets.

    Given a proper index (list[dict]), allows to process different datasets
    for the same task in the identical manner. Therefore, to work with
    several datasets, the user only have to define index in a nested class.

    Supported index fields per item:
    - path: str (required)
    - text: str (optional, can be absent for inference)
    - audio_len: float seconds (optional, will be computed if absent and needed)
    """

    def __init__(
        self,
        index: List[Dict[str, Any]],
        text_encoder=None,
        target_sr: int = 16000,
        limit: Optional[int] = None,
        max_audio_length: Optional[float] = None,
        max_text_length: Optional[int] = None,
        shuffle_index: bool = False,
        instance_transforms: Optional[Dict[str, Optional[Callable]]] = None,
        require_text: bool = True,
        compute_audio_len_if_missing: bool = True,
    ):
        """
        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
            text_encoder (CTCTextEncoder): text encoder.
            target_sr (int): supported sample rate.
            limit (int | None): if not None, limit the total number of elements
                in the dataset to 'limit' elements.
            max_audio_length (int): maximum allowed audio length.
            max_test_length (int): maximum allowed text length.
            shuffle_index (bool): if True, shuffle the index. Uses python
                random package with seed 42.
            instance_transforms (dict[Callable] | None): transforms that
                should be applied on the instance. Depend on the
                tensor name.
            require_text (bool): whether the ground truth transcription is required
            compute_audio_len_if_missing (bool): whether to compute missing audio length
        """
        self.require_text = require_text
        self.compute_audio_len_if_missing = compute_audio_len_if_missing
        self._assert_index_is_valid(index)

        if compute_audio_len_if_missing:
            index = self._ensure_audio_len(index)

        index = self._filter_records_from_dataset(
            index, max_audio_length, max_text_length
        )
        index = self._shuffle_and_limit_index(index, limit, shuffle_index)

        if not shuffle_index:
            index = self._sort_index(index)

        self._index: List[Dict[str, Any]] = index
        self.target_sr = target_sr

        if text_encoder is None:
            raise ValueError("text_encoder cannot be None")
        if instance_transforms is None:
            instance_transforms = {}
        if "get_spectrogram" not in instance_transforms:
             raise ValueError("No get_spectrogram transform provided in config")
        
        if "audio" not in instance_transforms:
            instance_transforms["audio"] = None
        if "spectrogram" not in instance_transforms:
            instance_transforms["spectrogram"] = None

        self.text_encoder = text_encoder
        self.instance_transforms = instance_transforms

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, ind: int) -> Dict[str, Any]:
        data_dict = self._index[ind]

        audio_path = data_dict["path"]
        audio = self.load_audio(audio_path)
        audio_orig = audio.clone()

        # wave augs
        if (
            "audio" in self.instance_transforms
            and self.instance_transforms["audio"] is not None
        ):
            audio = self.instance_transforms["audio"](audio)

        # spectrogram
        spectrogram = self.get_spectrogram(audio)

        # spectrogram augs / postprocessing
        if (
            "spectrogram" in self.instance_transforms
            and self.instance_transforms["spectrogram"] is not None
        ):
            spectrogram = self.instance_transforms["spectrogram"](spectrogram)

        instance_data: Dict[str, Any] = {
            "audio": audio,
            "audio_orig": audio_orig,
            "spectrogram": spectrogram,
            "audio_path": audio_path,
        }

        # text is optional
        text = data_dict.get("text", None)
        if text is not None:
            instance_data["text"] = text
            instance_data["text_encoded"] = self.text_encoder.encode(text)

        return instance_data

    def load_audio(self, path: str) -> torch.Tensor:
        audio_np, sr = sf.read(path, dtype="float32", always_2d=True)
        audio_np = audio_np[:, 0]  # первый канал
        audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)  # (1, T)

        if sr != self.target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, self.target_sr)

        return audio_tensor

    def get_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        transform = self.instance_transforms.get("get_spectrogram")
    
        if transform is None:
             raise ValueError("get_spectrogram transform is missing or None")
    
        return transform(audio)

    def _ensure_audio_len(self, index: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for el in index:
            if "audio_len" not in el or el["audio_len"] is None:
                try:
                    info = sf.info(el["path"])
                    el = dict(el)
                    el["audio_len"] = float(info.frames) / float(info.samplerate)
                except Exception:
                    el = dict(el)
                    el["audio_len"] = None
            out.append(el)
        return out

    def _filter_records_from_dataset(
        self,
        index: List[Dict[str, Any]],
        max_audio_length: Optional[float],
        max_text_length: Optional[int],
    ) -> List[Dict[str, Any]]:
        """
        Filter some of the elements from the dataset depending on
        the desired max_test_length or max_audio_length.

        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
            max_audio_length (int): maximum allowed audio length.
            max_test_length (int): maximum allowed text length.
        Returns:
            index (list[dict]): list, containing dict for each element of
                the dataset that satisfied the condition. The dict has
                required metadata information, such as label and object path.
        """
        if len(index) == 0:
            return index

        initial_size = len(index)

        # audio length filter
        if max_audio_length is not None:
            audio_lens = np.array(
                [
                    el.get("audio_len", None)
                    if el.get("audio_len", None) is not None
                    else np.inf
                    for el in index
                ],
                dtype=float,
            )
            exceeds_audio_length = audio_lens > float(max_audio_length)
            _total = int(exceeds_audio_length.sum())
            if _total > 0:
                logger.debug(
                    f"{_total} ({_total / initial_size:.1%}) records are longer than "
                    f"{max_audio_length} seconds. Excluding them."
                )
        else:
            exceeds_audio_length = np.zeros(len(index), dtype=bool)

        # text length filter
        if max_text_length is not None:
            text_lens = []
            for el in index:
                txt = el.get("text", None)
                if txt is None:
                    text_lens.append(0)  # keep items without text
                else:
                    text_lens.append(len(CTCTextEncoder.normalize_text(txt)))
            exceeds_text_length = np.array(text_lens, dtype=int) > int(max_text_length)
            _total = int(exceeds_text_length.sum())
            if _total > 0:
                logger.debug(
                    f"{_total} ({_total / initial_size:.1%}) records are longer than "
                    f"{max_text_length} characters. Excluding them."
                )
        else:
            exceeds_text_length = np.zeros(len(index), dtype=bool)

        records_to_filter = exceeds_text_length | exceeds_audio_length
        if records_to_filter.any():
            _total = int(records_to_filter.sum())
            index = [el for el, exclude in zip(index, records_to_filter) if not exclude]
            logger.debug(
                f"Filtered {_total} ({_total / initial_size:.1%}) records from dataset"
            )

        return index

    def _assert_index_is_valid(self, index: List[Dict[str, Any]]) -> None:
        """
        Check the structure of the index and ensure it satisfies the desired
        conditions.

        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
        """
        for i, entry in enumerate(index):
            if "path" not in entry:
                raise KeyError(f"Index item {i} is missing required field 'path'")

            if self.require_text and "text" not in entry:
                raise KeyError(
                    f"Index item {i} is missing required field 'text' "
                    "(ground truth transcription)."
                )

    @staticmethod
    def _sort_index(index: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sort index by audio length.

        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
        Returns:
            index (list[dict]): sorted list, containing dict for each element
                of the dataset. The dict has required metadata information,
                such as label and object path.
        """
        # sort only if audio_len exists for all items
        if len(index) == 0:
            return index
        if all(("audio_len" in el) and (el["audio_len"] is not None) for el in index):
            return sorted(index, key=lambda x: x["audio_len"])
        return index

    @staticmethod
    def _shuffle_and_limit_index(
        index: List[Dict[str, Any]],
        limit: Optional[int],
        shuffle_index: bool,
    ) -> List[Dict[str, Any]]:
        """
        Shuffle elements in index and limit the total number of elements.

        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
            limit (int | None): if not None, limit the total number of elements
                in the dataset to 'limit' elements.
            shuffle_index (bool): if True, shuffle the index. Uses python
                random package with seed 42.
        """
        if shuffle_index:
            random.seed(42)
            random.shuffle(index)
        if limit is not None:
            index = index[: int(limit)]
        return index
