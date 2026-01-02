import logging
import random
from typing import Optional

import hydra

from src.datasets.base_dataset import BaseDataset
from src.datasets.common_voice import CommonVoiceDataset
from src.datasets.librispeech_dataset import LibrispeechDataset
from src.text_encoder import CTCTextEncoder

logger = logging.getLogger(__name__)

class LibriSpeechCommonVoiceMixedDataset(BaseDataset):
    def __init__(
        self,
        text_encoder: CTCTextEncoder,

        # Параметры LibriSpeech
        libri_root: str,
        libri_part: str,

        # Параметры Common Voice
        cv_tar_path: str,
        cv_extract_root: str,
        cv_part: str,
        # Эти параметры определяют, какие файлы попадут в index.json и будут распакованы
        cv_min_duration: float = 0.5,
        cv_max_duration: float = 20.0,
        cv_index_limit: Optional[int] = None, # Лимит на парсинг

        # Параметры смешивания
        cv_fraction: float = 0.3, # Целевая доля CV в датасете
        seed: int = 42,

        # Глобальные параметры обучения (BaseDataset)
        limit: Optional[int] = None,         # Лимит на итоговый размер
        max_audio_length: float = 20.0,      # Фильтр для батчей
        max_text_length: int = 200,
        shuffle_index: bool = True,

        **kwargs
    ):
        instance_transforms = kwargs.get("instance_transforms")
        target_sr = kwargs.get("target_sr", 16000)

        if instance_transforms is None or "get_spectrogram" not in instance_transforms:
            raise ValueError("No get_spectrogram transform provided in config")

        if not (0.0 <= cv_fraction <= 1.0):
            raise ValueError(f"cv_fraction must be in [0, 1], got {cv_fraction}")

        # Загрузка LibriSpeech, берем всё что есть в папке
        logger.info(f"Loading LibriSpeech: {libri_part}")
        libri_ds = LibrispeechDataset(
            root=libri_root,
            part=libri_part,
            text_encoder=text_encoder,
            limit=None,
            shuffle_index=False,
            max_audio_length=None,
            max_text_length=None,
            target_sr=target_sr,
            instance_transforms=instance_transforms,
        )

        # Загрузка Common Voice
        logger.info(
            f"Loading Common Voice: {cv_part}. "
            f"Extraction params: limit={cv_index_limit}, "
            f"duration=[{cv_min_duration}, {cv_max_duration}]"
        )
        # Превращаем пути в абсолютные
        cv_tar_path = hydra.utils.to_absolute_path(cv_tar_path)
        cv_extract_root = hydra.utils.to_absolute_path(cv_extract_root)

        cv_ds = CommonVoiceDataset(
            tar_path=cv_tar_path,
            extract_root=cv_extract_root,
            part=cv_part,
            text_encoder=text_encoder,
            # Передаем лимит и длительности для формирования индекса
            limit=cv_index_limit,
            min_duration=cv_min_duration,
            max_duration=cv_max_duration,
            shuffle_index=False,
            max_audio_length=None,
            max_text_length=None,
            target_sr=target_sr,
            instance_transforms=instance_transforms,
        )

        # Смешивание
        libri_index = list(libri_ds._index)
        cv_index = list(cv_ds._index)

        logger.info(f"Pool size: Libri={len(libri_index)}, CV={len(cv_index)}")

        rng = random.Random(seed)

        if cv_fraction == 0.0 or len(cv_index) == 0:
            mixed = libri_index
            cv_sample = []
            logger.info("Mixing: using only LibriSpeech.")
        elif cv_fraction == 1.0:
            mixed = cv_index
            cv_sample = cv_index
            logger.info("Mixing: using only Common Voice.")
        else:
            target_cv_count = int(len(libri_index) * cv_fraction / (1.0 - cv_fraction))

            if len(cv_index) >= target_cv_count:
                # Если в распакованном CV данных достаточно - берем случайную подвыборку
                cv_sample = rng.sample(cv_index, target_cv_count)
                logger.info(f"Mixing: sampled {target_cv_count} items from CV to match fraction {cv_fraction}")
            else:
                # Если данных мало - берем всё
                cv_sample = cv_index
                logger.warning(
                    f"Mixing: requested {target_cv_count} CV items, but found only {len(cv_index)}. "
                    f"Using all available CV."
                )

            mixed = libri_index + cv_sample

            logger.info(
                f"Mix result: Libri_used={len(libri_index)}, "
                f"CV_used={len(cv_sample)}, total={len(mixed)}, "
                f"target_CV={target_cv_count if 0.0 < cv_fraction < 1.0 else 'n/a'}"
            )

        # Инициализация BaseDataset
        super().__init__(
            index=mixed,
            text_encoder=text_encoder,
            limit=limit,
            shuffle_index=shuffle_index,
            max_audio_length=max_audio_length,
            max_text_length=max_text_length,
            target_sr=target_sr,
            instance_transforms=instance_transforms,
        )
