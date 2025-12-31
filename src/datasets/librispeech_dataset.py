import json
import logging
import os
import shutil
from pathlib import Path

import hydra
import soundfile as sf
import wget
from tqdm import tqdm

from src.datasets.base_dataset import BaseDataset
from src.text_encoder import CTCTextEncoder

logger = logging.getLogger(__name__)

URL_LINKS = {
    "dev-clean": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
    "dev-other": "https://www.openslr.org/resources/12/dev-other.tar.gz",
    "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
    "test-other": "https://www.openslr.org/resources/12/test-other.tar.gz",
    "train-clean-100": "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
    "train-clean-360": "https://www.openslr.org/resources/12/train-clean-360.tar.gz",
    "train-other-500": "https://www.openslr.org/resources/12/train-other-500.tar.gz",
}


class LibrispeechDataset(BaseDataset):
    def __init__(
        self, 
        part: str, 
        text_encoder: CTCTextEncoder, 
        root: str,
        *args, 
        **kwargs
    ):
        assert part in URL_LINKS or part == "train_all", f"Unknown part: {part}"

        self._data_dir = Path(hydra.utils.to_absolute_path(root)) 
        self._data_dir.mkdir(exist_ok=True, parents=True)

        if part == "train_all":
            index = sum(
                [
                    self._get_or_load_index(sub_part)
                    for sub_part in URL_LINKS
                    if "train" in sub_part
                ],
                [],
            )
        else:
            index = self._get_or_load_index(part)

        super().__init__(index, text_encoder=text_encoder, *args, **kwargs)

    def _load_part(self, part):
        arch_path = self._data_dir / f"{part}.tar.gz"
        logger.info(f"Downloading LibriSpeech part '{part}' to {arch_path}...")
        wget.download(URL_LINKS[part], str(arch_path))
        
        logger.info(f"Unpacking {part}...")
        shutil.unpack_archive(arch_path, self._data_dir)
        
        # Перемещаем содержимое из LibriSpeech/part внутри архива в корень
        libri_root_inside = self._data_dir / "LibriSpeech"
        if libri_root_inside.exists():
            for fpath in libri_root_inside.iterdir():
                target = self._data_dir / fpath.name
                if not target.exists():
                    shutil.move(str(fpath), str(target))
            shutil.rmtree(str(libri_root_inside))
            
        os.remove(str(arch_path))

    def _get_or_load_index(self, part):
        index_path = self._data_dir / f"{part}_index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(part)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self, part):
        index = []
        split_dir = self._data_dir / part
        if not split_dir.exists():
            self._load_part(part)

        flac_dirs = set()
        for dirpath, dirnames, filenames in os.walk(str(split_dir)):
            if any([f.endswith(".flac") for f in filenames]):
                flac_dirs.add(dirpath)
        
        for flac_dir in tqdm(list(flac_dirs), desc=f"Indexing {part}"):
            flac_dir = Path(flac_dir)
            trans_files = list(flac_dir.glob("*.trans.txt"))
            if not trans_files:
                continue
            
            trans_path = trans_files[0]
            with trans_path.open() as f:
                for line in f:
                    parts = line.split()
                    f_id = parts[0]
                    f_text = " ".join(parts[1:]).strip()
                    flac_path = flac_dir / f"{f_id}.flac"
                    
                    if not flac_path.exists():
                        continue
                        
                    try:
                        t_info = sf.info(str(flac_path))
                        length = t_info.frames / t_info.samplerate
                        index.append(
                            {
                                "path": str(flac_path.absolute().resolve()),
                                "text": f_text.lower(),
                                "audio_len": length,
                            }
                        )
                    except Exception as e:
                        logger.warning(f"Error reading {flac_path}: {e}")
                        
        return index
