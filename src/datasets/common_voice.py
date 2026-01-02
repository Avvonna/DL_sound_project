import csv
import gzip
import json
import logging
import os
import random
import tarfile
from pathlib import Path
from typing import Optional, Tuple

import hydra
import pandas as pd
from tqdm import tqdm

from src.datasets.base_dataset import BaseDataset
from src.text_encoder import CTCTextEncoder

logger = logging.getLogger(__name__)

def _norm(x):
    if pd.isna(x):
        return ""
    try:
        return CTCTextEncoder.normalize_text(str(x))
    except Exception:
        return ""

class CommonVoiceDataset(BaseDataset):
    SPLITS = ["train", "dev", "test", "validated", "other", "invalidated"]

    def __init__(
        self,
        tar_path: str,
        extract_root: str,
        text_encoder: CTCTextEncoder,
        part: str = "train",
        min_duration: float = 0.5,
        max_duration: float = 20.0,
        limit: Optional[int] = None,
        force_extract: bool = False,
        *args,
        **kwargs,
    ):
        self.tar_path = Path(hydra.utils.to_absolute_path(tar_path))
        self.extract_root = Path(hydra.utils.to_absolute_path(extract_root))
        self.part = part
        self.min_duration = float(min_duration)
        self.max_duration = float(max_duration)
        self.limit = limit

        self.extract_root.mkdir(parents=True, exist_ok=True)

        # определяем структуру (версию, папки), чтобы знать куда смотреть
        # Используем кэширование, чтобы не открывать TAR каждый раз
        self.internal_root, self.version = self._get_or_create_metadata()

        self.prefix = f"{self.internal_root}/en"
        self.data_dir = self.extract_root / self.internal_root / "en"
        self.clips_dir = self.data_dir / "clips"

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.clips_dir.mkdir(parents=True, exist_ok=True)

        # Формируем имя индекса под текущие параметры
        index_name = f"{self.part}_{self.min_duration}-{self.max_duration}s_{self.version}.json"
        if self.limit:
            index_name = index_name.replace(".json", f"_limit{self.limit}.json")

        self.index_path = self.data_dir / index_name

        # Если индекса нет или форсировано, создаем его
        # При создании мы проверим наличие файлов и докачаем недостающие
        if (not self.index_path.exists()) or force_extract:
            logger.info(f"Index not found or forced: {self.index_path}")
            self._create_index_and_extract_missing()
        else:
            logger.info(f"Found cached index: {self.index_path}")

        # Загружаем индекс
        with open(self.index_path, "r", encoding="utf-8") as f:
            index = json.load(f)

        super().__init__(index, compute_audio_len_if_missing=False, text_encoder=text_encoder, *args, **kwargs)

    def _get_or_create_metadata(self) -> Tuple[str, str]:
        """
        Кэширует информацию о структуре архива (internal_root и версию),
        чтобы не парсить TAR при каждом изменении параметров длительности.
        """
        meta_path = self.extract_root / "cv_metadata.json"

        if meta_path.exists():
            try:
                with open(meta_path, "r") as f:
                    data = json.load(f)
                    # Простая проверка, что это мета для того же архива (по имени файла)
                    if data.get("tar_name") == self.tar_path.name:
                        return data["internal_root"], data["version"]
            except Exception:
                logger.warning("Metadata corrupted, recreating...")

        # Если кэша нет, детектим через TAR (медленно, 1 раз)
        logger.info("Scanning tarball structure (one-time)...")
        internal_root, _ = self._detect_prefix()
        version = self._extract_version(internal_root)

        # Сохраняем
        with open(meta_path, "w") as f:
            json.dump({
                "tar_name": self.tar_path.name,
                "internal_root": internal_root,
                "version": version
            }, f)

        return internal_root, version

    def _extract_version(self, internal_root: str) -> str:
        """Извлекает версию из названия корня (например, cv-corpus-24.0-... -> v24)"""
        parts = internal_root.split("-")
        for part in parts:
            if part.replace(".", "").isdigit():
                return f"v{part.split('.')[0]}"
        return "vUnknown"

    def _manifest_path(self) -> Path:
        return self.data_dir / f"manifest_en_{self.version}.csv.gz"

    def _find_column(self, df: pd.DataFrame, possible_names: list[str], contains: bool = False) -> Optional[str]:
        """
        Поиск колонки по списку возможных названий.

        Args:
            df: DataFrame для поиска
            possible_names: список возможных названий колонок (lowercase)
            contains: если True, ищет по вхождению подстроки
        """
        for col in df.columns:
            col_lower = str(col).lower()
            if contains:
                if any(name in col_lower for name in possible_names):
                    return col
            else:
                if col_lower in possible_names:
                    return col
        return None

    def build_manifest(self, force: bool = False) -> Path:
        """
        Собирает один общий manifest из архива и сохраняет в data_dir/manifest_en_{version}.csv.gz
        Запускается один раз (или force=True).
        """
        out_path = self._manifest_path()
        if out_path.exists() and not force:
            logger.info(f"Found cached manifest: {out_path}")
            return out_path

        self.data_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Building manifest (reading tar headers)...")

        dur_map = {}
        split_dfs = {}
        needed_splits = set(self.SPLITS)

        with tarfile.open(self.tar_path, "r:gz") as tar:
            for member in tar:
                if "/clips/" in member.name:
                    # дальше можно не смотреть
                    continue

                filename = Path(member.name).name

                # Собираем длительности аудио
                if filename == "clip_durations.tsv":
                    try:
                        f = tar.extractfile(member)
                        if f:
                            df_dur = pd.read_csv(f, sep="\t", quoting=csv.QUOTE_NONE, on_bad_lines='skip', dtype="string", low_memory=False)
                            clip_col = self._find_column(df_dur, ["clip", "path"], contains=True)
                            dur_col = self._find_column(df_dur, ["duration"], contains=True)
                            if clip_col and dur_col:
                                df_dur["filename"] = df_dur[clip_col].astype(str).map(lambda x: Path(x).name)
                                # ms -> s
                                df_dur["val"] = pd.to_numeric(df_dur[dur_col], errors="coerce") / 1000.0
                                df_dur = df_dur.dropna()
                                dur_map = dict(zip(df_dur["filename"], df_dur["val"]))
                    except Exception as e:
                        logger.error(f"Error parsing durations: {e}")

                # Читаем разметку для сплитов
                elif filename.endswith(".tsv"):
                    split_name = filename.replace(".tsv", "")
                    if split_name in needed_splits:
                        try:
                            f = tar.extractfile(member)
                            if f:
                                split_dfs[split_name] = pd.read_csv(f, sep="\t", quoting=csv.QUOTE_NONE, on_bad_lines='skip', dtype="string", low_memory=False)
                        except Exception as e:
                            logger.error(f"Error parsing {split_name}: {e}")

        logger.info("Saving manifest csv...")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Записываем итоговый файл
        with gzip.open(out_path, "wt", encoding="utf-8", newline="") as gz:
            writer = csv.writer(gz, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["split", "filename", "text", "audio_len"])

            for split in tqdm(self.SPLITS, desc="Processing splits"):
                df = split_dfs.get(split)
                if df is None or df.empty:
                    continue

                text_col = self._find_column(df, ["text", "sentence"])
                path_col = self._find_column(df, ["path", "clip"])
                if text_col is None or path_col is None:
                    continue

                df = df[[path_col, text_col]].copy()
                df["filename"] = df[path_col].astype(str).map(lambda x: Path(x).name)

                # Мэппим длительности к файлам. Если длительности нет в clip_durations — ставим None
                if dur_map:
                    df["audio_len"] = pd.to_numeric(df["filename"].map(dur_map), errors="coerce")
                else:
                    df["audio_len"] = None

                # Выкидываем битые данные (без длительности или текста)
                df = df.dropna(subset=["audio_len"])
                df["text"] = df[text_col].map(_norm)
                df = df[df["text"].str.len() > 0]

                for row in df.itertuples(index=False):
                    writer.writerow([split, row.filename, row.text, row.audio_len])

        return out_path

    def _iter_selected_from_manifest(self, split: str, seed: int = 42) -> list[dict]:
        """
        Читает готовый манифест и отбирает `limit` записей.

        Использует Reservoir Sampling: позволяет выбрать случайные N элементов из потока
        неизвестной длины за один проход, не загружая весь датасет в память.
        """
        manifest_path = self._manifest_path()
        rng = random.Random(seed)
        reservoir: list[dict] = []
        seen = 0
        chunksize = 100_000

        logger.info(f"Reading manifest for split '{split}'...")
        reader = pd.read_csv(manifest_path, compression="gzip", chunksize=chunksize)

        for chunk in reader:
            # Берем только нужный сплит (train/test/...)
            chunk = chunk[chunk["split"] == split]
            if chunk.empty:
                continue

            # Фильтр по длине аудио
            mask = (chunk["audio_len"] >= self.min_duration) & (chunk["audio_len"] <= self.max_duration)
            chunk = chunk[mask]

            if chunk.empty:
                continue

            for row in chunk.itertuples(index=False):
                seen += 1
                item = {"filename": row.filename, "text": row.text, "audio_len": float(row.audio_len)}
                if self.limit is None:
                    reservoir.append(item)
                else:
                    # Пока резервуар не полон, просто заполняем
                    if len(reservoir) < self.limit:
                        reservoir.append(item)
                    else:
                        # Если полон, с вероятностью limit/seen заменяем случайный элемент
                        j = rng.randrange(seen)
                        if j < self.limit:
                            reservoir[j] = item

        logger.info(f"Selected {len(reservoir)} items from {seen} candidates")
        return reservoir

    def _detect_prefix(self) -> tuple[str, str]:
        """Определяет внутреннюю структуру архива."""
        targets = {"train.tsv", "dev.tsv", "test.tsv", "clip_durations.tsv"}
        with tarfile.open(self.tar_path, "r:gz") as tar:
            for m in tar:
                if "/clips/" in m.name:
                    # Дальше клипов нет смысла искать конфиги
                    break
                parts = m.name.split("/")
                if len(parts) >= 2 and parts[-1] in targets:
                    if "en" in parts:
                        idx = parts.index("en")
                        internal_root = "/".join(parts[:idx])
                        return internal_root, f"{internal_root}/en"
        raise FileNotFoundError(f"Could not detect valid prefix in start of {self.tar_path}")

    def _create_index_and_extract_missing(self):
        # Смотрим / создаем манифест
        self.build_manifest(force=False)

        # Выбираем нужные записи по limit и duration
        selected = self._iter_selected_from_manifest(self.part, seed=42)

        dataset_entries = []
        needed_filenames = set()
        needed_files_map = {} # TarPath -> Filename

        # Формируем список того, что нужно
        for x in selected:
            filename = x["filename"]
            tar_member_path = f"{self.prefix}/clips/{filename}"

            needed_filenames.add(filename)
            needed_files_map[tar_member_path] = filename

            dataset_entries.append({
                "path": str((self.clips_dir / filename).absolute()),
                "text": x["text"],
                "audio_len": x["audio_len"],
            })

        # Проверяем, что уже есть
        existing_filenames = set()
        if self.clips_dir.exists():
            with os.scandir(self.clips_dir) as entries:
                for entry in entries:
                    if entry.is_file() and entry.name.endswith('.mp3'):
                        existing_filenames.add(entry.name)

        # Оставляем только те, которых нет
        files_to_extract = {
            k: v for k, v in needed_files_map.items()
            if v not in existing_filenames
        }

        # Если чего-то не хватает, лезем в архив
        if files_to_extract:
            logger.info(f"Extracting {len(files_to_extract)} missing files from TAR...")
            with tarfile.open(self.tar_path, "r:gz") as tar:
                extracted_count = 0
                total_needed = len(files_to_extract)
                has_filter = hasattr(tarfile, 'data_filter')

                with tqdm(total=total_needed, desc="Extracting") as pbar:
                    for member in tar:
                        if member.name in files_to_extract:
                            target_filename = files_to_extract[member.name]
                            member.name = target_filename # Убираем путь папок, распаковываем плоско

                            if has_filter:
                                tar.extract(member, path=self.clips_dir, filter='data')
                            else:
                                tar.extract(member, path=self.clips_dir)

                            extracted_count += 1
                            pbar.update(1)

                            # Если нашли всё, выходим досрочно
                            if extracted_count >= total_needed:
                                break
        else:
            logger.info("All required files are already present. Skipping extraction.")

        # Сохраняем индекс
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(dataset_entries, f, indent=2, ensure_ascii=False)
