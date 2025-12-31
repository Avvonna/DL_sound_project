import csv
import gzip
import json
import logging
import random
import tarfile
from pathlib import Path
from typing import Optional

import hydra
import pandas as pd
from tqdm import tqdm

from src.datasets.base_dataset import BaseDataset
from src.text_encoder import CTCTextEncoder

logger = logging.getLogger(__name__)

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
        self.internal_root, self.prefix = self._detect_prefix()

        self.version = self._extract_version(self.internal_root)

        self.data_dir = self.extract_root / self.internal_root / "en"
        self.clips_dir = self.data_dir / "clips"

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.clips_dir.mkdir(parents=True, exist_ok=True)

        index_name = f"{self.part}_{self.min_duration}-{self.max_duration}s_{self.version}.json"
        if self.limit:
            index_name = index_name.replace(".json", f"_limit{self.limit}.json")

        self.index_path = self.data_dir / index_name

        if (not self.index_path.exists()) or force_extract:
            logger.info(f"Index not found or forced: {self.index_path}")
            self._extract_and_prepare()
        else:
            logger.info(f"Found cached index: {self.index_path}")

        # Загружаем индекс
        with open(self.index_path, "r", encoding="utf-8") as f:
            index = json.load(f)

        super().__init__(index, compute_audio_len_if_missing=False, text_encoder=text_encoder, *args, **kwargs)

    def _extract_version(self, internal_root: str) -> str:
        """Извлекает версию из названия корня (например, cv-corpus-24.0-... -> v24)"""
        parts = internal_root.split("-")
        for part in parts:
            if part.replace(".", "").isdigit():
                major_version = part.split(".")[0]
                return f"v{major_version}"
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
        logger.info("Building manifest (one-time)...")

        with tarfile.open(self.tar_path, "r:gz") as tar:
            try:
                # durations
                df_dur = self._read_durations_df(tar)
                # dict: filename -> seconds
                dur_map = dict(zip(df_dur["filename"].tolist(), df_dur["audio_len"].tolist()))
            except Exception as e:
                logger.error(f"Failed to load durations: {e}. Will rely on manifest parsing only.")
                dur_map = {}

            # пишем gz-csv построчно с экранированием
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(out_path, "wt", encoding="utf-8", newline="") as gz:
                writer = csv.writer(gz, quoting=csv.QUOTE_MINIMAL)
                writer.writerow(["split", "filename", "text", "audio_len"])

                for split in tqdm(self.SPLITS, desc="Processing splits"):
                    df = self._read_split_df(tar, split)
                    if df is None or df.empty:
                        continue

                    # определяем текстовую колонку
                    text_col = self._find_column(df, ["text", "sentence"])
                    path_col = self._find_column(df, ["path", "clip"])

                    if text_col is None or path_col is None:
                        continue

                    # чтобы не держать лишнее
                    df = df[[path_col, text_col]].copy()
                    df["filename"] = df[path_col].astype(str).map(lambda x: Path(x).name)

                    # пытаемся взять длительность из dur_map, если нет - пропускаем
                    df["audio_len"] = pd.to_numeric(df["filename"].map(dur_map), errors="coerce")
                    df = df.dropna(subset=["audio_len"])
                    df["audio_len"] = df["audio_len"].astype("float64")

                    # normalize text
                    def _norm(x):
                        if pd.isna(x):
                            return ""
                        try:
                            return CTCTextEncoder.normalize_text(str(x))
                        except Exception:
                            return ""

                    df["text"] = df[text_col].map(_norm)
                    df = df[df["text"].str.len() > 0]

                    for row in df.itertuples(index=False):
                        writer.writerow([split, row.filename, row.text, row.audio_len])

        logger.info(f"Manifest saved: {out_path}")
        return out_path

    def _read_durations_df(self, tar: tarfile.TarFile) -> pd.DataFrame:
        full_path = f"{self.prefix}/clip_durations.tsv"
        try:
            m = tar.getmember(full_path)
        except KeyError:
             logger.warning("clip_durations.tsv not found via getmember.")
             raise

        f = tar.extractfile(m)
        if f is None:
            raise ValueError("clip_durations.tsv cannot be read")

        df = pd.read_csv(f, sep="\t", quoting=csv.QUOTE_NONE, on_bad_lines='skip', dtype="string", low_memory=False)

        clip_col = self._find_column(df, ["clip", "path"], contains=True)
        dur_col = self._find_column(df, ["duration"], contains=True)

        if not clip_col or not dur_col:
            raise ValueError(f"Columns not found in durations: {df.columns}")

        df = df[[clip_col, dur_col]].rename(columns={clip_col: "filename", dur_col: "duration_ms"})
        df["filename"] = df["filename"].astype(str).map(lambda x: Path(x).name)
        df["audio_len"] = pd.to_numeric(df["duration_ms"], errors="coerce") / 1000.0
        return df.dropna(subset=["audio_len"])

    def _read_split_df(self, tar: tarfile.TarFile, split: str) -> Optional[pd.DataFrame]:
        full_path = f"{self.prefix}/{split}.tsv"
        try:
            member = tar.getmember(full_path)
            f = tar.extractfile(member)
            if f is None:
                return None
            return pd.read_csv(f, sep="\t", quoting=csv.QUOTE_NONE, on_bad_lines='skip', dtype="string", low_memory=False)
        except (KeyError, Exception) as e:
            logger.debug(f"Split {split} not found or error: {e}")
            return None

    def _iter_selected_from_manifest(self, split: str, seed: int = 42) -> list[dict]:
        """
        Читает manifest и выбирает limit элементов после фильтра по min/max duration.
        Делает reservoir sampling, чтобы не грузить весь manifest в память.
        """
        manifest_path = self._manifest_path()
        rng = random.Random(seed)
        reservoir: list[dict] = []
        seen = 0
        chunksize = 100_000 # Поменьше для экономии памяти

        logger.info(f"Reading manifest for split '{split}'...")

        reader = pd.read_csv(manifest_path, compression="gzip", chunksize=chunksize)
        for chunk in reader:
            chunk = chunk[chunk["split"] == split]
            if chunk.empty:
                continue

            # фильтрация
            mask = (chunk["audio_len"] >= self.min_duration) & (chunk["audio_len"] <= self.max_duration)
            chunk = chunk[mask]

            if chunk.empty:
                continue

            for row in chunk.itertuples(index=False):
                seen += 1
                item = {
                    "filename": row.filename,
                    "text": row.text,
                    "audio_len": float(row.audio_len),
                }

                if self.limit is None:
                    reservoir.append(item)
                else:
                    if len(reservoir) < self.limit:
                        reservoir.append(item)
                    else:
                        j = rng.randrange(seen)
                        if j < self.limit:
                            reservoir[j] = item

        logger.info(f"Selected {len(reservoir)} items from {seen} candidates")
        if self.limit is not None and len(reservoir) < self.limit and seen > 0:
            logger.warning(f"Requested limit {self.limit} but only found {len(reservoir)}. Filters too strict?")

        return reservoir

    def _detect_prefix(self) -> tuple[str, str]:
        """
        Находит prefix вида "cv-corpus-24.0-2025-12-05/en"
        """
        targets = [f"/en/{self.part}.tsv", "/en/train.tsv", "/en/dev.tsv"]
        # Открываем для быстрого сканирования заголовков
        with tarfile.open(self.tar_path, "r:gz") as tar:
            for m in tar:
                # Обычно метаданные в начале, break как только нашли
                if any(m.name.endswith(t) for t in targets):
                    internal_root = m.name.split("/")[0]
                    return internal_root, f"{internal_root}/en"
                # Защита от полного перебора архива, если метаданных нет в начале
        raise FileNotFoundError(f"Could not detect prefix in {self.tar_path}")

    def _extract_and_prepare(self):
        # manifest
        self.build_manifest(force=False)
        selected = self._iter_selected_from_manifest(self.part, seed=42)

        needed_files_map = {}
        dataset_entries = []

        for x in selected:
            filename = x["filename"]
            # Полный путь внутри архива
            tar_member_path = f"{self.prefix}/clips/{filename}"
            needed_files_map[tar_member_path] = filename

            dataset_entries.append({
                "path": str((self.clips_dir / filename).absolute()),
                "text": x["text"],
                "audio_len": x["audio_len"],
            })

        existing_files = set(f.name for f in self.clips_dir.glob("*.mp3"))
        # Оставляем только те, которых еще нет
        files_to_extract = {k: v for k, v in needed_files_map.items() if v not in existing_files}

        if files_to_extract:
            logger.info(f"Extracting {len(files_to_extract)} files...")
            with tarfile.open(self.tar_path, "r:gz") as tar:
                extracted_count = 0
                total_needed = len(files_to_extract)

                # Check for safe extraction capability (Python 3.12+)
                has_filter = hasattr(tarfile, 'data_filter')

                with tqdm(total=total_needed, desc="Extracting") as pbar:
                    for member in tar:
                        if member.name in files_to_extract:
                            target_filename = files_to_extract[member.name]

                            member.name = target_filename

                            # Безопасная распаковка
                            if has_filter:
                                tar.extract(member, path=self.clips_dir, filter='data')
                            else:
                                tar.extract(member, path=self.clips_dir)

                            extracted_count += 1
                            pbar.update(1)
                            if extracted_count >= total_needed:
                                break
        else:
            logger.info("All files already extracted.")

        # Финальная проверка
        missing = [e["path"] for e in dataset_entries if not Path(e["path"]).exists()]
        if missing:
             raise RuntimeError(f"Missing {len(missing)} files (e.g. {missing[0]})")

        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(dataset_entries, f, indent=2, ensure_ascii=False)
