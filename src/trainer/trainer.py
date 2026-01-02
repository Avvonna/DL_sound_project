import logging
from pathlib import Path
from typing import cast

import pandas as pd
import torch
from torch.amp.autocast_mode import autocast

from src.datasets.base_dataset import BaseDataset
from src.logger.utils import plot_spectrogram_grid
from src.metrics.tracker import MetricTracker
from src.metrics.utils import calc_cer, calc_wer
from src.trainer.base_trainer import BaseTrainer
from src.utils.decoding_utils import decode_beam, parse_decoding_cfg

logger = logging.getLogger(__name__)


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)

        # сохраняем спектрограмму до batch spec augs
        if (
            getattr(self, "_need_spec_log", False)
            and "spectrogram" in batch
            and torch.is_tensor(batch["spectrogram"])
        ):
            batch["spectrogram_pre_batch"] = batch["spectrogram"].detach().clone()

        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]

        if self.use_amp:
            with autocast("cuda", dtype=self.amp_dtype):
                outputs = self.model(**batch)
                batch.update(outputs)
                all_losses = self.criterion(**batch)
                batch.update(all_losses)
        else:
            outputs = self.model(**batch)
            batch.update(outputs)
            all_losses = self.criterion(**batch)
            batch.update(all_losses)

        if self.is_train:
            self._accum_counter += 1
            loss = batch["loss"] / float(self.grad_accum_steps)

            if self.use_scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            do_step = (self._accum_counter % self.grad_accum_steps) == 0

            if do_step:
                if self.use_scaler:
                    self.scaler.unscale_(self.optimizer)

                # Логируем grad_norm только на шаге оптимизатора
                grad_norm = self._clip_grad_norm()
                batch["grad_norm"] = grad_norm
                metrics.update("grad_norm", grad_norm)
                
                if self.use_scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self._scheduler_step_batch()
                self.optimizer.zero_grad(set_to_none=True)

        # метрики по лоссам всегда
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        # остальные метрики по режиму
        for met in metric_funcs:
            metrics.update(met.name, met(**batch))

        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.
        """
        # спектры логируем редко
        if self._should_log_specs(batch_idx, mode):
            self.log_spectrogram(batch)

        # Предсказания логируем только на inference
        if mode != "train":
            self.log_predictions(**batch)

    def _unwrap_dataset(self, ds):
        """
        Возвращает «базовый» датасет, разворачивая типичные обертки.
        """
        while hasattr(ds, "dataset"):
            ds = ds.dataset
        return ds

    def _should_log_specs(self, batch_idx, mode):
        """
        Определяет, нужно ли логировать спектрограммы на текущем шаге.
        """
        if mode != "train":
            return False
        if batch_idx != 0:
            return False
        every = int(self.config.trainer.get("log_specs_every_n_epochs", 5))
        if every <= 0:
            return False
        return (self._last_epoch % every == 0)

    def log_spectrogram(self, batch):
        """
        Логирует набор спектрограмм
        """
        ds0 = self._unwrap_dataset(self.train_dataset)
        if not hasattr(ds0, "get_spectrogram"):
            return
        ds = cast(BaseDataset, ds0)

        with torch.no_grad():
            a_clean = batch["audio_orig"][0].detach().cpu()
            a_aug   = batch["audio"][0].detach().cpu()

            mel_clean_raw = ds.get_spectrogram(a_clean)
            mel_aug_raw   = ds.get_spectrogram(a_aug)

            mel_proc = None
            if "spectrogram_pre_batch" in batch:
                mel_proc = batch["spectrogram_pre_batch"][0].detach().cpu()

        mel_final = batch["spectrogram"][0].detach().cpu()

        img = plot_spectrogram_grid(
            [mel_clean_raw, mel_aug_raw, mel_proc if mel_proc is not None else mel_final, mel_final],
            titles=["clean_mel_raw", "wav_aug_mel_raw", "wav_aug_mel_proc", "final_to_model"],
        )

        self.writer.add_image("spec/pipeline", img)

    def log_predictions(
        self,
        text=None,
        log_probs=None,
        log_probs_length=None,
        audio_path=None,
        examples_to_log=10,
        **kwargs,
    ):
        if log_probs is None or log_probs_length is None:
            return

        # decoding config
        dcfg = parse_decoding_cfg(self.config)
        self.decode_type = dcfg.decode_type
        self.beam_size = dcfg.beam_size
        self.topk_per_timestep = dcfg.topk_per_timestep
        self.beam_threshold = dcfg.beam_threshold
        self.save_both_decodes = dcfg.save_both_decodes

        log_probs_length_cpu = log_probs_length.detach().cpu()
        log_probs_cpu = log_probs.detach().cpu()

        limit = min(len(log_probs_cpu), examples_to_log)
        lengths_list = log_probs_length_cpu[:limit].tolist()

        # Argmax (всегда)
        argmax_inds = log_probs_cpu[:limit].argmax(-1).numpy()
        argmax_inds = [inds[: int(L)] for inds, L in zip(argmax_inds, lengths_list)]
        argmax_texts = [self.text_encoder.ctc_decode(inds) for inds in argmax_inds]

        # Beam Search
        beam_texts = None
        if self.beam_size is not None and self.beam_size > 1:
            beam_texts = decode_beam(
                text_encoder=self.text_encoder,
                log_probs=log_probs_cpu[:limit],                 # [limit, T, V]
                lengths=log_probs_length_cpu[:limit],            # Tensor [limit]
                beam_size=int(self.beam_size),
                topk_per_timestep=self.topk_per_timestep,
                beam_threshold=float(self.beam_threshold),
            )

        if audio_path is None:
            audio_path = [f"sample_{i}" for i in range(len(argmax_texts))]

        if text is None:
            text = [""] * len(argmax_texts)

        rows = {}
        for i in range(limit):
            target = self.text_encoder.normalize_text(text[i])
            pred_argmax = argmax_texts[i]

            wer_argmax = calc_wer(target, pred_argmax) * 100
            cer_argmax = calc_cer(target, pred_argmax) * 100

            row_dict = {
                "target": target,
                "argmax_pred": pred_argmax,
                "wer_argmax": wer_argmax,
                "cer_argmax": cer_argmax,
            }

            if beam_texts is not None:
                pred_beam = beam_texts[i]
                row_dict["beam_pred"] = pred_beam
                row_dict["wer_beam"] = calc_wer(target, pred_beam) * 100
                row_dict["cer_beam"] = calc_cer(target, pred_beam) * 100

            rows[Path(audio_path[i]).name] = row_dict

        self.writer.add_table(
            "predictions", pd.DataFrame.from_dict(rows, orient="index")
        )
