from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

import hydra
import torch
from numpy import inf
from omegaconf import OmegaConf
from torch.nn import Module
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CyclicLR,
    LRScheduler,
    OneCycleLR,
    ReduceLROnPlateau,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.datasets.data_utils import inf_loop
from src.metrics.tracker import MetricTracker
from src.utils.batch_utils import apply_transforms_to_tensors, move_tensors_to_device
from src.utils.timing import log_examples_per_sec


class BaseTrainer(ABC):
    """
    Base class for all trainers.
    """

    def __init__(
        self,
        model: Module,
        criterion: Module,
        metrics: Dict[str, list],
        optimizer: Optimizer,
        lr_scheduler: Optional[LRScheduler],
        text_encoder,
        config,
        device: str,
        dataloaders: Dict[str, DataLoader],
        logger,
        writer,
        epoch_len: Optional[int] = None,
        skip_oom: bool = True,
        batch_transforms: Optional[Dict[str, Dict[str, Callable]]] = None,
    ):
        """
        Args:
            model (nn.Module): PyTorch model.
            criterion (nn.Module): loss function for model training.
            metrics (dict): dict with the definition of metrics for training
                (metrics[train]) and inference (metrics[inference]). Each
                metric is an instance of src.metrics.BaseMetric.
            optimizer (Optimizer): optimizer for the model.
            lr_scheduler (LRScheduler): learning rate scheduler for the
                optimizer.
            text_encoder (CTCTextEncoder): text encoder.
            config (DictConfig): experiment config containing training config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            logger (Logger): logger that logs output.
            writer (WandBWriter | CometMLWriter): experiment tracker.
            epoch_len (int | None): number of steps in each epoch for
                iteration-based training. If None, use epoch-based
                training (len(dataloader)).
            skip_oom (bool): skip batches with the OutOfMemory error.
            batch_transforms (dict[Callable] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
        """
        self.is_train = True

        self.config = config
        self.cfg_trainer = self.config.trainer

        self.device = device
        self.skip_oom = skip_oom

        self.logger = logger
        self.writer = writer
        self.log_step = self.cfg_trainer.get("log_step", 50)

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.text_encoder = text_encoder

        if batch_transforms is None:
            batch_transforms = {"train": {}, "inference": {}}
        self.batch_transforms = batch_transforms

        # Dataloaders
        self.train_loader = dataloaders["train"]
        self.train_dataset = self.train_loader.dataset

        bs = (
            getattr(self.train_loader, "batch_size", None)
            or getattr(self.config.dataloader, "batch_size", None)
            or 1
        )

        self.train_batch_size = int(bs)

        self.train_iter: Iterable = self.train_loader
        if epoch_len is None:
            # обычный итератор
            self.epoch_len = len(self.train_loader)
        else:
            # бесконечный итератор
            self.epoch_len = int(epoch_len)
            self.train_iter = inf_loop(self.train_loader)

        self.evaluation_dataloaders = {
            k: v for k, v in dataloaders.items() if k != "train"
        }

        # Epochs
        self._last_epoch = 0
        self.start_epoch = 1
        self.epochs = int(self.cfg_trainer.n_epochs)

        self.total_steps = int(self.epochs) * int(self.epoch_len)

        # Monitoring
        self.save_period = int(self.cfg_trainer.save_period)
        self.monitor = self.cfg_trainer.get("monitor", "off")

        if self.monitor == "off":
            self.mnt_mode = "off"
            self.mnt_best = 0.0
            self.early_stop = inf
            self.mnt_metric = None
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ["min", "max"]
            self.mnt_best = inf if self.mnt_mode == "min" else -inf
            self.early_stop = self.cfg_trainer.get("early_stop", inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        # Metrics
        self.metrics = metrics

        self.train_metrics = MetricTracker(
            *self.config.writer.loss_names,
            "grad_norm",
            *[m.name for m in self.metrics["train"]],
            writer=self.writer,
        )
        self.evaluation_metrics = MetricTracker(
            *self.config.writer.loss_names,
            *[m.name for m in self.metrics["inference"]],
            writer=self.writer,
        )
        self._need_spec_log: bool = False

        # Checkpoint dir
        save_dir_abs = hydra.utils.to_absolute_path(self.cfg_trainer.save_dir)
        self.checkpoint_dir = Path(save_dir_abs) / config.writer.run_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Resume / from_pretrained
        if self.cfg_trainer.get("resume_from") is not None:
            resume_path = hydra.utils.to_absolute_path(self.cfg_trainer.resume_from)
            self._resume_checkpoint(resume_path)

        if self.cfg_trainer.get("from_pretrained") is not None:
            pretrained_path = hydra.utils.to_absolute_path(self.cfg_trainer.get("from_pretrained"))
            self._from_pretrained(pretrained_path)

    def train(self):
        """
        Wrapper around training process to save model on keyboard interrupt.
        """
        try:
            self._train_process()
        except KeyboardInterrupt as e:
            self.logger.info("Saving model on keyboard interrupt")
            self._save_checkpoint(self._last_epoch, save_best=False)
            raise e

    def _train_process(self):
        """
        Full training logic:
        Training model for an epoch, evaluating it on non-train partitions,
        and monitoring the performance improvement (for early stopping
        and saving the best checkpoint).
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._last_epoch = epoch
            logs: Dict[str, float | int] = {"epoch": epoch}
            logs.update(self._train_epoch(epoch))

            for key, value in logs.items():
                self.logger.info(f"    {key:15s}: {value}")

            best, stop_process, not_improved_count = self._monitor_performance(
                logs, not_improved_count
            )

            # Epoch-level scheduler step (unified)
            self._scheduler_step_epoch(logs)

            if epoch % self.save_period == 0 or best:
                self._save_checkpoint(epoch, save_best=best, only_best=True)

            if stop_process:
                break

    def _scheduler_step_epoch(self, logs):
        """Step scheduler at end of epoch (only for epoch-based schedulers)"""
        if self.lr_scheduler is None:
            return

        # These step per batch, not per epoch
        if isinstance(self.lr_scheduler, (OneCycleLR, CyclicLR)):
            return

        # ReduceLROnPlateau requires a metric
        if isinstance(self.lr_scheduler, ReduceLROnPlateau):
            if self.mnt_mode != "off" and self.mnt_metric in logs:
                self.lr_scheduler.step(logs[self.mnt_metric])
            elif "loss" in logs:
                self.lr_scheduler.step(logs["loss"])
            return

        # Default: step without arguments
        self.lr_scheduler.step()

    def _scheduler_step_batch(self):
        if self.lr_scheduler is None:
            return
        if isinstance(self.lr_scheduler, ReduceLROnPlateau):
            return
        if isinstance(self.lr_scheduler, (OneCycleLR, CyclicLR)):
            self.lr_scheduler.step()

    def _n_train_examples(self) -> int:
        return int(self.epoch_len) * int(self.train_batch_size)

    def _n_eval_examples(self, dataloader: DataLoader) -> int:
        bs = int(getattr(dataloader, "batch_size", None) or 1)
        return int(len(dataloader)) * bs

    def _should_log_specs(self, batch_idx: int, mode: str) -> bool:
        """Переопределяется в Trainer"""
        return False

    @log_examples_per_sec(
        get_n_examples=lambda self, epoch: self._n_train_examples(),
        get_mode=lambda self, epoch: "train",
    )
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.is_train = True
        self.model.train()

        # reset epoch+window
        self.train_metrics.reset()

        progress_bar = tqdm(
            enumerate(self.train_iter), desc="train", total=self.epoch_len
        )

        for batch_idx, batch in progress_bar:
            self._need_spec_log = self._should_log_specs(batch_idx, mode="train")

            try:
                batch = self.process_batch(batch, metrics=self.train_metrics)
            except torch.cuda.OutOfMemoryError as e:
                if self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    torch.cuda.empty_cache()
                    continue
                raise e

            # Logging each log_step
            if self.log_step is not None and batch_idx % self.log_step == 0:
                if self.writer is not None:
                    global_step0 = (epoch - 1) * self.epoch_len + batch_idx
                    global_step1 = global_step0 + 1

                    self.writer.set_step(global_step0, mode="train")

                    train_progress = 100.0 * global_step1 / self.total_steps
                    self.writer.add_scalar("trainer/progress", train_progress)

                    current_lr = self._get_current_lr()
                    self.writer.add_scalar("learning_rate", float(current_lr))

                    self._log_scalars_window(self.train_metrics)
                    self._log_batch(batch_idx, batch, mode="train")
                    self.train_metrics.reset_window()

            if batch_idx + 1 >= self.epoch_len:
                break

        if self.writer is not None:
            end_step = epoch * self.epoch_len
            self.writer.set_step(end_step, mode="train")

            self.writer.add_scalar("epoch", epoch)
            self.writer.add_scalar("trainer/progress", 100.0 * float(epoch) / float(self.epochs))

            self._log_scalars_epoch(self.train_metrics)

        logs: Dict[str, float] = self.train_metrics.result()

        # Validation/test
        for part, dataloader in self.evaluation_dataloaders.items():
            val_logs = self._evaluation_epoch(epoch, part, dataloader)
            logs.update({f"{part}_{name}": value for name, value in val_logs.items()})

        return logs

    @log_examples_per_sec(
        get_n_examples=lambda self, epoch, part, dataloader: self._n_eval_examples(
            dataloader
        ),
        get_mode=lambda self, epoch, part, dataloader: str(part),
    )
    def _evaluation_epoch(
        self, epoch: int, part: str, dataloader: DataLoader
    ) -> Dict[str, float]:
        self.is_train = False
        self.model.eval()

        # reset epoch+window
        self.evaluation_metrics.reset()

        last_batch: Optional[Dict[str, Any]] = None
        last_batch_idx = 0

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader), desc=part, total=len(dataloader)
            ):
                self._need_spec_log = self._should_log_specs(batch_idx, mode=part)
                last_batch_idx = batch_idx
                last_batch = self.process_batch(batch, metrics=self.evaluation_metrics)

            if self.writer is not None:
                self.writer.set_step(epoch * self.epoch_len, mode=part)
                self.writer.add_scalar("trainer/progress", 100.0 * float(epoch) / float(self.epochs))
                self._log_scalars_epoch(self.evaluation_metrics)

                if last_batch is not None:
                    self._log_batch(last_batch_idx, last_batch, mode=part)

        return self.evaluation_metrics.result()

    def _monitor_performance(
        self, logs: Dict[str, float | int], not_improved_count: int
    ):
        """
        Check if there is an improvement in the metrics. Used for early
        stopping and saving the best checkpoint.

        Args:
            logs (dict): logs after training and evaluating the model for
                an epoch.
            not_improved_count (int): the current number of epochs without
                improvement.
        Returns:
            best (bool): if True, the monitored metric has improved.
            stop_process (bool): if True, stop the process (early stopping).
                The metric did not improve for too much epochs.
            not_improved_count (int): updated number of epochs without
                improvement.
        """
        best = False
        stop_process = False

        if self.mnt_mode == "off" or self.mnt_metric is None:
            return best, stop_process, not_improved_count

        try:
            current = logs[self.mnt_metric]
        except KeyError:
            self.logger.warning(
                f"Warning: Metric '{self.mnt_metric}' is not found. Model performance monitoring is disabled."
            )
            self.mnt_mode = "off"
            return best, stop_process, not_improved_count

        # check whether model performance improved or not,
        # according to specified metric(mnt_metric)
        improved = (
            current <= self.mnt_best
            if self.mnt_mode == "min"
            else current >= self.mnt_best
        )

        if improved:
            self.mnt_best = current
            not_improved_count = 0
            best = True
        else:
            not_improved_count += 1

        if not_improved_count >= self.early_stop:
            self.logger.info(
                "Validation performance didn't improve for {} epochs. "
                "Training stops.".format(self.early_stop)
            )
            stop_process = True

        return best, stop_process, not_improved_count

    def move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return move_tensors_to_device(batch, self.device, self.cfg_trainer.device_tensors)

    def transform_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        # do batch transforms on device
        transform_type = "train" if self.is_train else "inference"
        transforms = (self.batch_transforms or {}).get(transform_type) or {}
        return apply_transforms_to_tensors(batch, transforms)

    def _clip_grad_norm(self):
        """
        Clips the gradient norm by the value defined in
        config.trainer.max_grad_norm
        """
        max_gn = self.cfg_trainer.get("max_grad_norm", None)
        if max_gn is not None:
            norm = clip_grad_norm_(self.model.parameters(), max_gn)
            return float(norm)
        return float(self._get_grad_norm())

    @torch.no_grad()
    def _get_grad_norm(self, norm_type=2):
        """
        Calculates the gradient norm for logging.

        Args:
            norm_type (float | str | None): the order of the norm.
        Returns:
            total_norm (float): the calculated norm.
        """
        parameters = list(self.model.parameters())
        grads = [p.grad.detach() for p in parameters if p.grad is not None]
        if len(grads) == 0:
            return 0.0

        total_norm = torch.norm(
            torch.stack([torch.norm(g, norm_type) for g in grads]),
            norm_type,
        )
        return total_norm.item()

    def _get_current_lr(self) -> float:
        if self.lr_scheduler is not None:
            try:
                return float(self.lr_scheduler.get_last_lr()[0])
            except Exception:
                pass
        return float(self.optimizer.param_groups[0]["lr"])

    def _log_scalars_window(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(
                f"{metric_name}", float(metric_tracker.avg_window(metric_name))
            )

    def _log_scalars_epoch(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(
                f"{metric_name}", float(metric_tracker.avg(metric_name))
            )

    @abstractmethod
    def process_batch(self, batch, metrics):
        """
        Abstract method. Should be defined in the nested Trainer Class.
        Process batch through model, calculate loss and metrics.

        Args:
            batch (dict): dict-based batch containing the data from dataloader.
            metrics (MetricTracker): metrics tracker to update.

        Returns:
            batch (dict): dict-based batch with added model outputs and loss.
        """
        raise NotImplementedError()

    @abstractmethod
    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Abstract method. Should be defined in the nested Trainer Class.
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.
        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        raise NotImplementedError()

    def _log_scalars(self, metric_tracker: MetricTracker):
        """
        Wrapper around the writer 'add_scalar' to log all metrics.

        Args:
            metric_tracker (MetricTracker): calculated metrics.
        """
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(
                f"{metric_name}", float(metric_tracker.avg(metric_name))
            )

    def _save_checkpoint(
        self, epoch: int, save_best: bool = False, only_best: bool = False
    ):
        """
        Save the checkpoints.

        Args:
            epoch (int): current epoch number.
            save_best (bool): if True, rename the saved checkpoint to 'model_best.pth'.
            only_best (bool): if True and the checkpoint is the best, save it only as
                'model_best.pth'(do not duplicate the checkpoint as
                checkpoint-epochEpochNumber.pth)
        """
        arch = type(self.model).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict()
            if self.lr_scheduler is not None
            else None,
            "monitor_best": self.mnt_best,
            "config": OmegaConf.to_container(self.config, resolve=True, enum_to_str=True),
        }

        filename = str(self.checkpoint_dir / f"checkpoint-epoch{epoch}.pth")
        if not (only_best and save_best):
            torch.save(state, filename)
            if (
                getattr(self.config.writer, "log_checkpoints", False)
                and self.writer is not None
            ):
                self.writer.add_checkpoint(filename, str(self.checkpoint_dir.parent))
            self.logger.info(f"Saving checkpoint: {filename} ...")

        if save_best:
            best_path = str(self.checkpoint_dir / "model_best.pth")
            torch.save(state, best_path)
            if (
                getattr(self.config.writer, "log_checkpoints", False)
                and self.writer is not None
            ):
                self.writer.add_checkpoint(best_path, str(self.checkpoint_dir.parent))
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from a saved checkpoint (in case of server crash, etc.).
        The function loads state dicts for everything, including model,
        optimizers, etc.

        Notice that the checkpoint should be located in the current experiment
        saved directory (where all checkpoints are saved in '_save_checkpoint').

        Args:
            resume_path (str): Path to the checkpoint to be resumed.
        """
        resume_path = str(resume_path)
        self.logger.info(f"Loading checkpoint: {resume_path} ...")
        checkpoint = torch.load(resume_path, map_location=self.device)

        self.start_epoch = int(checkpoint["epoch"]) + 1
        self.mnt_best = checkpoint.get("monitor_best", self.mnt_best)

        self.model.load_state_dict(checkpoint["state_dict"])

        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        if self.lr_scheduler is not None and checkpoint.get("lr_scheduler") is not None:
            try:
                self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            except Exception:
                self.logger.warning(
                    "Warning: failed to load lr_scheduler state_dict. Skipping."
                )

        self.logger.info(
            f"Checkpoint loaded. Resume training from epoch {self.start_epoch}"
        )

    def _from_pretrained(self, pretrained_path):
        """
        Init model with weights from pretrained pth file.

        Notice that 'pretrained_path' can be any path on the disk. It is not
        necessary to locate it in the experiment saved dir. The function
        initializes only the model.

        Args:
            pretrained_path (str): path to the model state dict.
        """
        pretrained_path = str(pretrained_path)
        self.logger.info(f"Loading model weights from: {pretrained_path} ...")
        checkpoint = torch.load(
            pretrained_path,
            map_location=self.device,
            weights_only=False,     # Может давать конфликты со старыми версиями torch
                                    # Оставлено для совместимости с имеющимися моделями
                                    # В перспективе удалить
        )

        if checkpoint.get("state_dict") is not None:
            self.model.load_state_dict(checkpoint["state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
