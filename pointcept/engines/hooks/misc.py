"""
Misc Hook

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import sys
import glob
import os
import shutil
import time
import gc
import wandb
import torch
import torch.utils.data
from collections import OrderedDict
import torch.nn.functional as F
import numpy as np

if sys.version_info >= (3, 10):
    from collections.abc import Sequence
else:
    from collections import Sequence
from pointcept.utils.timer import Timer
from pointcept.utils.comm import is_main_process, synchronize
from pointcept.utils.cache import shared_dict
from pointcept.utils.scheduler import CosineScheduler
import pointcept.utils.comm as comm

from .default import HookBase
from .builder import HOOKS


@HOOKS.register_module()
class IterationTimer(HookBase):
    def __init__(self, warmup_iter=1):
        self._warmup_iter = warmup_iter
        self._start_time = time.perf_counter()
        self._iter_timer = Timer()
        self._remain_iter = 0

    def before_train(self):
        self._start_time = time.perf_counter()
        _remain_epoch = self.trainer.max_epoch - self.trainer.start_epoch
        self._remain_iter = _remain_epoch * len(self.trainer.train_loader)

    def before_epoch(self):
        self._iter_timer.reset()

    def before_step(self):
        data_time = self._iter_timer.seconds()
        self.trainer.storage.put_scalar("data_time", data_time)

    def after_step(self):
        batch_time = self._iter_timer.seconds()
        self._iter_timer.reset()
        self.trainer.storage.put_scalar("batch_time", batch_time)
        self._remain_iter -= 1
        remain_time = self._remain_iter * self.trainer.storage.history("batch_time").avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = "{:02d}:{:02d}:{:02d}".format(int(t_h), int(t_m), int(t_s))
        if "iter_info" in self.trainer.comm_info.keys():
            info = (
                "Data {data_time_val:.3f} ({data_time_avg:.3f}) "
                "Batch {batch_time_val:.3f} ({batch_time_avg:.3f}) "
                "Remain {remain_time} ".format(
                    data_time_val=self.trainer.storage.history("data_time").val,
                    data_time_avg=self.trainer.storage.history("data_time").avg,
                    batch_time_val=self.trainer.storage.history("batch_time").val,
                    batch_time_avg=self.trainer.storage.history("batch_time").avg,
                    remain_time=remain_time,
                )
            )
            self.trainer.comm_info["iter_info"] += info
        if self.trainer.comm_info["iter"] <= self._warmup_iter:
            self.trainer.storage.history("data_time").reset()
            self.trainer.storage.history("batch_time").reset()


@HOOKS.register_module()
class InformationWriter(HookBase):
    def __init__(self):
        self.curr_iter = 0
        self.model_output_keys = []

    def before_train(self):
        self.trainer.comm_info["iter_info"] = ""
        self.curr_iter = self.trainer.start_epoch * len(self.trainer.train_loader)
        if self.trainer.writer is not None and self.trainer.cfg.enable_wandb:
            wandb.define_metric("params/*", step_metric="Iter")
            wandb.define_metric("train_batch/*", step_metric="Iter")
            wandb.define_metric("train/*", step_metric="Epoch")

    def before_step(self):
        self.curr_iter += 1
        info = "Train: [{epoch}/{max_epoch}][{iter}/{max_iter}] ".format(
            epoch=self.trainer.epoch + 1,
            max_epoch=self.trainer.max_epoch,
            iter=self.trainer.comm_info["iter"] + 1,
            max_iter=len(self.trainer.train_loader),
        )
        self.trainer.comm_info["iter_info"] += info

    def after_step(self):
        if "model_output_dict" in self.trainer.comm_info.keys():
            model_output_dict = self.trainer.comm_info["model_output_dict"]
            self.model_output_keys = model_output_dict.keys()
            for key in self.model_output_keys:
                self.trainer.storage.put_scalar(key, model_output_dict[key].item())

        for key in self.model_output_keys:
            self.trainer.comm_info["iter_info"] += "{key}: {value:.4f} ".format(
                key=key, value=self.trainer.storage.history(key).val
            )
        lr = self.trainer.optimizer.state_dict()["param_groups"][0]["lr"]
        self.trainer.comm_info["iter_info"] += "Lr: {lr:.5f}".format(lr=lr)
        self.trainer.logger.info(self.trainer.comm_info["iter_info"])
        self.trainer.comm_info["iter_info"] = ""  # reset iter info
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("params/lr", lr, self.curr_iter)
            for key in self.model_output_keys:
                self.trainer.writer.add_scalar(
                    "train_batch/" + key,
                    self.trainer.storage.history(key).val,
                    self.curr_iter,
                )
            if self.trainer.cfg.enable_wandb:

                wandb.log(
                    {"Iter": self.curr_iter, "params/lr": lr}, step=self.curr_iter
                )
                for key in self.model_output_keys:
                    wandb.log(
                        {
                            "Iter": self.curr_iter,
                            f"train_batch/{key}": self.trainer.storage.history(key).val,
                        },
                        step=wandb.run.step,
                    )

    def after_epoch(self):
        epoch_info = "Train result: "
        for key in self.model_output_keys:
            epoch_info += "{key}: {value:.4f} ".format(
                key=key, value=self.trainer.storage.history(key).avg
            )
        self.trainer.logger.info(epoch_info)
        if self.trainer.writer is not None:
            for key in self.model_output_keys:
                self.trainer.writer.add_scalar(
                    "train/" + key,
                    self.trainer.storage.history(key).avg,
                    self.trainer.epoch + 1,
                )

            if self.trainer.cfg.enable_wandb:

                for key in self.model_output_keys:
                    wandb.log(
                        {
                            "Epoch": self.trainer.epoch + 1,
                            f"train/{key}": self.trainer.storage.history(key).avg,
                        },
                        step=wandb.run.step,
                    )


@HOOKS.register_module()
class CheckpointSaver(HookBase):
    def __init__(self, save_freq=None):
        self.save_freq = save_freq  # None or int, None indicate only save model last

    def after_epoch(self):
        if is_main_process():
            is_best = False
            if self.trainer.cfg.evaluate:
                current_metric_value = self.trainer.comm_info["current_metric_value"]
                current_metric_name = self.trainer.comm_info["current_metric_name"]
                if current_metric_value > self.trainer.best_metric_value:
                    self.trainer.best_metric_value = current_metric_value
                    is_best = True
                    self.trainer.logger.info(
                        "Best validation {} updated to: {:.4f}".format(
                            current_metric_name, current_metric_value
                        )
                    )
                self.trainer.logger.info(
                    "Currently Best {}: {:.4f}".format(
                        current_metric_name, self.trainer.best_metric_value
                    )
                )

            filename = os.path.join(
                self.trainer.cfg.save_path, "model", "model_last.pth"
            )
            self.trainer.logger.info("Saving checkpoint to: " + filename)
            torch.save(
                {
                    "epoch": self.trainer.epoch + 1,
                    "state_dict": self.trainer.model.state_dict(),
                    "optimizer": self.trainer.optimizer.state_dict(),
                    "scheduler": self.trainer.scheduler.state_dict(),
                    "scaler": (
                        self.trainer.scaler.state_dict()
                        if self.trainer.cfg.enable_amp
                        else None
                    ),
                    "best_metric_value": self.trainer.best_metric_value,
                },
                filename + ".tmp",
            )
            os.replace(filename + ".tmp", filename)
            if is_best:
                shutil.copyfile(
                    filename,
                    os.path.join(self.trainer.cfg.save_path, "model", "model_best.pth"),
                )
            if self.save_freq and (self.trainer.epoch + 1) % self.save_freq == 0:
                shutil.copyfile(
                    filename,
                    os.path.join(
                        self.trainer.cfg.save_path,
                        "model",
                        f"epoch_{self.trainer.epoch + 1}.pth",
                    ),
                )


@HOOKS.register_module()
class CheckpointLoader(HookBase):
    def __init__(self, keywords="", replacement=None, strict=False):
        self.keywords = keywords
        self.replacement = replacement if replacement is not None else keywords
        self.strict = strict

    def before_train(self):
        self.trainer.logger.info("=> Loading checkpoint & weight ...")
        if self.trainer.cfg.weight and os.path.isfile(self.trainer.cfg.weight):
            self.trainer.logger.info(f"Loading weight at: {self.trainer.cfg.weight}")
            checkpoint = torch.load(
                self.trainer.cfg.weight,
                map_location=lambda storage, loc: storage.cuda(),
                weights_only=False,
            )
            self.trainer.logger.info(
                f"Loading layer weights with keyword: {self.keywords}, "
                f"replace keyword with: {self.replacement}"
            )
            weight = OrderedDict()
            for key, value in checkpoint["state_dict"].items():
                if not key.startswith("module."):
                    key = "module." + key  # xxx.xxx -> module.xxx.xxx
                # Now all keys contain "module." no matter DDP or not.
                if self.keywords in key:
                    key = key.replace(self.keywords, self.replacement, 1)
                if comm.get_world_size() == 1:
                    key = key[7:]  # module.xxx.xxx -> xxx.xxx
                weight[key] = value
            load_state_info = self.trainer.model.load_state_dict(
                weight, strict=self.strict
            )
            self.trainer.logger.info(f"Missing keys: {load_state_info[0]}")
            if self.trainer.cfg.resume:
                self.trainer.logger.info(
                    f"Resuming train at eval epoch: {checkpoint['epoch']}"
                )
                self.trainer.start_epoch = checkpoint["epoch"]
                self.trainer.best_metric_value = checkpoint["best_metric_value"]
                self.trainer.optimizer.load_state_dict(checkpoint["optimizer"])
                self.trainer.scheduler.load_state_dict(checkpoint["scheduler"])
                if self.trainer.cfg.enable_amp:
                    self.trainer.scaler.load_state_dict(checkpoint["scaler"])
        else:
            self.trainer.logger.info(f"No weight found at: {self.trainer.cfg.weight}")


@HOOKS.register_module()
class PreciseEvaluator(HookBase):
    def __init__(self, test_last=False):
        self.test_last = test_last

    def after_train(self):
        from pointcept.engines.test import TESTERS

        self.trainer.logger.info(
            ">>>>>>>>>>>>>>>> Start Precise Evaluation >>>>>>>>>>>>>>>>"
        )
        torch.cuda.empty_cache()
        cfg = self.trainer.cfg
        test_cfg = dict(cfg=cfg, model=self.trainer.model, **cfg.test)
        tester = TESTERS.build(test_cfg)
        if self.test_last:
            self.trainer.logger.info("=> Testing on model_last ...")
        else:
            self.trainer.logger.info("=> Testing on model_best ...")
            best_path = os.path.join(
                self.trainer.cfg.save_path, "model", "model_best.pth"
            )
            checkpoint = torch.load(best_path, weights_only=False)
            weight = OrderedDict()
            for key, value in checkpoint["state_dict"].items():
                if not key.startswith("module."):
                    key = "module." + key  # xxx.xxx -> module.xxx.xxx
                # Now all keys contain "module." no matter DDP or not.
                if comm.get_world_size() == 1:
                    key = key[7:]  # module.xxx.xxx -> xxx.xxx
                weight[key] = value
            tester.model.load_state_dict(weight, strict=True)
        tester.test()


@HOOKS.register_module()
class DataCacheOperator(HookBase):
    def __init__(self, data_root, split):
        self.data_root = data_root
        self.split = split
        self.data_list = self.get_data_list()

    def get_data_list(self):
        if isinstance(self.split, str):
            data_list = glob.glob(os.path.join(self.data_root, self.split))
        elif isinstance(self.split, Sequence):
            data_list = []
            for split in self.split:
                data_list += glob.glob(os.path.join(self.data_root, split))
        else:
            raise NotImplementedError
        return data_list

    def get_cache_name(self, data_path):
        data_name = data_path.replace(os.path.dirname(self.data_root), "")
        return "pointcept" + data_name.replace(os.path.sep, "-")

    def before_train(self):
        self.trainer.logger.info(
            f"=> Caching dataset: {self.data_root}, split: {self.split} ..."
        )
        if is_main_process():
            dataset = self.trainer.train_loader.dataset
            for i in range(len(dataset)):
                data_dict = dataset[i]
                name = data_dict["name"]
                shared_dict(f"Pointcept-{name}", data_dict)
        synchronize()


@HOOKS.register_module()
class RuntimeProfiler(HookBase):
    def __init__(
        self,
        forward=True,
        backward=True,
        interrupt=False,
        warm_up=2,
        sort_by="cuda_time_total",
        row_limit=30,
    ):
        self.forward = forward
        self.backward = backward
        self.interrupt = interrupt
        self.warm_up = warm_up
        self.sort_by = sort_by
        self.row_limit = row_limit

    def before_train(self):
        self.trainer.logger.info("Profiling runtime ...")
        from torch.profiler import profile, record_function, ProfilerActivity

        for i, input_dict in enumerate(self.trainer.train_loader):
            if i == self.warm_up + 1:
                break
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            if self.forward:
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True,
                ) as forward_prof:
                    with record_function("model_inference"):
                        output_dict = self.trainer.model(input_dict)
            else:
                output_dict = self.trainer.model(input_dict)
            loss = output_dict["loss"]
            if self.backward:
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True,
                ) as backward_prof:
                    with record_function("model_inference"):
                        loss.backward()
            self.trainer.logger.info(f"Profile: [{i + 1}/{self.warm_up + 1}]")
        if self.forward:
            self.trainer.logger.info(
                "Forward profile: \n"
                + str(
                    forward_prof.key_averages().table(
                        sort_by=self.sort_by, row_limit=self.row_limit
                    )
                )
            )
            forward_prof.export_chrome_trace(
                os.path.join(self.trainer.cfg.save_path, "forward_trace.json")
            )

        if self.backward:
            self.trainer.logger.info(
                "Backward profile: \n"
                + str(
                    backward_prof.key_averages().table(
                        sort_by=self.sort_by, row_limit=self.row_limit
                    )
                )
            )
            backward_prof.export_chrome_trace(
                os.path.join(self.trainer.cfg.save_path, "backward_trace.json")
            )
        if self.interrupt:
            sys.exit(0)


@HOOKS.register_module()
class RuntimeProfilerV2(HookBase):
    def __init__(
        self,
        interrupt=False,
        wait=1,
        warmup=1,
        active=10,
        repeat=1,
        sort_by="cuda_time_total",
        row_limit=30,
    ):
        self.interrupt = interrupt
        self.wait = wait
        self.warmup = warmup
        self.active = active
        self.repeat = repeat
        self.sort_by = sort_by
        self.row_limit = row_limit

    def before_train(self):
        self.trainer.logger.info("Profiling runtime ...")
        from torch.profiler import (
            profile,
            record_function,
            ProfilerActivity,
            schedule,
            tensorboard_trace_handler,
        )

        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(
                wait=self.wait,
                warmup=self.warmup,
                active=self.active,
                repeat=self.repeat,
            ),
            on_trace_ready=tensorboard_trace_handler(self.trainer.cfg.save_path),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        prof.start()
        for i, input_dict in enumerate(self.trainer.train_loader):
            if i >= (self.wait + self.warmup + self.active) * self.repeat:
                break
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with record_function("model_forward"):
                output_dict = self.trainer.model(input_dict)
                loss = output_dict["loss"]
            with record_function("model_backward"):
                loss.backward()
            prof.step()
            self.trainer.logger.info(
                f"Profile: [{i + 1}/{(self.wait + self.warmup + self.active) * self.repeat}]"
            )
        self.trainer.logger.info(
            "Profile: \n"
            + str(
                prof.key_averages().table(
                    sort_by=self.sort_by, row_limit=self.row_limit
                )
            )
        )
        prof.stop()

        if self.interrupt:
            sys.exit(0)


@HOOKS.register_module()
class WeightDecaySchedular(HookBase):
    def __init__(
        self,
        base_value=0.04,
        final_value=0.2,
    ):
        self.base_value = base_value
        self.final_value = final_value
        self.scheduler = None

    def before_train(self):
        curr_step = self.trainer.start_epoch * len(self.trainer.train_loader)
        self.scheduler = CosineScheduler(
            base_value=self.base_value,
            final_value=self.final_value,
            total_iters=self.trainer.cfg.scheduler.total_steps,
        )
        self.scheduler.iter = curr_step

    def before_step(self):
        wd = self.scheduler.step()
        for param_group in self.trainer.optimizer.param_groups:
            param_group["weight_decay"] = wd
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("params/wd", wd, self.scheduler.iter)


@HOOKS.register_module()
class GarbageHandler(HookBase):
    def __init__(self, interval=150, disable_auto=True, empty_cache=False):
        self.interval = interval
        self.disable_auto = disable_auto
        self.empty_cache = empty_cache
        self.iter = 1

    def before_train(self):
        if self.disable_auto:
            gc.disable()
            self.trainer.logger.info("Disable automatic garbage collection")

    def before_epoch(self):
        self.iter = 1

    def after_step(self):
        if self.iter % self.interval == 0:
            gc.collect()
            if self.empty_cache:
                torch.cuda.empty_cache()
            self.trainer.logger.info("Garbage collected")
        self.iter += 1

    def after_train(self):
        gc.collect()
        torch.cuda.empty_cache()


@HOOKS.register_module()  
class DisplacementEvaluator(HookBase):
    METRICS = [
        "loss", "mae", "cos_dist",
        "bracket_mae", "incisal_mae", "outer_mae",
        "bracket_cos_dist", "incisal_cos_dist", "outer_cos_dist"
    ]

    def before_train(self):  
        # Define metric step for WandB  
        if self.trainer.writer is not None and self.trainer.cfg.enable_wandb:  
            wandb.define_metric("val/*", step_metric="Epoch")  
 
    def after_epoch(self):
        if self.trainer.cfg.evaluate:  
            self.eval()

    def after_train(self):
        best_loss = -self.trainer.best_metric_value
        self.trainer.logger.info(f"Best LOSS value: {best_loss:.6f}")

    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()

        metrics = self._compute_metrics()
        self._log_metrics(metrics)
        self._write_metrics(metrics)
        
        # Store NEGATIVE Loss so that "higher is better" logic works
        self.trainer.comm_info["current_metric_value"] = -metrics["loss"]
        self.trainer.comm_info["current_metric_name"] = "LOSS"
        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
    
    def _compute_metrics(self):
        """Compute average metrics across validation set."""
        metric_sums = {metric: 0.0 for metric in self.METRICS}
        count = 0
        
        for input_dict in self.trainer.val_loader:
            # Move tensors to GPU
            for key, value in input_dict.items():
                if isinstance(value, torch.Tensor):
                    input_dict[key] = value.cuda(non_blocking=True)

            # Forward pass
            with torch.no_grad():
                output_dict = self.trainer.model(input_dict)
                
            # Accumulate metrics
            for metric in self.METRICS:
                metric_value = output_dict[metric].item()
                metric_sums[metric] += metric_value
                # Log individual batch metrics for first three
                if metric in ["loss", "cos_dist", "mae"]:
                    self.trainer.storage.put_scalar(f"val_{metric}", metric_value)
            count += 1
        return {metric: metric_sums[metric] / count for metric in self.METRICS}
    
    def _log_metrics(self, metrics):
        """Log metrics to console."""
        self.trainer.logger.info(f"Loss: {metrics['loss']:.6f}")
        self.trainer.logger.info(f"MAE: {metrics['mae']:.6f}")
        self.trainer.logger.info(f"Bracket MAE: {metrics['bracket_mae']:.6f}")
        self.trainer.logger.info(f"Incisal MAE: {metrics['incisal_mae']:.6f}")
        self.trainer.logger.info(f"Outer MAE: {metrics['outer_mae']:.6f}")
        self.trainer.logger.info(f"Cosine Distance: {metrics['cos_dist']:.6f}")
        self.trainer.logger.info(f"Bracket Cosine Distance: {metrics['bracket_cos_dist']:.6f}")
        self.trainer.logger.info(f"Incisal Cosine Distance: {metrics['incisal_cos_dist']:.6f}")
        self.trainer.logger.info(f"Outer Cosine Distance: {metrics['outer_cos_dist']:.6f}")
    
    def _write_metrics(self, metrics):
        """Write metrics to TensorBoard and WandB."""
        if self.trainer.writer is None:
            return
        
        current_epoch = self.trainer.epoch + 1
        
        for metric, value in metrics.items():
            self.trainer.writer.add_scalar(f"val/{metric}", value, current_epoch)

        if self.trainer.cfg.enable_wandb:
            wandb_log_dict = {"Epoch": current_epoch}
            wandb_log_dict.update({f"val/{metric}": value for metric, value in metrics.items()})
            wandb.log(wandb_log_dict, step=wandb.run.step)


@HOOKS.register_module()
class OrientationEvaluator(HookBase):
    def before_train(self):
        if self.trainer.writer is not None and self.trainer.cfg.enable_wandb:
            wandb.define_metric("val/*", step_metric="Epoch")

    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            self.eval()

    def after_train(self):
        best_loss = -self.trainer.best_metric_value
        self.trainer.logger.info(f"Best orientation loss: {best_loss:.6f}")

    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()
        loss_sum = 0.0
        recon_sum = 0.0
        angle_sum = 0.0
        count = 0
        for input_dict in self.trainer.val_loader:
            for key, value in input_dict.items():
                if isinstance(value, torch.Tensor):
                    input_dict[key] = value.cuda(non_blocking=True)
            with torch.no_grad():
                output = self.trainer.model(input_dict)
            loss_sum += output["loss"].item()
            recon_sum += output["recon_loss"].item()
            angle_sum += output["angle_loss"].item()
            count += 1
        loss = loss_sum / max(count, 1)
        recon_loss = recon_sum / max(count, 1)
        angle_loss = angle_sum / max(count, 1)
        self.trainer.logger.info(
            f"Val result: loss/recon/angle {loss:.6f}/{recon_loss:.6f}/{angle_loss:.6f}"
        )
        current_epoch = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("val/loss", loss, current_epoch)
            self.trainer.writer.add_scalar("val/recon_loss", recon_loss, current_epoch)
            self.trainer.writer.add_scalar("val/angle_loss", angle_loss, current_epoch)
            if self.trainer.cfg.enable_wandb:
                wandb.log(
                    {
                        "Epoch": current_epoch,
                        "val/loss": loss,
                        "val/recon_loss": recon_loss,
                        "val/angle_loss": angle_loss,
                    },
                    step=wandb.run.step,
                )
        self.trainer.comm_info["current_metric_value"] = -loss
        self.trainer.comm_info["current_metric_name"] = "orientation_loss"
        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")


@HOOKS.register_module()  
class HeatmapEvaluator(HookBase):  
    def __init__(self):  
        pass  
      
    def before_train(self):  
        if self.trainer.writer is not None and self.trainer.cfg.enable_wandb:  
            wandb.define_metric("val/*", step_metric="Epoch")

    def after_epoch(self):  
        if self.trainer.cfg.evaluate:  
            self.eval()  
      
    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")  
        self.trainer.model.eval()  
        all_predictions = [] 
        for i, input_dict in enumerate(self.trainer.val_loader):  
            for key in input_dict.keys():  
                if isinstance(input_dict[key], torch.Tensor):  
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)  
              
            with torch.no_grad():  
                output_dict = self.trainer.model(input_dict)  
              
            # Get predictions and clamp to [0, 1]  
            heatmap_pred = output_dict["seg_logits"]
            heatmap_pred = torch.clamp(heatmap_pred, 0.0, 1.0)  

            all_predictions.append(heatmap_pred.cpu()) 
            # Get ground truth  
            segment = input_dict["segment"]
 
            # Handle inverse mapping if present  
            if "inverse" in input_dict.keys():  
                assert "origin_segment" in input_dict.keys()  
                heatmap_pred = heatmap_pred[input_dict["inverse"]]  
                segment = input_dict["origin_segment"]  
              
            # Compute MSE and MAE
            mse = F.mse_loss(heatmap_pred, segment)  
            mae = F.l1_loss(heatmap_pred, segment)  
            loss = output_dict["loss"]
 
            mse = mse.item()  
            mae = mae.item()  
            loss = loss.item()  
              
            # Store metrics  
            self.trainer.storage.put_scalar("val_mse", mse)  
            self.trainer.storage.put_scalar("val_mae", mae)  
            self.trainer.storage.put_scalar("val_loss", loss)  
              
            info = "Test: [{iter}/{max_iter}] ".format(  
                iter=i + 1, max_iter=len(self.trainer.val_loader)  
            )  
            if "origin_coord" in input_dict.keys():  
                info = "Interp. " + info  
            self.trainer.logger.info(  
                info + "Loss {loss:.6f} MSE {mse:.6f} MAE {mae:.6f}".format(  
                    loss=loss, mse=mse, mae=mae  
                )  
            )  
 
        # Compute averages  
        loss_avg = self.trainer.storage.history("val_loss").avg  
        mse_avg = self.trainer.storage.history("val_mse").avg  
        mae_avg = self.trainer.storage.history("val_mae").avg 

        all_predictions = torch.cat(all_predictions, dim=0)
          
        self.trainer.logger.info(  
            "Val result: Loss/MSE/MAE {:.6f}/{:.6f}/{:.6f}".format(  
                loss_avg, mse_avg, mae_avg  
            )  
        )  
          
        current_epoch = self.trainer.epoch + 1  
        if self.trainer.writer is not None:  
            self.trainer.writer.add_scalar("val/loss", loss_avg, current_epoch)  
            self.trainer.writer.add_scalar("val/MSE", mse_avg, current_epoch)  
            self.trainer.writer.add_scalar("val/MAE", mae_avg, current_epoch)  
            self.trainer.writer.add_histogram(
                "val/predictions_distribution",
                all_predictions,
                current_epoch
            )
            if self.trainer.cfg.enable_wandb:  
                wandb.log(  
                    {
                        "Epoch": current_epoch,  
                        "val/loss": loss_avg,  
                        "val/MSE": mse_avg,  
                        "val/MAE": mae_avg,  
                        "val/predictions_histogram": wandb.Histogram(all_predictions.numpy()),
                    },  
                    step=wandb.run.step,  
                )  
          
        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")  
        # Save MSE as the metric for checkpoint saving (lower is better)  
        self.trainer.comm_info["current_metric_value"] = -mse_avg  # Negative because CheckpointSaver maximizes  
        self.trainer.comm_info["current_metric_name"] = "MSE"  
      
    def after_train(self):  
        self.trainer.logger.info(  
            "Best MSE: {:.6f}".format(-self.trainer.best_metric_value)  # Negate back  
        )

@HOOKS.register_module()
class HeatmapEvaluatorV2(HookBase):
    def __init__(self):
        pass

    def before_train(self):
        if self.trainer.writer is not None and self.trainer.cfg.enable_wandb:
            wandb.define_metric("val/*", step_metric="Epoch")

    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            self.eval()

    @staticmethod
    def _masked_mean(x, mask, eps=1e-8):
        return (x * mask).sum() / mask.sum().clamp_min(eps)

    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()

        all_predictions = []
        loss_list = []
        mse_list = []
        mae_list = []

        for i, input_dict in enumerate(self.trainer.val_loader):
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)

            with torch.no_grad():
                output_dict = self.trainer.model(input_dict)

            # Prediction
            heatmap_pred = output_dict["seg_logits"]
            heatmap_pred = torch.clamp(heatmap_pred, 0.0, 1.0)
            all_predictions.append(heatmap_pred.detach().cpu())

            # Ground truth + validity mask
            segment = input_dict["segment"].to(heatmap_pred.device).float()
            validity_mask = input_dict["validity_mask"].to(heatmap_pred.device).float()

            # Handle inverse mapping if present
            if "inverse" in input_dict:
                assert "origin_segment" in input_dict
                heatmap_pred = heatmap_pred[input_dict["inverse"]]
                segment = input_dict["origin_segment"].to(heatmap_pred.device).float()

                if "origin_validity_mask" in input_dict:
                    validity_mask = input_dict["origin_validity_mask"].to(heatmap_pred.device).float()
                else:
                    validity_mask = validity_mask[input_dict["inverse"]]

            if heatmap_pred.shape != segment.shape:
                raise ValueError(
                    f"Prediction/GT shape mismatch: pred={heatmap_pred.shape}, gt={segment.shape}"
                )
            if validity_mask.shape != segment.shape:
                raise ValueError(
                    f"Mask/GT shape mismatch: mask={validity_mask.shape}, gt={segment.shape}"
                )

            # Masked metrics
            diff = heatmap_pred - segment
            mse = self._masked_mean(diff.pow(2), validity_mask)
            mae = self._masked_mean(diff.abs(), validity_mask)

            # Loss from forward (already masked if your model is updated)
            loss = output_dict["loss"]

            mse_item = mse.item()
            mae_item = mae.item()
            loss_item = loss.item()

            loss_list.append(loss_item)
            mse_list.append(mse_item)
            mae_list.append(mae_item)

            # Store metrics
            self.trainer.storage.put_scalar("val_loss", loss_item)
            self.trainer.storage.put_scalar("val_mse", mse_item)
            self.trainer.storage.put_scalar("val_mae", mae_item)

            info = "Val: [{iter}/{max_iter}] ".format(
                iter=i + 1, max_iter=len(self.trainer.val_loader)
            )
            if "origin_coord" in input_dict:
                info = "Interp. " + info

            self.trainer.logger.info(
                info + "Loss {loss:.6f} MSE {mse:.6f} MAE {mae:.6f}".format(
                    loss=loss_item, mse=mse_item, mae=mae_item
                )
            )

        # Per-epoch averages
        loss_avg = float(np.mean(loss_list)) if len(loss_list) > 0 else 0.0
        mse_avg = float(np.mean(mse_list)) if len(mse_list) > 0 else 0.0
        mae_avg = float(np.mean(mae_list)) if len(mae_list) > 0 else 0.0

        all_predictions = torch.cat(all_predictions, dim=0)

        self.trainer.logger.info(
            "Val result: Loss/MSE/MAE {:.6f}/{:.6f}/{:.6f}".format(
                loss_avg, mse_avg, mae_avg
            )
        )

        current_epoch = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("val/loss", loss_avg, current_epoch)
            self.trainer.writer.add_scalar("val/MSE", mse_avg, current_epoch)
            self.trainer.writer.add_scalar("val/MAE", mae_avg, current_epoch)
            self.trainer.writer.add_histogram(
                "val/predictions_distribution",
                all_predictions,
                current_epoch,
            )

            if self.trainer.cfg.enable_wandb:
                wandb.log(
                    {
                        "Epoch": current_epoch,
                        "val/loss": loss_avg,
                        "val/MSE": mse_avg,
                        "val/MAE": mae_avg,
                        "val/predictions_histogram": wandb.Histogram(all_predictions.numpy()),
                    },
                    step=wandb.run.step,
                )

        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

        # Lower MSE is better, but CheckpointSaver maximizes
        self.trainer.comm_info["current_metric_value"] = -mse_avg
        self.trainer.comm_info["current_metric_name"] = "MSE"

    def after_train(self):
        self.trainer.logger.info(
            "Best MSE: {:.6f}".format(-self.trainer.best_metric_value)
        )
