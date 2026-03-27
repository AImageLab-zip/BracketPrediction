"""
Tester

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import json
from uuid import uuid4
import os
import time
import numpy as np
from pathlib import Path
from collections import OrderedDict
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data
import re
from .defaults import create_ddp_model
import pointcept.utils.comm as comm
from pointcept.datasets import build_dataset, collate_fn
from pointcept.models import build_model
from pointcept.utils.logger import get_root_logger
from pointcept.utils.registry import Registry
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from pointcept.utils.misc import (
    AverageMeter,
    intersection_and_union,
    intersection_and_union_gpu,
    make_dirs,
)
# custom import
from pointcept.utils.bracket_mapper import BracketMapper
from sklearn.cluster import KMeans

try:
    import trimesh
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
except ImportError:
    trimesh = None
    plt = None

try:
    import pointops
except:
    pointops = None


TESTERS = Registry("testers")


class TesterBase:
    def __init__(self, cfg, model=None, test_loader=None, verbose=False) -> None:
        torch.multiprocessing.set_sharing_strategy("file_system")
        self.logger = get_root_logger(
            log_file=os.path.join(cfg.save_path, "test.log"),
            file_mode="a" if cfg.resume else "w",
        )
        self.logger.info("=> Loading config ...")
        self.cfg = cfg
        self.verbose = verbose
        if self.verbose and model is None:
            # if model is not none, trigger tester with trainer, no need to print config
            self.logger.info(f"Save path: {cfg.save_path}")
            self.logger.info(f"Config:\n{cfg.pretty_text}")
        if model is None:
            self.logger.info("=> Building model ...")
            self.model = self.build_model()
        else:
            self.model = model
        if test_loader is None:
            self.logger.info("=> Building test dataset & dataloader ...")
            self.test_loader = self.build_test_loader()
        else:
            self.test_loader = test_loader

    def build_model(self):
        model = build_model(self.cfg.model)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Num params: {n_parameters}")
        model = create_ddp_model(
            model.cuda(),
            broadcast_buffers=False,
            find_unused_parameters=self.cfg.find_unused_parameters,
        )
        if os.path.isfile(self.cfg.weight):
            self.logger.info(f"Loading weight at: {self.cfg.weight}")
            checkpoint = torch.load(self.cfg.weight, weights_only=False)
            weight = OrderedDict()
            for key, value in checkpoint["state_dict"].items():
                if key.startswith("module."):
                    if comm.get_world_size() == 1:
                        key = key[7:]  # module.xxx.xxx -> xxx.xxx
                else:
                    if comm.get_world_size() > 1:
                        key = "module." + key  # xxx.xxx -> module.xxx.xxx
                weight[key] = value
            model.load_state_dict(weight, strict=True)
            self.logger.info(
                "=> Loaded weight '{}' (epoch {})".format(
                    self.cfg.weight, checkpoint["epoch"]
                )
            )
        else:
            raise RuntimeError("=> No checkpoint found at '{}'".format(self.cfg.weight))
        return model

    def build_test_loader(self):
        # Validate test dataset config early to provide a clearer error
        # message when the user misconfigures the config file.
        test_cfg = None
        try:
            test_cfg = self.cfg.data.test
        except Exception:
            # cfg.data might not expose attributes like a dict; try mapping access
            try:
                test_cfg = self.cfg.data["test"]
            except Exception:
                test_cfg = None

        if test_cfg is None:
            # Try a reasonable fallback: use validation dataset if test is missing
            val_cfg = None
            try:
                val_cfg = self.cfg.data.val
            except Exception:
                try:
                    val_cfg = self.cfg.data["val"]
                except Exception:
                    val_cfg = None

            if val_cfg is not None:
                self.logger.warning(
                    "cfg.data.test is None, falling back to cfg.data.val for testing."
                )
                test_dataset = build_dataset(val_cfg)
            else:
                raise RuntimeError(
                    "cfg.data.test is None or missing — please set `data.test` in your config to a dataset dict."
                )
        else:
            test_dataset = build_dataset(test_cfg)
        if comm.get_world_size() > 1:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        else:
            test_sampler = None
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.cfg.batch_size_test_per_gpu,
            shuffle=False,
            num_workers=self.cfg.batch_size_test_per_gpu,
            pin_memory=True,
            sampler=test_sampler,
            collate_fn=self.__class__.collate_fn,
        )
        return test_loader

    def test(self):
        raise NotImplementedError

    @staticmethod
    def collate_fn(batch):
        raise collate_fn(batch)


@TESTERS.register_module()
class SemSegTester(TesterBase):
    def test(self):
        assert self.test_loader.batch_size == 1
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        self.model.eval()

        save_path = os.path.join(self.cfg.save_path, "result")
        make_dirs(save_path)
        # create submit folder only on main process
        if (
            self.cfg.data.test.type == "ScanNetDataset"
            or self.cfg.data.test.type == "ScanNet200Dataset"
            or self.cfg.data.test.type == "ScanNetPPDataset"
        ) and comm.is_main_process():
            make_dirs(os.path.join(save_path, "submit"))
        elif (
            self.cfg.data.test.type == "SemanticKITTIDataset" and comm.is_main_process()
        ):
            make_dirs(os.path.join(save_path, "submit"))
        elif self.cfg.data.test.type == "NuScenesDataset" and comm.is_main_process():
            import json

            make_dirs(os.path.join(save_path, "submit", "lidarseg", "test"))
            make_dirs(os.path.join(save_path, "submit", "test"))
            submission = dict(
                meta=dict(
                    use_camera=False,
                    use_lidar=True,
                    use_radar=False,
                    use_map=False,
                    use_external=False,
                )
            )
            with open(
                os.path.join(save_path, "submit", "test", "submission.json"), "w"
            ) as f:
                json.dump(submission, f, indent=4)
        comm.synchronize()
        record = {}
        # fragment inference
        for idx, data_dict in enumerate(self.test_loader): # for each sample
            start = time.time()
            data_dict = data_dict[0]  # current assume batch size is 1
            fragment_list = data_dict.pop("fragment_list")
            segment = data_dict.pop("segment")
            data_name = data_dict.pop("name")
            pred_save_path = os.path.join(save_path, "{}_pred.npy".format(data_name))
            if os.path.isfile(pred_save_path):
                logger.info(
                    "{}/{}: {}, loaded pred and label.".format(
                        idx + 1, len(self.test_loader), data_name
                    )
                )
                pred = np.load(pred_save_path)
                if "origin_segment" in data_dict.keys():
                    segment = data_dict["origin_segment"]
            else:
                pred = torch.zeros((segment.size, self.cfg.data.num_classes)).cuda()
                for i in range(len(fragment_list)): # for each fragment of that sample
                    fragment_batch_size = 1
                    s_i, e_i = i * fragment_batch_size, min(
                        (i + 1) * fragment_batch_size, len(fragment_list)
                    )
                    input_dict = collate_fn(fragment_list[s_i:e_i])
                    for key in input_dict.keys():
                        if isinstance(input_dict[key], torch.Tensor):
                            input_dict[key] = input_dict[key].cuda(non_blocking=True)
                    idx_part = input_dict["index"]
                    with torch.no_grad():
                        pred_part = self.model(input_dict)["seg_logits"]  # (n, k)
                        pred_part = F.softmax(pred_part, -1)
                        if self.cfg.empty_cache:
                            torch.cuda.empty_cache()
                        bs = 0
                        for be in input_dict["offset"]:
                            pred[idx_part[bs:be], :] += pred_part[bs:be]
                            bs = be

                    logger.info(
                        "Test: {}/{}-{data_name}, Batch: {batch_idx}/{batch_num}".format(
                            idx + 1,
                            len(self.test_loader),
                            data_name=data_name,
                            batch_idx=i,
                            batch_num=len(fragment_list),
                        )
                    )
                if self.cfg.data.test.type == "ScanNetPPDataset":
                    pred = pred.topk(3, dim=1)[1].data.cpu().numpy()
                else:
                    pred = pred.max(1)[1].data.cpu().numpy()
                if "origin_segment" in data_dict.keys():
                    assert "inverse" in data_dict.keys()
                    pred = pred[data_dict["inverse"]]
                    segment = data_dict["origin_segment"]
                np.save(pred_save_path, pred)
            if (
                self.cfg.data.test.type == "ScanNetDataset"
                or self.cfg.data.test.type == "ScanNet200Dataset"
            ):
                np.savetxt(
                    os.path.join(save_path, "submit", "{}.txt".format(data_name)),
                    self.test_loader.dataset.class2id[pred].reshape([-1, 1]),
                    fmt="%d",
                )
            elif self.cfg.data.test.type == "ScanNetPPDataset":
                np.savetxt(
                    os.path.join(save_path, "submit", "{}.txt".format(data_name)),
                    pred.astype(np.int32),
                    delimiter=",",
                    fmt="%d",
                )
                pred = pred[:, 0]  # for mIoU, TODO: support top3 mIoU
            elif self.cfg.data.test.type == "SemanticKITTIDataset":
                # 00_000000 -> 00, 000000
                sequence_name, frame_name = data_name.split("_")
                os.makedirs(
                    os.path.join(
                        save_path, "submit", "sequences", sequence_name, "predictions"
                    ),
                    exist_ok=True,
                )
                submit = pred.astype(np.uint32)
                submit = np.vectorize(
                    self.test_loader.dataset.learning_map_inv.__getitem__
                )(submit).astype(np.uint32)
                submit.tofile(
                    os.path.join(
                        save_path,
                        "submit",
                        "sequences",
                        sequence_name,
                        "predictions",
                        f"{frame_name}.label",
                    )
                )
            elif self.cfg.data.test.type == "NuScenesDataset":
                np.array(pred + 1).astype(np.uint8).tofile(
                    os.path.join(
                        save_path,
                        "submit",
                        "lidarseg",
                        "test",
                        "{}_lidarseg.bin".format(data_name),
                    )
                )

            intersection, union, target = intersection_and_union(
                pred, segment, self.cfg.data.num_classes, self.cfg.data.ignore_index
            )
            intersection_meter.update(intersection)
            union_meter.update(union)
            target_meter.update(target)
            record[data_name] = dict(
                intersection=intersection, union=union, target=target
            )

            mask = union != 0
            iou_class = intersection / (union + 1e-10)
            iou = np.mean(iou_class[mask])
            acc = sum(intersection) / (sum(target) + 1e-10)

            m_iou = np.mean(intersection_meter.sum / (union_meter.sum + 1e-10))
            m_acc = np.mean(intersection_meter.sum / (target_meter.sum + 1e-10))

            batch_time.update(time.time() - start)
            logger.info(
                "Test: {} [{}/{}]-{} "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "Accuracy {acc:.4f} ({m_acc:.4f}) "
                "mIoU {iou:.4f} ({m_iou:.4f})".format(
                    data_name,
                    idx + 1,
                    len(self.test_loader),
                    segment.size,
                    batch_time=batch_time,
                    acc=acc,
                    m_acc=m_acc,
                    iou=iou,
                    m_iou=m_iou,
                )
            )

        logger.info("Syncing ...")
        comm.synchronize()
        record_sync = comm.gather(record, dst=0)

        if comm.is_main_process():
            record = {}
            for _ in range(len(record_sync)):
                r = record_sync.pop()
                record.update(r)
                del r
            intersection = np.sum(
                [meters["intersection"] for _, meters in record.items()], axis=0
            )
            union = np.sum([meters["union"] for _, meters in record.items()], axis=0)
            target = np.sum([meters["target"] for _, meters in record.items()], axis=0)

            if self.cfg.data.test.type == "S3DISDataset":
                torch.save(
                    dict(intersection=intersection, union=union, target=target),
                    os.path.join(save_path, f"{self.test_loader.dataset.split}.pth"),
                )

            iou_class = intersection / (union + 1e-10)
            accuracy_class = intersection / (target + 1e-10)
            mIoU = np.mean(iou_class)
            mAcc = np.mean(accuracy_class)
            allAcc = sum(intersection) / (sum(target) + 1e-10)

            logger.info(
                "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}".format(
                    mIoU, mAcc, allAcc
                )
            )
            for i in range(self.cfg.data.num_classes):
                logger.info(
                    "Class_{idx} - {name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                        idx=i,
                        name=self.cfg.data.names[i],
                        iou=iou_class[i],
                        accuracy=accuracy_class[i],
                    )
                )
            logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return batch


@TESTERS.register_module()
class DINOSemSegTester(TesterBase):
    def test(self):
        assert self.test_loader.batch_size == 1
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        self.model.eval()

        save_path = os.path.join(self.cfg.save_path, "result")
        make_dirs(save_path)
        # create submit folder only on main process
        if (
            self.cfg.data.test.type == "ScanNetDataset"
            or self.cfg.data.test.type == "ScanNet200Dataset"
            or self.cfg.data.test.type == "ScanNetPPDataset"
        ) and comm.is_main_process():
            make_dirs(os.path.join(save_path, "submit"))
        elif (
            self.cfg.data.test.type == "SemanticKITTIDataset" and comm.is_main_process()
        ):
            make_dirs(os.path.join(save_path, "submit"))
        elif self.cfg.data.test.type == "NuScenesDataset" and comm.is_main_process():
            import json

            make_dirs(os.path.join(save_path, "submit", "lidarseg", "test"))
            make_dirs(os.path.join(save_path, "submit", "test"))
            submission = dict(
                meta=dict(
                    use_camera=False,
                    use_lidar=True,
                    use_radar=False,
                    use_map=False,
                    use_external=False,
                )
            )
            with open(
                os.path.join(save_path, "submit", "test", "submission.json"), "w"
            ) as f:
                json.dump(submission, f, indent=4)
        comm.synchronize()
        record = {}
        # fragment inference
        for idx, data_dict in enumerate(self.test_loader):
            end = time.time()
            data_dict = data_dict[0]  # current assume batch size is 1
            fragment_list = data_dict.pop("fragment_list")
            segment = data_dict.pop("segment")
            data_name = data_dict.pop("name")
            dino_coord = data_dict.pop("dino_coord").cuda(non_blocking=True)
            dino_feat = data_dict.pop("dino_feat").cuda(non_blocking=True)
            dino_offset = data_dict.pop("dino_offset").cuda(non_blocking=True)
            pred_save_path = os.path.join(save_path, "{}_pred.npy".format(data_name))
            if os.path.isfile(pred_save_path):
                logger.info(
                    "{}/{}: {}, loaded pred and label.".format(
                        idx + 1, len(self.test_loader), data_name
                    )
                )
                pred = np.load(pred_save_path)
                if "origin_segment" in data_dict.keys():
                    segment = data_dict["origin_segment"]
            else:
                pred = torch.zeros((segment.size, self.cfg.data.num_classes)).cuda()
                for i in range(len(fragment_list)):
                    fragment_batch_size = 1
                    s_i, e_i = i * fragment_batch_size, min(
                        (i + 1) * fragment_batch_size, len(fragment_list)
                    )
                    input_dict = collate_fn(fragment_list[s_i:e_i])
                    for key in input_dict.keys():
                        if isinstance(input_dict[key], torch.Tensor):
                            input_dict[key] = input_dict[key].cuda(non_blocking=True)
                    input_dict["dino_coord"] = dino_coord
                    input_dict["dino_feat"] = dino_feat
                    input_dict["dino_offset"] = dino_offset
                    idx_part = input_dict["index"]
                    with torch.no_grad():
                        pred_part = self.model(input_dict)["seg_logits"]  # (n, k)
                        pred_part = F.softmax(pred_part, -1)
                        if self.cfg.empty_cache:
                            torch.cuda.empty_cache()
                        bs = 0
                        for be in input_dict["offset"]:
                            pred[idx_part[bs:be], :] += pred_part[bs:be]
                            bs = be

                    logger.info(
                        "Test: {}/{}-{data_name}, Batch: {batch_idx}/{batch_num}".format(
                            idx + 1,
                            len(self.test_loader),
                            data_name=data_name,
                            batch_idx=i,
                            batch_num=len(fragment_list),
                        )
                    )
                if self.cfg.data.test.type == "ScanNetPPDataset":
                    pred = pred.topk(3, dim=1)[1].data.cpu().numpy()
                else:
                    pred = pred.max(1)[1].data.cpu().numpy()
                if "origin_segment" in data_dict.keys():
                    assert "inverse" in data_dict.keys()
                    pred = pred[data_dict["inverse"]]
                    segment = data_dict["origin_segment"]
                np.save(pred_save_path, pred)
            if (
                self.cfg.data.test.type == "ScanNetDataset"
                or self.cfg.data.test.type == "ScanNet200Dataset"
            ):
                np.savetxt(
                    os.path.join(save_path, "submit", "{}.txt".format(data_name)),
                    self.test_loader.dataset.class2id[pred].reshape([-1, 1]),
                    fmt="%d",
                )
            elif self.cfg.data.test.type == "ScanNetPPDataset":
                np.savetxt(
                    os.path.join(save_path, "submit", "{}.txt".format(data_name)),
                    pred.astype(np.int32),
                    delimiter=",",
                    fmt="%d",
                )
                pred = pred[:, 0]  # for mIoU, TODO: support top3 mIoU
            elif self.cfg.data.test.type == "SemanticKITTIDataset":
                # 00_000000 -> 00, 000000
                sequence_name, frame_name = data_name.split("_")
                os.makedirs(
                    os.path.join(
                        save_path, "submit", "sequences", sequence_name, "predictions"
                    ),
                    exist_ok=True,
                )
                submit = pred.astype(np.uint32)
                submit = np.vectorize(
                    self.test_loader.dataset.learning_map_inv.__getitem__
                )(submit).astype(np.uint32)
                submit.tofile(
                    os.path.join(
                        save_path,
                        "submit",
                        "sequences",
                        sequence_name,
                        "predictions",
                        f"{frame_name}.label",
                    )
                )
            elif self.cfg.data.test.type == "NuScenesDataset":
                np.array(pred + 1).astype(np.uint8).tofile(
                    os.path.join(
                        save_path,
                        "submit",
                        "lidarseg",
                        "test",
                        "{}_lidarseg.bin".format(data_name),
                    )
                )

            intersection, union, target = intersection_and_union(
                pred, segment, self.cfg.data.num_classes, self.cfg.data.ignore_index
            )
            intersection_meter.update(intersection)
            union_meter.update(union)
            target_meter.update(target)
            record[data_name] = dict(
                intersection=intersection, union=union, target=target
            )

            mask = union != 0
            iou_class = intersection / (union + 1e-10)
            iou = np.mean(iou_class[mask])
            acc = sum(intersection) / (sum(target) + 1e-10)

            m_iou = np.mean(intersection_meter.sum / (union_meter.sum + 1e-10))
            m_acc = np.mean(intersection_meter.sum / (target_meter.sum + 1e-10))

            batch_time.update(time.time() - end)
            logger.info(
                "Test: {} [{}/{}]-{} "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "Accuracy {acc:.4f} ({m_acc:.4f}) "
                "mIoU {iou:.4f} ({m_iou:.4f})".format(
                    data_name,
                    idx + 1,
                    len(self.test_loader),
                    segment.size,
                    batch_time=batch_time,
                    acc=acc,
                    m_acc=m_acc,
                    iou=iou,
                    m_iou=m_iou,
                )
            )

        logger.info("Syncing ...")
        comm.synchronize()
        record_sync = comm.gather(record, dst=0)

        if comm.is_main_process():
            record = {}
            for _ in range(len(record_sync)):
                r = record_sync.pop()
                record.update(r)
                del r
            intersection = np.sum(
                [meters["intersection"] for _, meters in record.items()], axis=0
            )
            union = np.sum([meters["union"] for _, meters in record.items()], axis=0)
            target = np.sum([meters["target"] for _, meters in record.items()], axis=0)

            if self.cfg.data.test.type == "S3DISDataset":
                torch.save(
                    dict(intersection=intersection, union=union, target=target),
                    os.path.join(save_path, f"{self.test_loader.dataset.split}.pth"),
                )

            iou_class = intersection / (union + 1e-10)
            accuracy_class = intersection / (target + 1e-10)
            mIoU = np.mean(iou_class)
            mAcc = np.mean(accuracy_class)
            allAcc = sum(intersection) / (sum(target) + 1e-10)

            logger.info(
                "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}".format(
                    mIoU, mAcc, allAcc
                )
            )
            for i in range(self.cfg.data.num_classes):
                logger.info(
                    "Class_{idx} - {name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                        idx=i,
                        name=self.cfg.data.names[i],
                        iou=iou_class[i],
                        accuracy=accuracy_class[i],
                    )
                )
            logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return batch


@TESTERS.register_module()
class ClsTester(TesterBase):
    def test(self):
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        self.model.eval()

        for i, input_dict in enumerate(self.test_loader):
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            end = time.time()
            with torch.no_grad():
                output_dict = self.model(input_dict)
            output = output_dict["cls_logits"]
            pred = output.max(1)[1]
            label = input_dict["category"]
            intersection, union, target = intersection_and_union_gpu(
                pred, label, self.cfg.data.num_classes, self.cfg.data.ignore_index
            )
            if comm.get_world_size() > 1:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(
                    target
                )
            intersection, union, target = (
                intersection.cpu().numpy(),
                union.cpu().numpy(),
                target.cpu().numpy(),
            )
            intersection_meter.update(intersection), union_meter.update(
                union
            ), target_meter.update(target)

            accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
            batch_time.update(time.time() - end)

            logger.info(
                "Test: [{}/{}] "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "Accuracy {accuracy:.4f} ".format(
                    i + 1,
                    len(self.test_loader),
                    batch_time=batch_time,
                    accuracy=accuracy,
                )
            )

        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
        mIoU = np.mean(iou_class)
        mAcc = np.mean(accuracy_class)
        allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
        logger.info(
            "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(
                mIoU, mAcc, allAcc
            )
        )

        for i in range(self.cfg.data.num_classes):
            logger.info(
                "Class_{idx} - {name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                    idx=i,
                    name=self.cfg.data.names[i],
                    iou=iou_class[i],
                    accuracy=accuracy_class[i],
                )
            )
        logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return collate_fn(batch)


@TESTERS.register_module()
class ClsVotingTester(TesterBase):
    def __init__(
        self,
        num_repeat=100,
        metric="allAcc",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_repeat = num_repeat
        self.metric = metric
        self.best_idx = 0
        self.best_record = None
        self.best_metric = 0

    def test(self):
        for i in range(self.num_repeat):
            logger = get_root_logger()
            logger.info(f">>>>>>>>>>>>>>>> Start Evaluation {i + 1} >>>>>>>>>>>>>>>>")
            record = self.test_once()
            if comm.is_main_process():
                if record[self.metric] > self.best_metric:
                    self.best_record = record
                    self.best_idx = i
                    self.best_metric = record[self.metric]
                info = f"Current best record is Evaluation {i + 1}: "
                for m in self.best_record.keys():
                    info += f"{m}: {self.best_record[m]:.4f} "
                logger.info(info)

    def test_once(self):
        logger = get_root_logger()
        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        target_meter = AverageMeter()
        record = {}
        self.model.eval()

        for idx, data_dict in enumerate(self.test_loader):
            end = time.time()
            data_dict = data_dict[0]  # current assume batch size is 1
            voting_list = data_dict.pop("voting_list")
            category = data_dict.pop("category")
            data_name = data_dict.pop("name")
            # pred = torch.zeros([1, self.cfg.data.num_classes]).cuda()
            # for i in range(len(voting_list)):
            #     input_dict = voting_list[i]
            #     for key in input_dict.keys():
            #         if isinstance(input_dict[key], torch.Tensor):
            #             input_dict[key] = input_dict[key].cuda(non_blocking=True)
            #     with torch.no_grad():
            #         pred += F.softmax(self.model(input_dict)["cls_logits"], -1)
            input_dict = collate_fn(voting_list)
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                pred = F.softmax(self.model(input_dict)["cls_logits"], -1).sum(
                    0, keepdim=True
                )
            pred = pred.max(1)[1].cpu().numpy()
            intersection, union, target = intersection_and_union(
                pred, category, self.cfg.data.num_classes, self.cfg.data.ignore_index
            )
            intersection_meter.update(intersection)
            target_meter.update(target)
            record[data_name] = dict(intersection=intersection, target=target)
            acc = sum(intersection) / (sum(target) + 1e-10)
            m_acc = np.mean(intersection_meter.sum / (target_meter.sum + 1e-10))
            batch_time.update(time.time() - end)
            logger.info(
                "Test: {} [{}/{}] "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "Accuracy {acc:.4f} ({m_acc:.4f}) ".format(
                    data_name,
                    idx + 1,
                    len(self.test_loader),
                    batch_time=batch_time,
                    acc=acc,
                    m_acc=m_acc,
                )
            )

        logger.info("Syncing ...")
        comm.synchronize()
        record_sync = comm.gather(record, dst=0)

        if comm.is_main_process():
            record = {}
            for _ in range(len(record_sync)):
                r = record_sync.pop()
                record.update(r)
                del r
            intersection = np.sum(
                [meters["intersection"] for _, meters in record.items()], axis=0
            )
            target = np.sum([meters["target"] for _, meters in record.items()], axis=0)
            accuracy_class = intersection / (target + 1e-10)
            mAcc = np.mean(accuracy_class)
            allAcc = sum(intersection) / (sum(target) + 1e-10)

            logger.info("Val result: mAcc/allAcc {:.4f}/{:.4f}".format(mAcc, allAcc))
            for i in range(self.cfg.data.num_classes):
                logger.info(
                    "Class_{idx} - {name} Result: iou/accuracy {accuracy:.4f}".format(
                        idx=i,
                        name=self.cfg.data.names[i],
                        accuracy=accuracy_class[i],
                    )
                )
            return dict(mAcc=mAcc, allAcc=allAcc)

    @staticmethod
    def collate_fn(batch):
        return batch


@TESTERS.register_module()
class PartSegTester(TesterBase):
    def test(self):
        test_dataset = self.test_loader.dataset
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

        batch_time = AverageMeter()

        num_categories = len(self.test_loader.dataset.categories)
        iou_category, iou_count = np.zeros(num_categories), np.zeros(num_categories)
        self.model.eval()

        save_path = os.path.join(
            self.cfg.save_path, "result", "test_epoch{}".format(self.cfg.test_epoch)
        )
        make_dirs(save_path)

        for idx in range(len(test_dataset)):
            end = time.time()
            data_name = test_dataset.get_data_name(idx)

            data_dict_list, label = test_dataset[idx]
            pred = torch.zeros((label.size, self.cfg.data.num_classes)).cuda()
            batch_num = int(np.ceil(len(data_dict_list) / self.cfg.batch_size_test))
            for i in range(batch_num):
                s_i, e_i = i * self.cfg.batch_size_test, min(
                    (i + 1) * self.cfg.batch_size_test, len(data_dict_list)
                )
                input_dict = collate_fn(data_dict_list[s_i:e_i])
                for key in input_dict.keys():
                    if isinstance(input_dict[key], torch.Tensor):
                        input_dict[key] = input_dict[key].cuda(non_blocking=True)
                with torch.no_grad():
                    pred_part = self.model(input_dict)["cls_logits"]
                    pred_part = F.softmax(pred_part, -1)
                if self.cfg.empty_cache:
                    torch.cuda.empty_cache()
                pred_part = pred_part.reshape(-1, label.size, self.cfg.data.num_classes)
                pred = pred + pred_part.total(dim=0)
                logger.info(
                    "Test: {} {}/{}, Batch: {batch_idx}/{batch_num}".format(
                        data_name,
                        idx + 1,
                        len(test_dataset),
                        batch_idx=i,
                        batch_num=batch_num,
                    )
                )
            pred = pred.max(1)[1].data.cpu().numpy()

            category_index = data_dict_list[0]["cls_token"]
            category = self.test_loader.dataset.categories[category_index]
            parts_idx = self.test_loader.dataset.category2part[category]
            parts_iou = np.zeros(len(parts_idx))
            for j, part in enumerate(parts_idx):
                if (np.sum(label == part) == 0) and (np.sum(pred == part) == 0):
                    parts_iou[j] = 1.0
                else:
                    i = (label == part) & (pred == part)
                    u = (label == part) | (pred == part)
                    parts_iou[j] = np.sum(i) / (np.sum(u) + 1e-10)
            iou_category[category_index] += parts_iou.mean()
            iou_count[category_index] += 1

            batch_time.update(time.time() - end)
            logger.info(
                "Test: {} [{}/{}] "
                "Batch {batch_time.val:.3f} "
                "({batch_time.avg:.3f}) ".format(
                    data_name, idx + 1, len(self.test_loader), batch_time=batch_time
                )
            )

        ins_mIoU = iou_category.sum() / (iou_count.sum() + 1e-10)
        cat_mIoU = (iou_category / (iou_count + 1e-10)).mean()
        logger.info(
            "Val result: ins.mIoU/cat.mIoU {:.4f}/{:.4f}.".format(ins_mIoU, cat_mIoU)
        )
        for i in range(num_categories):
            logger.info(
                "Class_{idx}-{name} Result: iou_cat/num_sample {iou_cat:.4f}/{iou_count:.4f}".format(
                    idx=i,
                    name=self.test_loader.dataset.categories[i],
                    iou_cat=iou_category[i] / (iou_count[i] + 1e-10),
                    iou_count=int(iou_count[i]),
                )
            )
        logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return collate_fn(batch)


@TESTERS.register_module()
class InsSegTester(TesterBase):
    def __init__(
        self,
        segment_ignore_index,
        instance_ignore_index,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.segment_ignore_index = segment_ignore_index
        self.instance_ignore_index = instance_ignore_index
        self.valid_class_names = [
            self.cfg.data.names[i]
            for i in range(self.cfg.data.num_classes)
            if i not in self.segment_ignore_index
        ]
        self.overlaps = np.append(np.arange(0.5, 0.95, 0.05), 0.25)
        self.min_region_sizes = 100
        self.distance_threshes = float("inf")
        self.distance_confs = -float("inf")

    def test(self):
        assert self.test_loader.batch_size == 1
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

        batch_time = AverageMeter()

        self.model.eval()
        scenes = []

        for idx, data_dict in enumerate(self.test_loader):
            start = time.time()
            data_name = data_dict.pop("name")
            for key in data_dict.keys():
                if isinstance(data_dict[key], torch.Tensor):
                    data_dict[key] = data_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                output_dict = self.model(data_dict)
                segment = data_dict["origin_segment"]
                instance = data_dict["origin_instance"]

                if "origin_coord" in data_dict.keys():
                    reverse, _ = pointops.knn_query(
                        1,
                        data_dict["coord"].float(),
                        data_dict["offset"].int(),
                        data_dict["origin_coord"].float(),
                        data_dict["origin_offset"].int(),
                    )
                    reverse = reverse.cpu().flatten().long()
                    output_dict["pred_masks"] = output_dict["pred_masks"][:, reverse]
                    segment = data_dict["origin_segment"]
                    instance = data_dict["origin_instance"]

                gt_instances, pred_instance = self.associate_instances(
                    output_dict, segment, instance
                )

            scenes.append(dict(gt=gt_instances, pred=pred_instance))
            batch_time.update(time.time() - start)
            logger.info(
                "Test: {} [{}/{}] "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) ".format(
                    data_name,
                    idx + 1,
                    len(self.test_loader),
                    batch_time=batch_time,
                )
            )
            if self.cfg.data.test.type == "ScanNetPPDataset":
                self.write_scannetpp_results(
                    output_dict["pred_scores"],
                    output_dict["pred_masks"],
                    output_dict["pred_classes"],
                    data_name,
                )

        comm.synchronize()
        scenes_sync = comm.gather(scenes, dst=0)
        scenes = [scene for scenes_ in scenes_sync for scene in scenes_]
        ap_scores = self.evaluate_matches(scenes)
        all_ap = ap_scores["all_ap"]
        all_ap_50 = ap_scores["all_ap_50%"]
        all_ap_25 = ap_scores["all_ap_25%"]
        logger.info(
            "Val result: mAP/AP50/AP25 {:.4f}/{:.4f}/{:.4f}.".format(
                all_ap, all_ap_50, all_ap_25
            )
        )
        for i, label_name in enumerate(self.valid_class_names):
            ap = ap_scores["classes"][label_name]["ap"]
            ap_50 = ap_scores["classes"][label_name]["ap50%"]
            ap_25 = ap_scores["classes"][label_name]["ap25%"]
            logger.info(
                "Class_{idx}-{name} Result: AP/AP50/AP25 {AP:.4f}/{AP50:.4f}/{AP25:.4f}".format(
                    idx=i, name=label_name, AP=ap, AP50=ap_50, AP25=ap_25
                )
            )
        logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    def write_scannetpp_results(
        self,
        pred_scores,
        pred_masks,
        pred_classes,
        data_name,
    ):
        pred_scores[pred_scores < 0] = 0
        pred_scores[pred_scores >= 0] = 1

        save_dir = os.path.join(self.cfg.save_path, "result", "submit")
        mask_dir = os.path.join(save_dir, "predicted_masks")
        make_dirs(mask_dir)

        result_path = os.path.join(save_dir, f"{data_name}.txt")
        result_file = open(result_path, "w")
        for i, (score, mask, cls) in enumerate(
            zip(
                pred_scores.cpu().numpy(),
                pred_masks.cpu().numpy(),
                pred_classes.cpu().numpy(),
            )
        ):
            mask = mask.astype(np.uint8)
            length = mask.shape[0]
            mask = np.concatenate([[0], mask, [0]])
            runs = np.where(mask[1:] != mask[:-1])[0] + 1
            runs[1::2] -= runs[::2]
            counts = " ".join(str(x) for x in runs)
            rle = dict(length=length, counts=counts)

            mask_path = os.path.join(mask_dir, f"{data_name}_{i:03d}.json")
            relative_path = os.path.join("predicted_masks", f"{data_name}_{i:03d}.json")
            with open(mask_path, "w") as mask_file:
                json.dump(rle, mask_file, indent=2)
            result_file.write(f"{relative_path} {cls} {score:.3f}\n")
        result_file.close()

    def associate_instances(self, pred, segment, instance):
        segment = segment.cpu().numpy()
        instance = instance.cpu().numpy()
        void_mask = np.in1d(segment, self.segment_ignore_index)

        assert (
            pred["pred_classes"].shape[0]
            == pred["pred_scores"].shape[0]
            == pred["pred_masks"].shape[0]
        )
        assert pred["pred_masks"].shape[1] == segment.shape[0] == instance.shape[0]
        # get gt instances
        gt_instances = dict()
        for i in range(self.cfg.data.num_classes):
            if i not in self.segment_ignore_index:
                gt_instances[self.cfg.data.names[i]] = []
        instance_ids, idx, counts = np.unique(
            instance, return_index=True, return_counts=True
        )
        segment_ids = segment[idx]
        for i in range(len(instance_ids)):
            if instance_ids[i] == self.instance_ignore_index:
                continue
            if segment_ids[i] in self.segment_ignore_index:
                continue
            gt_inst = dict()
            gt_inst["instance_id"] = instance_ids[i]
            gt_inst["segment_id"] = segment_ids[i]
            gt_inst["dist_conf"] = 0.0
            gt_inst["med_dist"] = -1.0
            gt_inst["vert_count"] = counts[i]
            gt_inst["matched_pred"] = []
            gt_instances[self.cfg.data.names[segment_ids[i]]].append(gt_inst)

        # get pred instances and associate with gt
        pred_instances = dict()
        for i in range(self.cfg.data.num_classes):
            if i not in self.segment_ignore_index:
                pred_instances[self.cfg.data.names[i]] = []
        instance_id = 0
        for i in range(len(pred["pred_classes"])):
            if pred["pred_classes"][i] in self.segment_ignore_index:
                continue
            pred_inst = dict()
            pred_inst["uuid"] = uuid4()
            pred_inst["instance_id"] = instance_id
            pred_inst["segment_id"] = pred["pred_classes"][i]
            pred_inst["confidence"] = pred["pred_scores"][i]
            pred_inst["mask"] = np.not_equal(pred["pred_masks"][i], 0)
            pred_inst["vert_count"] = np.count_nonzero(pred_inst["mask"])
            pred_inst["void_intersection"] = np.count_nonzero(
                np.logical_and(void_mask, pred_inst["mask"])
            )
            if pred_inst["vert_count"] < self.min_region_sizes:
                continue  # skip if empty
            segment_name = self.cfg.data.names[pred_inst["segment_id"]]
            matched_gt = []
            for gt_idx, gt_inst in enumerate(gt_instances[segment_name]):
                intersection = np.count_nonzero(
                    np.logical_and(
                        instance == gt_inst["instance_id"], pred_inst["mask"]
                    )
                )
                if intersection > 0:
                    gt_inst_ = gt_inst.copy()
                    pred_inst_ = pred_inst.copy()
                    gt_inst_["intersection"] = intersection
                    pred_inst_["intersection"] = intersection
                    matched_gt.append(gt_inst_)
                    gt_inst["matched_pred"].append(pred_inst_)
            pred_inst["matched_gt"] = matched_gt
            pred_instances[segment_name].append(pred_inst)
            instance_id += 1
        return gt_instances, pred_instances

    def evaluate_matches(self, scenes):
        overlaps = self.overlaps
        min_region_sizes = [self.min_region_sizes]
        dist_threshes = [self.distance_threshes]
        dist_confs = [self.distance_confs]

        # results: class x overlap
        ap_table = np.zeros(
            (len(dist_threshes), len(self.valid_class_names), len(overlaps)), float
        )
        for di, (min_region_size, distance_thresh, distance_conf) in enumerate(
            zip(min_region_sizes, dist_threshes, dist_confs)
        ):
            for oi, overlap_th in enumerate(overlaps):
                pred_visited = {}
                for scene in scenes:
                    for _ in scene["pred"]:
                        for label_name in self.valid_class_names:
                            for p in scene["pred"][label_name]:
                                if "uuid" in p:
                                    pred_visited[p["uuid"]] = False
                for li, label_name in enumerate(self.valid_class_names):
                    y_true = np.empty(0)
                    y_score = np.empty(0)
                    hard_false_negatives = 0
                    has_gt = False
                    has_pred = False
                    for scene in scenes:
                        pred_instances = scene["pred"][label_name]
                        gt_instances = scene["gt"][label_name]
                        # filter groups in ground truth
                        gt_instances = [
                            gt
                            for gt in gt_instances
                            if gt["vert_count"] >= min_region_size
                            and gt["med_dist"] <= distance_thresh
                            and gt["dist_conf"] >= distance_conf
                        ]
                        if gt_instances:
                            has_gt = True
                        if pred_instances:
                            has_pred = True

                        cur_true = np.ones(len(gt_instances))
                        cur_score = np.ones(len(gt_instances)) * (-float("inf"))
                        cur_match = np.zeros(len(gt_instances), dtype=bool)
                        # collect matches
                        for gti, gt in enumerate(gt_instances):
                            found_match = False
                            for pred in gt["matched_pred"]:
                                # greedy assignments
                                if pred_visited[pred["uuid"]]:
                                    continue
                                overlap = float(pred["intersection"]) / (
                                    gt["vert_count"]
                                    + pred["vert_count"]
                                    - pred["intersection"]
                                )
                                if overlap > overlap_th:
                                    confidence = pred["confidence"]
                                    # if already have a prediction for this gt,
                                    # the prediction with the lower score is automatically a false positive
                                    if cur_match[gti]:
                                        max_score = max(cur_score[gti], confidence)
                                        min_score = min(cur_score[gti], confidence)
                                        cur_score[gti] = max_score
                                        # append false positive
                                        cur_true = np.append(cur_true, 0)
                                        cur_score = np.append(cur_score, min_score)
                                        cur_match = np.append(cur_match, True)
                                    # otherwise set score
                                    else:
                                        found_match = True
                                        cur_match[gti] = True
                                        cur_score[gti] = confidence
                                        pred_visited[pred["uuid"]] = True
                            if not found_match:
                                hard_false_negatives += 1
                        # remove non-matched ground truth instances
                        cur_true = cur_true[cur_match]
                        cur_score = cur_score[cur_match]

                        # collect non-matched predictions as false positive
                        for pred in pred_instances:
                            found_gt = False
                            for gt in pred["matched_gt"]:
                                overlap = float(gt["intersection"]) / (
                                    gt["vert_count"]
                                    + pred["vert_count"]
                                    - gt["intersection"]
                                )
                                if overlap > overlap_th:
                                    found_gt = True
                                    break
                            if not found_gt:
                                num_ignore = pred["void_intersection"]
                                for gt in pred["matched_gt"]:
                                    if gt["segment_id"] in self.segment_ignore_index:
                                        num_ignore += gt["intersection"]
                                    # small ground truth instances
                                    if (
                                        gt["vert_count"] < min_region_size
                                        or gt["med_dist"] > distance_thresh
                                        or gt["dist_conf"] < distance_conf
                                    ):
                                        num_ignore += gt["intersection"]
                                proportion_ignore = (
                                    float(num_ignore) / pred["vert_count"]
                                )
                                # if not ignored append false positive
                                if proportion_ignore <= overlap_th:
                                    cur_true = np.append(cur_true, 0)
                                    confidence = pred["confidence"]
                                    cur_score = np.append(cur_score, confidence)

                        # append to overall results
                        y_true = np.append(y_true, cur_true)
                        y_score = np.append(y_score, cur_score)

                    # compute average precision
                    if has_gt and has_pred:
                        # compute precision recall curve first

                        # sorting and cumsum
                        score_arg_sort = np.argsort(y_score)
                        y_score_sorted = y_score[score_arg_sort]
                        y_true_sorted = y_true[score_arg_sort]
                        y_true_sorted_cumsum = np.cumsum(y_true_sorted)

                        # unique thresholds
                        (thresholds, unique_indices) = np.unique(
                            y_score_sorted, return_index=True
                        )
                        num_prec_recall = len(unique_indices) + 1

                        # prepare precision recall
                        num_examples = len(y_score_sorted)
                        # https://github.com/ScanNet/ScanNet/pull/26
                        # all predictions are non-matched but also all of them are ignored and not counted as FP
                        # y_true_sorted_cumsum is empty
                        # num_true_examples = y_true_sorted_cumsum[-1]
                        num_true_examples = (
                            y_true_sorted_cumsum[-1]
                            if len(y_true_sorted_cumsum) > 0
                            else 0
                        )
                        precision = np.zeros(num_prec_recall)
                        recall = np.zeros(num_prec_recall)

                        # deal with the first point
                        y_true_sorted_cumsum = np.append(y_true_sorted_cumsum, 0)
                        # deal with remaining
                        for idx_res, idx_scores in enumerate(unique_indices):
                            cumsum = y_true_sorted_cumsum[idx_scores - 1]
                            tp = num_true_examples - cumsum
                            fp = num_examples - idx_scores - tp
                            fn = cumsum + hard_false_negatives
                            p = float(tp) / (tp + fp)
                            r = float(tp) / (tp + fn)
                            precision[idx_res] = p
                            recall[idx_res] = r

                        # first point in curve is artificial
                        precision[-1] = 1.0
                        recall[-1] = 0.0

                        # compute average of precision-recall curve
                        recall_for_conv = np.copy(recall)
                        recall_for_conv = np.append(recall_for_conv[0], recall_for_conv)
                        recall_for_conv = np.append(recall_for_conv, 0.0)

                        stepWidths = np.convolve(
                            recall_for_conv, [-0.5, 0, 0.5], "valid"
                        )
                        # integrate is now simply a dot product
                        ap_current = np.dot(precision, stepWidths)

                    elif has_gt:
                        ap_current = 0.0
                    else:
                        ap_current = float("nan")
                    ap_table[di, li, oi] = ap_current
        d_inf = 0
        o50 = np.where(np.isclose(self.overlaps, 0.5))
        o25 = np.where(np.isclose(self.overlaps, 0.25))
        oAllBut25 = np.where(np.logical_not(np.isclose(self.overlaps, 0.25)))
        ap_scores = dict()
        ap_scores["all_ap"] = np.nanmean(ap_table[d_inf, :, oAllBut25])
        ap_scores["all_ap_50%"] = np.nanmean(ap_table[d_inf, :, o50])
        ap_scores["all_ap_25%"] = np.nanmean(ap_table[d_inf, :, o25])
        ap_scores["classes"] = {}
        for li, label_name in enumerate(self.valid_class_names):
            ap_scores["classes"][label_name] = {}
            ap_scores["classes"][label_name]["ap"] = np.average(
                ap_table[d_inf, li, oAllBut25]
            )
            ap_scores["classes"][label_name]["ap50%"] = np.average(
                ap_table[d_inf, li, o50]
            )
            ap_scores["classes"][label_name]["ap25%"] = np.average(
                ap_table[d_inf, li, o25]
            )
        return ap_scores

    @staticmethod
    def collate_fn(batch):
        # Restrict to bs 1
        return batch[0]

@TESTERS.register_module()  
class BracketTester_v2(TesterBase):  
    def test(self):  
        assert self.test_loader.batch_size == 1  
        self.logger.info(">>>>>>>>>>>>>>>> Start Inference >>>>>>>>>>>>>>>>")  
        self.model.eval()  
        results = []  
        predictions_dict = {}  # Dictionary to store all predictions  
        
        # Create results directory in model folder  
        save_path = os.path.join(self.cfg.save_path, "results")  
        os.makedirs(save_path, exist_ok=True)  

        # Timer
        batch_time = AverageMeter()
        
        for idx, data_dict in enumerate(self.test_loader):
            start = time.time()
            data_dict = data_dict[0]  
            fragment_list = data_dict.pop("fragment_list")  
            data_name = data_dict.pop("name")  
            
            # Extract ground truth points (may be None)
            bracket_point_gt = data_dict.pop("bracket", None)  
            incisal_point_gt = data_dict.pop("incisal", None)  
            outer_point_gt = data_dict.pop("outer", None)  
            
            # Move ground truth to GPU if available
            if bracket_point_gt is not None:
                if isinstance(bracket_point_gt, np.ndarray):  
                    bracket_point_gt = torch.from_numpy(bracket_point_gt).cuda()  
                elif isinstance(bracket_point_gt, torch.Tensor):  
                    bracket_point_gt = bracket_point_gt.cuda()  
            
            if incisal_point_gt is not None:
                if isinstance(incisal_point_gt, np.ndarray):  
                    incisal_point_gt = torch.from_numpy(incisal_point_gt).cuda()  
                elif isinstance(incisal_point_gt, torch.Tensor):  
                    incisal_point_gt = incisal_point_gt.cuda()  
            
            if outer_point_gt is not None:
                if isinstance(outer_point_gt, np.ndarray):  
                    outer_point_gt = torch.from_numpy(outer_point_gt).cuda()  
                elif isinstance(outer_point_gt, torch.Tensor):  
                    outer_point_gt = outer_point_gt.cuda()  
            
            # Accumulate predictions across fragments (shape: [1, 9] -> [bracket, incisal, outer])
            pred_sum = torch.zeros(1, 9).cuda()  

            self.logger.info("Processing {} fragments.".format(len(fragment_list))) 
            for fragment in fragment_list:
                for key in fragment.keys():  
                    if isinstance(fragment[key], torch.Tensor):  
                        fragment[key] = fragment[key].cuda(non_blocking=True)  
                
                with torch.no_grad():  
                    output = self.model(fragment)  
                    pred_sum += output["bracket_point_pred"]  
            
            # Average across fragments
            pred = (pred_sum / len(fragment_list)).squeeze(0).cpu()  
            
            # Reshape to 3 points: [bracket, incisal, outer]
            pred_reshaped = pred.reshape(3, 3).numpy()
            bracket_pred = pred_reshaped[0]
            incisal_pred = pred_reshaped[1]
            outer_pred = pred_reshaped[2]
            
            # Store all predictions in dictionary  
            predictions_dict[data_name] = {
                "bracket": bracket_pred.tolist(),
                "incisal": incisal_pred.tolist(),
                "outer": outer_pred.tolist(),
            }
            
            # Calculate errors if ground truth available
            sample_result = {"name": data_name}
            
            if bracket_point_gt is not None:
                bracket_error = torch.norm(torch.from_numpy(bracket_pred).float() - bracket_point_gt.cpu()).item()  
                sample_result["bracket_error"] = bracket_error
            
            if incisal_point_gt is not None:
                incisal_error = torch.norm(torch.from_numpy(incisal_pred).float() - incisal_point_gt.cpu()).item()  
                sample_result["incisal_error"] = incisal_error
            
            if outer_point_gt is not None:
                outer_error = torch.norm(torch.from_numpy(outer_pred).float() - outer_point_gt.cpu()).item()  
                sample_result["outer_error"] = outer_error
            
            if sample_result:
                results.append(sample_result)
                log_msg = f"Test: {data_name}"
                if "bracket_error" in sample_result:
                    log_msg += f", Bracket Error: {sample_result['bracket_error']:.4f}"
                if "incisal_error" in sample_result:
                    log_msg += f", Incisal Error: {sample_result['incisal_error']:.4f}"
                if "outer_error" in sample_result:
                    log_msg += f", Outer Error: {sample_result['outer_error']:.4f}"
                self.logger.info(log_msg)
            batch_time.update(time.time() - start)

        # Save all predictions to single JSON file
        predictions_file = os.path.join(save_path, "results.json")  
        with open(predictions_file, "w") as f:  
            json.dump(predictions_dict, f, indent=4)  
        
        # Save summary with errors
        summary_data = {
            "total_samples": len(results),
            "results": results,
        }
        
        # Compute mean errors
        bracket_errors = [r["bracket_error"] for r in results if "bracket_error" in r]
        incisal_errors = [r["incisal_error"] for r in results if "incisal_error" in r]
        outer_errors = [r["outer_error"] for r in results if "outer_error" in r]
        
        if bracket_errors:
            summary_data["mean_bracket_error"] = float(np.mean(bracket_errors))
        if incisal_errors:
            summary_data["mean_incisal_error"] = float(np.mean(incisal_errors))
        if outer_errors:
            summary_data["mean_outer_error"] = float(np.mean(outer_errors))

        summary_file = os.path.join(save_path, "summary.json")  
        with open(summary_file, "w") as f:  
            json.dump(summary_data, f, indent=4)  
        
        # Log summary
        if bracket_errors:
            self.logger.info(f"Mean Bracket Error: {summary_data['mean_bracket_error']:.4f}")  
        if incisal_errors:
            self.logger.info(f"Mean Incisal Error: {summary_data['mean_incisal_error']:.4f}")  
        if outer_errors:
            self.logger.info(f"Mean Outer Error: {summary_data['mean_outer_error']:.4f}")  
        self.logger.info(f"Avg time per batch: {batch_time.avg:.3f}")
        self.logger.info(f"Predictions saved to: {predictions_file}")
        self.logger.info("<<<<<<<<<<<<<<<<< End Inference <<<<<<<<<<<<<<<<<")
      
    @staticmethod  
    def collate_fn(batch):  
        return batch  # Don't collate, just return the list


@TESTERS.register_module()
class HeatmapTester(TesterBase):
    def test(self):
        assert self.test_loader.batch_size == 1    
        logger = get_root_logger()    
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")    
 
        batch_time = AverageMeter()
        mse_meter = AverageMeter()
        mae_meter = AverageMeter()
        self.model.eval()
 
        save_path = os.path.join(self.cfg.save_path, "results")    
        heatmap_path = os.path.join(save_path, "heatmaps")    
        make_dirs(save_path)
        make_dirs(heatmap_path)    
 
        # Dictionary to store results in the format: sample_name -> [coord_x, coord_y, coord_z]
        all_results = {}
        self.channels = ['bracket', 'incisal', 'outer'] 
        for idx, data_dict in enumerate(self.test_loader):
            start = time.time()
            data_dict = data_dict[0]
            fragment_list = data_dict.pop("fragment_list")
            segment = data_dict.pop("segment")
            data_name = data_dict.pop("name")

            # bracket = data_dict.pop("bracket")  # Get bracket point if available
            # incisal = data_dict.pop("incisal")
            # outer = data_dict.pop("outer")
 
            pred_save_path = os.path.join(heatmap_path, f"{data_name}_pred.npy")
            pred = torch.zeros(segment.shape[0], 3).cuda()    

            for i in range(len(fragment_list)):
                fragment_batch_size = 1  
                s_i, e_i = i * fragment_batch_size, min(    
                    (i + 1) * fragment_batch_size, len(fragment_list)    
                )    
                input_dict = collate_fn(fragment_list[s_i:e_i])    
                for key in input_dict.keys():    
                    if isinstance(input_dict[key], torch.Tensor):    
                        input_dict[key] = input_dict[key].cuda(non_blocking=True)    
                    
                idx_part = input_dict["index"]    
                with torch.no_grad():    
                    output_dict = self.model(input_dict)    
                    pred_part = output_dict["seg_logits"]    
                    if pred_part.dim() > 2:    
                        pred_part = pred_part.squeeze(-1)    
                        
                    if self.cfg.empty_cache:    
                        torch.cuda.empty_cache()    
                        
                    bs = 0    
                    for be in input_dict["offset"]:  
                        pred[idx_part[bs:be]] += pred_part[bs:be]    
                        bs = be    
                    
                logger.info(    
                    f"Test: {idx + 1}/{len(self.test_loader)}-{data_name}, "
                    f"Batch: {i}/{len(fragment_list)}"    
                )    
                
            pred = pred / len(fragment_list)  
            pred = pred.cpu().numpy()    
            pred = np.clip(pred, 0.0, 1.0) # N x 3
             
            # Handle inverse mapping    
            if "origin_segment" in data_dict.keys():    
                assert "inverse" in data_dict.keys()    
                pred = pred[data_dict["inverse"]]    
                segment = data_dict["origin_segment"]    
                
            #np.save(pred_save_path, pred)
            
            # Compute metrics    
            mse = np.mean((pred - segment) ** 2)    
            mae = np.mean(np.abs(pred - segment))    
            mse_meter.update(mse)    
            mae_meter.update(mae)   
                
            batch_time.update(time.time() - start)    
            logger.info(    
                f"Test: {data_name} [{idx + 1}/{len(self.test_loader)}]-{segment.size} "    
                f"Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "    
                f"MSE {mse:.6f} ({mse_meter.avg:.6f}) " 
                f"MAE {mae:.6f} ({mae_meter.avg:.6f})"    
            )    
            # Compute proposal point from top 5% of heatmap values
            full_path = data_dict.get("full_path")
            mesh = trimesh.load(full_path, force="mesh")
            verts = np.asarray(mesh.vertices)
            channels_proposals = {}
            # need to activate this if in production, for now it's always active
            #bm = BracketMapper()
            #orientation_map = bm.get_sim_values(full_path, mesh, save=False)
            #bm.plot_mesh(full_path, orientation_map)
            # ============== load orientation map ==================
            #full_path_p = Path(full_path)
            #orientation_map_p = full_path_p.with_name(full_path_p.stem + "_orient.npy")
            #orientation_map = np.load(orientation_map_p)
            # ======================================================
            for channel_idx in range(3):
                channel_pred = pred[:, channel_idx] 
                #channel_pred = channel_pred * orientation_map
                thresh = np.percentile(channel_pred, 95)
                mask = channel_pred >= thresh
                inds = np.nonzero(mask)[0]

                if inds.size == 0:
                    # fallback to max
                    max_idx = int(np.argmax(channel_pred))
                    proposal = verts[max_idx].astype(np.float64)
                else:
                    weights = channel_pred[inds].astype(np.float64)
                    wsum = weights.sum()
                    if wsum == 0:
                        max_idx = int(np.argmax(channel_pred))
                        proposal = verts[max_idx].astype(np.float64)
                    else:
                        proposal = (verts[inds].astype(np.float64).T @ weights) / wsum
                channels_proposals[self.channels[channel_idx]] = proposal.tolist()
            all_results[data_name] = channels_proposals

        logger.info(    
            f"Val result: MSE/MAE {mse_meter.avg:.6f}/{mae_meter.avg:.6f}"    
        ) 
 
        # Save all results to JSON file
        results_json_path = os.path.join(save_path, "predictions.json")
        try:
            with open(results_json_path, 'w') as f:
                json.dump(all_results, f, indent=4)
            logger.info(f"Saved results to {results_json_path}")
        except Exception as e:
            logger.warning(f"Failed to save results JSON: {e}")
        
        logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
 
    @staticmethod
    def collate_fn(batch):
        return batch

@TESTERS.register_module()
class HeatmapTesterV2(TesterBase):
    # Unified channel layout:
    # 0: Bracket
    # 1: Incisal
    # 2: Gingival
    # 3: Planar
    # 4: Mesial
    # 5: Distal
    # 6: Cusp
    # 7: InnerPoint
    # 8: OuterPoint
    # 9: FacialPoint

    SINGLE_POINT_CHANNELS = {
        0: "Bracket",
        1: "Incisal",
        2: "Gingival",
        4: "Mesial",
        5: "Distal",
        7: "InnerPoint",
        8: "OuterPoint",
        9: "FacialPoint",
    }
    MULTI_POINT_CHANNELS = { # fixed K=4 points
        3: ("Planar", 4),
    }
    VARIABLE_POINT_CHANNELS = { # variable cusps
        6: "Cusp",
    }

    def _extract_single_point_proposal(self, verts, channel_pred, percentile=95):
        thresh = np.percentile(channel_pred, percentile)
        inds = np.nonzero(channel_pred >= thresh)[0]

        if inds.size == 0:
            max_idx = int(np.argmax(channel_pred))
            return verts[max_idx].astype(np.float64)

        local_verts = verts[inds].astype(np.float64)
        weights = channel_pred[inds].astype(np.float64)

        wsum = weights.sum()
        if wsum <= 0:
            max_idx = int(np.argmax(channel_pred))
            return verts[max_idx].astype(np.float64)

        centroid = (local_verts.T @ weights) / wsum
        dists = np.sum((local_verts - centroid) ** 2, axis=1)
        nearest_idx = np.argmin(dists)

        return local_verts[nearest_idx]

    def _extract_multi_point_proposal(self, verts, channel_pred, k, percentile=95):
        thresh = np.percentile(channel_pred, percentile)
        inds = np.nonzero(channel_pred >= thresh)[0]

        if inds.size < k:
            inds = np.argpartition(channel_pred, -k)[-k:]

        local_verts = verts[inds].astype(np.float64)
        weights = channel_pred[inds].astype(np.float64)

        kmeans = KMeans(n_clusters=k, n_init=10, random_state=0).fit(
            local_verts, sample_weight=weights
        )
        proposals = []
        for center in kmeans.cluster_centers_:
            dists = np.sum((local_verts - center) ** 2, axis=1)
            proposals.append(local_verts[np.argmin(dists)].tolist())
        return proposals

    def _get_num_clusters_from_components(self, mesh, inds, min_k, max_k):
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components

        ind_set = set(inds.tolist())
        local_idx = {g: l for l, g in enumerate(inds.tolist())}
        n = len(inds)

        rows, cols = [], []
        for v0, v1 in mesh.edges_unique:
            v0, v1 = int(v0), int(v1)
            if v0 in ind_set and v1 in ind_set:
                rows += [local_idx[v0], local_idx[v1]]
                cols += [local_idx[v1], local_idx[v0]]

        graph = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
        n_components, _ = connected_components(graph, directed=False)
        return int(np.clip(n_components, min_k, max_k))

    def _extract_variable_point_proposal(self, mesh, channel_pred, min_k=2, max_k=6, percentile=95):
        verts = np.asarray(mesh.vertices)
        thresh = np.percentile(channel_pred, percentile)
        inds = np.nonzero(channel_pred >= thresh)[0]

        k = self._get_num_clusters_from_components(mesh, inds, min_k, max_k)

        if inds.size < k:
            inds = np.argpartition(channel_pred, -k)[-k:]

        local_verts = verts[inds].astype(np.float64)
        weights = channel_pred[inds].astype(np.float64)

        kmeans = KMeans(n_clusters=k, n_init=10, random_state=0).fit(
            local_verts, sample_weight=weights
        )
        proposals = []
        for center in kmeans.cluster_centers_:
            dists = np.sum((local_verts - center) ** 2, axis=1)
            proposals.append(local_verts[np.argmin(dists)].tolist())
        return proposals

    def _swap_mesial_distal(self, channels_proposals):
        channels_proposals['Mesial'], channels_proposals['Distal'] = channels_proposals['Distal'], channels_proposals['Mesial']

    def _mesial_distal_correction(self, channels_proposals:dict, full_path:str):
        patient_left = list(range(23, 29)) + list(range(33, 39))
        patient_right = list(range(13, 19)) + list(range(43, 49))
        front_left =  [21, 22, 31, 32]
        front_right = [11, 12, 41, 42]
        file_name = Path(full_path).stem
        ide, arch, _, FDI = file_name.split("_")
        FDI = int(FDI)
        mesial = channels_proposals["Mesial"]
        distal = channels_proposals["Distal"]
        if FDI in patient_right or FDI in patient_left:
            y_mesial, y_distal = mesial[1], distal[1]
            if y_distal > y_mesial: self._swap_mesial_distal(channels_proposals)
        elif FDI in front_left:
            x_mesial, x_distal = mesial[0], distal[0]
            if x_distal > x_mesial: self._swap_mesial_distal(channels_proposals)
        elif FDI in front_right:
            x_mesial, x_distal = mesial[0], distal[0]
            if x_mesial > x_distal: self._swap_mesial_distal(channels_proposals)

    def test(self):
        assert self.test_loader.batch_size == 1
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

        batch_time = AverageMeter()
        mse_meter = AverageMeter()
        mae_meter = AverageMeter()
        self.model.eval()

        save_path = os.path.join(self.cfg.save_path, "results")
        heatmap_path = os.path.join(save_path, "heatmaps")
        make_dirs(save_path)
        make_dirs(heatmap_path)

        all_results = {}

        for idx, data_dict in enumerate(self.test_loader):
            start = time.time()
            data_dict = data_dict[0]

            fragment_list = data_dict.pop("fragment_list")
            segment = data_dict.pop("segment")
            validity_mask = data_dict.pop("validity_mask")
            _ = data_dict.pop("landmarks", None)
            data_name = data_dict.pop("name")
            full_path = data_dict.get("full_path")

            pred = torch.zeros(segment.shape[0], 10).cuda()

            for i in range(len(fragment_list)):
                fragment_batch_size = 1
                s_i, e_i = i * fragment_batch_size, min(
                    (i + 1) * fragment_batch_size, len(fragment_list)
                )
                input_dict = collate_fn(fragment_list[s_i:e_i])

                for key in input_dict.keys():
                    if isinstance(input_dict[key], torch.Tensor):
                        input_dict[key] = input_dict[key].cuda(non_blocking=True)

                idx_part = input_dict["index"]

                with torch.no_grad():
                    output_dict = self.model(input_dict)
                    pred_part = output_dict["seg_logits"]

                    if pred_part.dim() > 2:
                        pred_part = pred_part.squeeze(-1)

                    if self.cfg.empty_cache:
                        torch.cuda.empty_cache()

                    bs = 0
                    for be in input_dict["offset"]:
                        pred[idx_part[bs:be]] += pred_part[bs:be]
                        bs = be

                logger.info(
                    f"Test: {idx + 1}/{len(self.test_loader)}-{data_name}, "
                    f"Batch: {i}/{len(fragment_list)}"
                )

            pred = pred / len(fragment_list)
            pred = pred.cpu().numpy()
            pred = np.clip(pred, 0.0, 1.0)

            if "origin_segment" in data_dict:
                assert "inverse" in data_dict
                pred = pred[data_dict["inverse"]]
                segment = data_dict["origin_segment"]
                if "origin_validity_mask" in data_dict:
                    validity_mask = data_dict["origin_validity_mask"]
                else:
                    validity_mask = validity_mask[data_dict["inverse"]]

            valid_mask = validity_mask.astype(np.float32)
            if valid_mask.ndim == 1:
                valid_mask_metrics = np.broadcast_to(valid_mask[None, :], segment.shape)
            else:
                valid_mask_metrics = valid_mask

            valid_count = np.sum(valid_mask_metrics)

            if valid_count > 0:
                diff = pred - segment
                mse = np.sum((diff ** 2) * valid_mask_metrics) / valid_count
                mae = np.sum(np.abs(diff) * valid_mask_metrics) / valid_count
                mse_meter.update(mse)
                mae_meter.update(mae)
            else:
                mse = float("nan")
                mae = float("nan")
                logger.warning(f"No valid channels for sample {data_name}, skipping metrics.")

            batch_time.update(time.time() - start)
            logger.info(
                f"Test: {data_name} [{idx + 1}/{len(self.test_loader)}]-{segment.size} "
                f"Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                f"MSE {mse:.6f} ({mse_meter.avg:.6f}) "
                f"MAE {mae:.6f} ({mae_meter.avg:.6f})"
            )

            mesh = trimesh.load(full_path, force="mesh")
            verts = np.asarray(mesh.vertices)

            channels_proposals = {}
            for channel_idx, landmark_name in self.SINGLE_POINT_CHANNELS.items():
                channel_pred = pred[:, channel_idx]
                proposal = self._extract_single_point_proposal(verts, channel_pred)
                channels_proposals[landmark_name] = proposal.tolist()

            self._mesial_distal_correction(channels_proposals, full_path)

            for channel_idx, (landmark_name, k) in self.MULTI_POINT_CHANNELS.items():
                channel_pred = pred[:, channel_idx]
                channels_proposals[landmark_name] = self._extract_multi_point_proposal(verts, channel_pred, k)

            for channel_idx, landmark_name in self.VARIABLE_POINT_CHANNELS.items():
                channel_pred = pred[:, channel_idx]
                channels_proposals[landmark_name] = self._extract_variable_point_proposal(mesh, channel_pred)

            all_results[data_name] = channels_proposals

        logger.info(
            f"Val result: MSE/MAE {mse_meter.avg:.6f}/{mae_meter.avg:.6f}"
        )

        results_json_path = os.path.join(save_path, "predictions.json")
        with open(results_json_path, "w") as f:
            json.dump(all_results, f, indent=4)
        logger.info(f"Saved results to {results_json_path}")

        logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return batch