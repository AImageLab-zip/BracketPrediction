from pointcept.models.builder import MODELS, build_model  
import torch
import torch.nn as nn  
from pointcept.models.utils.structure import Point
import math
  
@MODELS.register_module("LandmarksHeatmapV2")
class LandmarksHeatmapV2(nn.Module):  
    def __init__(
        self,
        backbone,
        backbone_out_channels=64,
        loss_type: str = "l1",
        dice_smooth: float = 1e-6,
    ):
        super().__init__()  
        self.backbone = build_model(backbone)
        self.regression_head = nn.Linear(backbone_out_channels, 10)
        self.loss_type = loss_type.lower() if isinstance(loss_type, str) else "l1"
        self.dice_smooth = float(dice_smooth)
      
    def forward(self, data_dict):  
        point = self.backbone(data_dict)
 
        # Handle Point structure from PT-v3  
        if isinstance(point, Point):  
            while "pooling_parent" in point.keys():  
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent  
            feat = point.feat  
        else:
            feat = point
 
        # Regression output
        heatmap_pred = self.regression_head(feat)
        heatmap_pred = torch.sigmoid(heatmap_pred)  # [0, 1] range
        # for validation also return predictions to compute extra metrics later.
        out = {} if self.training else {"seg_logits": heatmap_pred}
        if "segment" in data_dict: # training or validation, so we need to compute loss
            heatmap_gt = data_dict["segment"].float()
            validity_mask = data_dict["validity_mask"].float()
            pred = heatmap_pred
            gt = heatmap_gt
            mask = validity_mask

            if self.loss_type in ("l1", "mae"):
                l1 = torch.abs(pred - gt)
                # Keep only valid entries
                l1 = l1 * mask
                denom = mask.sum().clamp_min(1.0)
                loss = l1.sum() / denom

            elif self.loss_type in ("dice", "softdice"):
                # soft-dice over all elements (channels x batch x points)
                # apply mask to both pred and gt
                p = pred * mask
                g = gt * mask
                intersection = (p * g).sum()
                cardinality = p.sum() + g.sum()
                dice = (2.0 * intersection + self.dice_smooth) / (cardinality + self.dice_smooth)
                loss = 1.0 - dice

            else:
                raise ValueError(f"Unsupported loss_type '{self.loss_type}'")
            out["loss"] = loss
        return out