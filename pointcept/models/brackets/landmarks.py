from pointcept.models.builder import MODELS, build_model  
import torch
import torch.nn as nn  
from pointcept.models.utils.structure import Point
  
@MODELS.register_module("LandmarksHeatmapV2")
class LandmarksHeatmapV2(nn.Module):  
    def __init__(
        self,
        backbone,
        backbone_out_channels=64,
    ):
        super().__init__()  
        self.backbone = build_model(backbone)
        self.regression_head = nn.Linear(backbone_out_channels, 10)
      
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
            l1 = torch.abs(heatmap_pred - heatmap_gt)
            # Keep only valid channels
            l1 = l1 * validity_mask
            # Normalize by number of valid entries only
            denom = validity_mask.sum().clamp_min(1.0)
            loss = l1.sum() / denom
            out["loss"] = loss
        return out