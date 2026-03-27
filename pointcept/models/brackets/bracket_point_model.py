"""
Model for Bracket Point Prediction
Place this file in: pointcept/models/bracket_point_model.py
"""
import torch
import torch.nn as nn
from pointcept.models.builder import MODELS
from pointcept.models.losses import build_criteria
from torch_scatter import segment_csr  
import torch.nn.functional as F

from ..builder import build_model  
from pointcept.models.utils.structure import Point  
import json
import os
from collections import defaultdict

@MODELS.register_module("LandmarksRegression")  
class LandmarksRegression(nn.Module):  
    """
    """  
      
    def __init__(  
        self,
        backbone,
        backbone_out_channels=96,
        output_dim=3,  # 3D point coordinates
        class_embedding=False,
        output_dir:str = "output",
    ):  
        super().__init__()  
          
        self.backbone = build_model(backbone)
        self.output_dir = output_dir
        self.class_embedding=class_embedding

        # ========= loss weights =============
        self.num_classes = 8 # FDI index % 10
        self.tooth_embedding_dim = 128
        # embedding layer for the one hot tooth class
        self.embedder = nn.Embedding(self.num_classes, self.tooth_embedding_dim)
        # Separate regression heads for each point (bracket, incisal, outer)
        head_input = backbone_out_channels if not self.class_embedding else backbone_out_channels + self.tooth_embedding_dim
        
        # Bracket head
        self.bracket_head = nn.Sequential( 
            nn.Linear(head_input, 256),  
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, output_dim),  # 3 coordinates
        )
        
        # Incisal head
        self.incisal_head = nn.Sequential( 
            nn.Linear(head_input, 256),  
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, output_dim),  # 3 coordinates
        )

        # Outer head
        self.outer_head = nn.Sequential( 
            nn.Linear(head_input, 256),  
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, output_dim),  # 3 coordinates
        )
 
    def _save(self, input_dict: dict, bracket_point_pred: torch.Tensor) -> None:
        """
        Writes the predicted points as JSON files.

        Each file will be named:
            {sample_name}_epoch{N}.json
        where N is how many times this function has been called for that sample.

        Args:
            input_dict (dict): Contains "name" (list of sample names without extension).
            bracket_point_pred (torch.Tensor): Predictions of shape [B, 9] (3 points * 3 coordinates).
        """
        # Initialize a counter the first time this is called
        if not hasattr(self, "_save_counter"):
            self._save_counter = defaultdict(int)

        os.makedirs(self.output_dir, exist_ok=True)

        names = input_dict["name"]
        preds = bracket_point_pred.detach().cpu().numpy()  # convert to numpy for JSON serialization

        for name, pred in zip(names, preds):
            # Increment epoch counter for this specific sample
            self._save_counter[name] += 1
            epoch_idx = self._save_counter[name]

            # Construct output filename
            filename = f"{name}_epoch{epoch_idx}.json"
            filepath = os.path.join(self.output_dir, filename)

            # Reshape predictions to 3 points
            bracket = pred[:3].tolist()
            incisal = pred[3:6].tolist()
            outer = pred[6:9].tolist()
            
            # Write coordinates to JSON
            with open(filepath, "w") as f:
                json.dump({
                    "bracket": bracket,
                    "incisal": incisal,
                    "outer": outer
                }, f, indent=4)


    def forward(self, input_dict):
        point = self.backbone(input_dict)
        # Handle Point structure from voxel-based backbones    
        if isinstance(point, Point):    
            # Global average pooling across all points    
            point.feat = segment_csr(    
                src=point.feat,
                indptr=nn.functional.pad(point.offset, (1, 0)),    
                reduce="mean",
            )    
            feat = point.feat    
        else:
            feat = point
        if self.class_embedding:
            if type(input_dict["name"]) == type(""): # then we are in test mode
                class_indices = torch.tensor([(int(input_dict["name"].split("_")[-1]) % 10) - 1]).long()
            else:
                class_indices = torch.tensor(  
                    [(int(x.split("_")[-1]) % 10) - 1 for x in input_dict["name"]],  
                    device=feat.device  
                ).long()
            class_indices = torch.clamp(class_indices, min=0, max=7)
            class_embeddings = self.embedder(class_indices)
            input_tensor = torch.cat([feat, class_embeddings], dim=1)
        else:
            input_tensor = feat 

        # Regress position of 3 points (bracket, incisal, outer), each with 3D coordinates
        bracket_pred = self.bracket_head(input_tensor)  # [B, 3]
        incisal_pred = self.incisal_head(input_tensor)  # [B, 3]
        outer_pred = self.outer_head(input_tensor)     # [B, 3]
        
        # Concatenate predictions for output (shape [B, 9])
        prediction = torch.cat([bracket_pred, incisal_pred, outer_pred], dim=1)  # [B, 9]
        
        if not self.training:
            out = {"bracket_point_pred": prediction}  # Add predictions to output (keep as [B, 9])
        else:
            out = {}
         
        # Compute MAE loss if ground truth available
        if "bracket" in input_dict:
            bracket_target = input_dict["bracket"].view_as(bracket_pred)
            incisal_target = input_dict["incisal"].view_as(incisal_pred)
            outer_target = input_dict["outer"].view_as(outer_pred)
            
            bracket_mae = nn.functional.l1_loss(bracket_pred, bracket_target)
            incisal_mae = nn.functional.l1_loss(incisal_pred, incisal_target)
            outer_mae = nn.functional.l1_loss(outer_pred, outer_target)
            
            bracket_cos_sim = nn.functional.cosine_similarity(bracket_pred, bracket_target).mean()
            incisal_cos_sim = nn.functional.cosine_similarity(incisal_pred, incisal_target).mean()
            outer_cos_sim = nn.functional.cosine_similarity(outer_pred, outer_target).mean()
            
            bracket_cos_dist = 1.0 - bracket_cos_sim
            incisal_cos_dist = 1.0 - incisal_cos_sim
            outer_cos_dist = 1.0 - outer_cos_sim
            
            mae = bracket_mae + incisal_mae + outer_mae
            cos_dist = (bracket_cos_dist + incisal_cos_dist + outer_cos_dist) / 3.0
            
            out["bracket_mae"] = bracket_mae
            out["incisal_mae"] = incisal_mae
            out["outer_mae"] = outer_mae
            out["bracket_cos_dist"] = bracket_cos_dist
            out["incisal_cos_dist"] = incisal_cos_dist
            out["outer_cos_dist"] = outer_cos_dist
            
            out["mae"] = mae
            out["cos_dist"] = cos_dist
            out["loss"] = mae
 
        return out