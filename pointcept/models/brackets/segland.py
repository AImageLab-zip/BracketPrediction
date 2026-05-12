import torch
import torch.nn as nn
from pointcept.models.builder import MODELS, build_model
from pointcept.models.default import DefaultSegmentorV2

@MODELS.register_module()  
class SegLand(nn.Module):  
    def __init__(self,
                 segmentor_config,
                 tooth_model_config,
                 num_classes):
        super().__init__()
 
        # Build segmentor via registry to ensure nested configs (backbone etc.) are handled
        self.segmentor = build_model(segmentor_config)
        self.tooth_model = build_model(tooth_model_config)
        self.num_classes = num_classes
 
    def forward(self, input_dict):
        # Step 1: Segment the mesh into teeth
        self.segmentor.eval()
        seg_results = self.segmentor(input_dict)
        seg_logits = seg_results["seg_logits"]
        # Only return segmentation logits and leave tooth-model inference
        # to the tester so it can apply per-tooth normalization and inverse transforms.
        return {
            "seg_logits": seg_logits,
        }
 
    def extract_tooth_meshes(self, input_dict, seg_logits):  
        """Extract individual tooth meshes based on segmentation results"""  
        # Get predicted tooth labels (excluding background class 0)  
        pred_labels = torch.argmax(seg_logits, dim=-1) 
        tooth_meshes = []  
        unique_labels = torch.unique(pred_labels)  
        for tooth_id in unique_labels:  
            if tooth_id == 0:  # Skip background  
                continue  
            tooth_mask = (pred_labels == tooth_id)  
            if tooth_mask.sum() == 0:  
                continue
            tooth_dict = {}  
            for key in ["coord", "feat", "offset"]:
                if key in input_dict:  
                    tooth_dict[key] = input_dict[key][tooth_mask]  
            # Recalculate offset for single tooth  
            if "offset" in tooth_dict:  
                tooth_dict["offset"] = torch.tensor([len(tooth_dict["coord"])],   
                                                   device=tooth_dict["coord"].device) 
            # attach original full_path if available so testers can know jaw
            if "full_path" in input_dict:
                tooth_dict["full_path"] = input_dict["full_path"]
                # crude jaw detection from path
                fp = str(input_dict["full_path"]).lower()
                tooth_dict["jaw"] = "upper" if "upper" in fp else "lower"
            tooth_meshes.append(tooth_dict)  
        return tooth_meshes