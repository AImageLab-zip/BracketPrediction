import torch
import torch.nn as nn
from torch_scatter import segment_csr

from pointcept.models.builder import MODELS, build_model
from pointcept.models.utils.structure import Point


@MODELS.register_module("IOSOrientationModel")
class IOSOrientationModel(nn.Module):
    def __init__(self, backbone, backbone_out_channels=512, num_angle_bins=180, angle_min_deg=1.0):
        super().__init__()
        self.num_angle_bins = num_angle_bins
        self.angle_min_deg = angle_min_deg
        self.backbone = build_model(backbone)
        self.head = nn.Sequential(
            nn.Linear(backbone_out_channels, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3 * num_angle_bins),
        )
        self.register_buffer(
            "angle_values",
            torch.arange(num_angle_bins, dtype=torch.float32) + angle_min_deg,
            persistent=False,
        )

    def _global_feature(self, input_dict):
        point = self.backbone(input_dict)
        if isinstance(point, Point):
            return segment_csr(
                src=point.feat,
                indptr=nn.functional.pad(point.offset, (1, 0)),
                reduce="mean",
            )
        return point

    @staticmethod
    def _rotation_matrix_from_angles(angles_deg):
        angles = torch.deg2rad(angles_deg)
        x, y, z = angles[:, 0], angles[:, 1], angles[:, 2]
        cx, sx = torch.cos(x), torch.sin(x)
        cy, sy = torch.cos(y), torch.sin(y)
        cz, sz = torch.cos(z), torch.sin(z)

        zeros = torch.zeros_like(x)
        ones = torch.ones_like(x)
        rx = torch.stack(
            [
                ones, zeros, zeros,
                zeros, cx, -sx,
                zeros, sx, cx,
            ],
            dim=1,
        ).reshape(-1, 3, 3)
        ry = torch.stack(
            [
                cy, zeros, sy,
                zeros, ones, zeros,
                -sy, zeros, cy,
            ],
            dim=1,
        ).reshape(-1, 3, 3)
        rz = torch.stack(
            [
                cz, -sz, zeros,
                sz, cz, zeros,
                zeros, zeros, ones,
            ],
            dim=1,
        ).reshape(-1, 3, 3)
        return rz @ ry @ rx

    def _matrix_from_angles(self, angles_deg):
        batch = angles_deg.shape[0]
        matrix = torch.eye(4, device=angles_deg.device, dtype=angles_deg.dtype).unsqueeze(0).repeat(batch, 1, 1)
        matrix[:, :3, :3] = self._rotation_matrix_from_angles(angles_deg)
        return matrix

    def _angles_from_logits(self, logits):
        if self.training:
            probabilities = torch.softmax(logits, dim=-1)
            return torch.sum(probabilities * self.angle_values.to(logits.dtype), dim=-1)
        return torch.argmax(logits, dim=-1).to(logits.dtype) + self.angle_min_deg

    @staticmethod
    def _apply_matrix(coord, matrix, offset):
        starts = nn.functional.pad(offset, (1, 0))[:-1]
        out = torch.empty_like(coord)
        for sample_idx, (start, end) in enumerate(zip(starts.tolist(), offset.tolist())):
            linear = matrix[sample_idx, :3, :3]
            translation = matrix[sample_idx, :3, 3]
            out[start:end] = coord[start:end] @ linear.T + translation
        return out

    def forward(self, input_dict):
        feat = self._global_feature(input_dict)
        logits = self.head(feat).reshape(-1, 3, self.num_angle_bins)
        angles = self._angles_from_logits(logits)
        matrix = self._matrix_from_angles(angles)
        recon_coord = self._apply_matrix(input_dict["coord"], matrix, input_dict["offset"])

        out = {}
        if not self.training:
            out["angle_logits"] = logits
            out["angle"] = angles
            out["transform"] = matrix
            out["recon_coord"] = recon_coord
        if "target_coord" in input_dict:
            recon_loss = nn.functional.smooth_l1_loss(recon_coord, input_dict["target_coord"])
            if "angle" in input_dict:
                angle_target = input_dict["angle"].long().reshape(-1, 3)
                angle_loss = nn.functional.cross_entropy(
                    logits.reshape(-1, self.num_angle_bins), angle_target.reshape(-1)
                )
            else:
                angle_loss = recon_loss.new_tensor(0.0)
            out["recon_loss"] = recon_loss
            out["angle_loss"] = angle_loss
            out["loss"] = angle_loss
        return out
