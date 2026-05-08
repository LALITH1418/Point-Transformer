import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from Networks.pointnet_loader import load_pointnet_model


# encodes the 3D offset between two points
class RelativePositionMLP(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
        )

    def forward(self, delta_xyz):
        return self.mlp(delta_xyz)


# vector attention from point transformer
# uses per-channel weights instead of one scalar per pair
class VectorAttention(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.query_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        self.key_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        self.value_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        self.pos_mlp = RelativePositionMLP(feature_dim)
        self.attn_mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
        )

    def forward(self, x, xyz):
        # x: [B, N, D], xyz: [B, N, 3]
        q = self.query_proj(x)
        k = self.key_proj(x)
        v = self.value_proj(x)

        # pairwise position differences
        delta_xyz = xyz.unsqueeze(2) - xyz.unsqueeze(1)
        pos_bias = self.pos_mlp(delta_xyz)

        attn_input = q.unsqueeze(2) - k.unsqueeze(1) + pos_bias
        attn_weights = torch.softmax(self.attn_mlp(attn_input), dim=2)

        out = (attn_weights * (v.unsqueeze(1) + pos_bias)).sum(dim=2)
        return out


# SwiGLU FFN, hidden dim = 4 * input dim
class SwiGLUFFN(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        hidden_dim = 4 * feature_dim
        self.gate_proj = nn.Linear(feature_dim, hidden_dim)
        self.value_proj = nn.Linear(feature_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, feature_dim)

    def forward(self, x):
        return self.out_proj(F.silu(self.gate_proj(x)) * self.value_proj(x))


# one transformer block: attention + ffn with pre-norm
class PointTransformerLayer(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(feature_dim)
        self.attention = VectorAttention(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.ffn = SwiGLUFFN(feature_dim)

    def forward(self, x, xyz):
        x = x + self.attention(self.norm1(x), xyz)
        x = x + self.ffn(self.norm2(x))
        return x


# main model
class ObjectDetectionModel(nn.Module):
    def __init__(self, num_classes=3, feature_dim=64, num_layers=16):
        super().__init__()

        # load pointnet++ and freeze it
        self.pointnet = load_pointnet_model()
        for param in self.pointnet.parameters():
            param.requires_grad = False

        # project pointnet features (640) + xyz (3) down to feature_dim
        self.input_projection = nn.Linear(640 + 3, feature_dim)

        # learnable cls token (like in BERT/ViT)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, feature_dim))

        # transformer stack
        self.transformer_layers = nn.ModuleList(
            [PointTransformerLayer(feature_dim) for _ in range(num_layers)]
        )

        self.final_norm = nn.LayerNorm(feature_dim)

        # classification head
        self.class_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

        # box regression head: h, w, l, x, y, z_offset, sin(ry), cos(ry)
        # +1 for the depth anchor input
        self.box_head = nn.Sequential(
            nn.Linear(feature_dim + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 8),
        )

    def forward(self, x):
        # x: [B, N, 3] in camera frame
        B = x.size(0)

        # depth anchor (normalized) so the box head only learns a residual
        mean_depth = x[..., 2].mean(dim=1, keepdim=True) / 70.0

        # pointnet expects [B, 3, N]
        points = x.permute(0, 2, 1)
        xyz1, feats1 = self.pointnet.sa1(points, None)
        xyz2, feats2 = self.pointnet.sa2(xyz1, feats1)
        # now we have 128 tokens with 640-dim features

        point_xyz = xyz2.permute(0, 2, 1)  # save for position encoding

        # concat xyz + features, then project
        combined = torch.cat([xyz2, feats2], dim=1)
        point_tokens = combined.permute(0, 2, 1)
        point_tokens = self.input_projection(point_tokens)

        # prepend cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, point_tokens], dim=1)

        # cls token has no real position, so use (0,0,0)
        cls_xyz = torch.zeros(B, 1, 3, device=x.device)
        xyz = torch.cat([cls_xyz, point_xyz], dim=1)

        # run through transformer (gradient checkpointing to save memory)
        for layer in self.transformer_layers:
            tokens = checkpoint(layer, tokens, xyz, use_reentrant=False)

        tokens = self.final_norm(tokens)

        # use cls token for predictions
        cls_output = tokens[:, 0]

        class_logits = self.class_head(cls_output)

        # add depth anchor as input to box head
        reg_input = torch.cat([cls_output, mean_depth], dim=1)
        box_preds = self.box_head(reg_input)

        # final z = anchor + predicted offset
        z_final = mean_depth.squeeze(1) + box_preds[:, 5]

        pred_boxes = torch.stack([
            box_preds[:, 0], box_preds[:, 1], box_preds[:, 2],
            box_preds[:, 3], box_preds[:, 4], z_final,
            box_preds[:, 6], box_preds[:, 7],
        ], dim=1)

        return class_logits, pred_boxes
