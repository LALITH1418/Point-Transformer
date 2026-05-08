import math
import torch
import torch.nn as nn


#pred_boxes has 8 values: (h, w, l, x, y, z, sin_ry, cos_ry)
#true_boxes has 7 values: (h, w, l, x, y, z, ry/pi)
class CombinedLoss(nn.Module):
    def __init__(self, alpha=2.5, beta=2.0, class_weights=(1.0, 2.0, 4.0)):
        super().__init__()
        self.smooth_l1 = nn.SmoothL1Loss()
        self.alpha = alpha
        self.beta = beta
        self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32))
        self.cls_loss = nn.CrossEntropyLoss(weight=self.class_weights)

    def forward(self, pred_boxes, true_boxes, pred_classes, true_classes):
        cls_loss = self.cls_loss(pred_classes, true_classes)

        #converting yaw angle to sin/cos so the loss doesn't break
        true_ry = true_boxes[:, 6] * math.pi
        true_sin = torch.sin(true_ry).unsqueeze(1)
        true_cos = torch.cos(true_ry).unsqueeze(1)
        true_extended = torch.cat([true_boxes[:, :6], true_sin, true_cos], dim=1)
        reg_loss = self.smooth_l1(pred_boxes, true_extended)

        iou_per_sample = self.calculate_iou(pred_boxes, true_boxes)
        iou_loss = 1.0 - iou_per_sample.mean()

        total = self.alpha * reg_loss + self.beta * iou_loss + cls_loss
        info = {
            "cls_loss": cls_loss.detach(),
            "reg_loss": reg_loss.detach(),
            "iou_loss": iou_loss.detach(),
            "iou_per_sample": iou_per_sample.detach(),
        }
        return total, info

    @staticmethod
    def calculate_iou(pred_boxes, true_boxes):
        h_p, w_p, l_p = pred_boxes[:, 0].abs(), pred_boxes[:, 1].abs(), pred_boxes[:, 2].abs()
        x_p, y_p, z_p = pred_boxes[:, 3], pred_boxes[:, 4], pred_boxes[:, 5]
        h_t, w_t, l_t = true_boxes[:, 0].abs(), true_boxes[:, 1].abs(), true_boxes[:, 2].abs()
        x_t, y_t, z_t = true_boxes[:, 3], true_boxes[:, 4], true_boxes[:, 5]

        p_min = torch.stack([x_p - l_p / 2, y_p - w_p / 2, z_p - h_p / 2], dim=1)
        p_max = torch.stack([x_p + l_p / 2, y_p + w_p / 2, z_p + h_p / 2], dim=1)
        t_min = torch.stack([x_t - l_t / 2, y_t - w_t / 2, z_t - h_t / 2], dim=1)
        t_max = torch.stack([x_t + l_t / 2, y_t + w_t / 2, z_t + h_t / 2], dim=1)

        inter = torch.clamp(torch.min(p_max, t_max) - torch.max(p_min, t_min), min=0)
        intersection = inter.prod(dim=1)

        p_vol = l_p * w_p * h_p
        t_vol = l_t * w_t * h_t
        union = p_vol + t_vol - intersection
        return intersection / (union + 1e-6)
