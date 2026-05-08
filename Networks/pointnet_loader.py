import sys
import os
import torch

# add pointnet++ folder to path so we can import it
sys.path.append(os.path.abspath('./Pointnet_Pointnet2_pytorch/models'))

from Pointnet_Pointnet2_pytorch.models.pointnet2_cls_msg import get_model


def load_pointnet_model():
    # 40 classes for modelnet40, no normals in our lidar data
    model = get_model(num_class=40, normal_channel=False)

    checkpoint_path = './Pointnet_Pointnet2_pytorch/log/classification/pointnet2_msg_normals/checkpoints/best_model.pth'

    # load on cpu so it works without a gpu
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
    pretrained_weights = checkpoint['model_state_dict']

    # only keep weights that match in name and shape
    model_weights = model.state_dict()
    matching_weights = {
        k: v for k, v in pretrained_weights.items()
        if k in model_weights and model_weights[k].shape == v.shape
    }

    model_weights.update(matching_weights)
    model.load_state_dict(model_weights)

    model.eval()
    return model
