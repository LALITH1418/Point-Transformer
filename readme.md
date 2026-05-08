# Custom Transformer model for 3d object detection paired with pointnet++ for feature extraction

This project focuses on 3D object detection using LiDAR data from the KITTI dataset. It includes data preprocessing, model training, and visualization of results.

## Model

The model are located in the `Networks` folder. The model is implemented in `network.py`.

## Training

The training script is located in `training.py`. This script handles both the training and validation of the model using the KITTI dataset. 

### Training the Model

1. Ensure you have the required dependencies installed.
2. Download the KITTI dataset using the script `download_kitti_dataset.py`.
3. Ensure Pointnet_pointnet2_pytorch is available in the project root as this file as the code files for pointnet++ and the frozen checkpoint we are using for this project.
3. Train the model by running the following command:

   ```bash
   python training.py

Our training is done on RTX 5090 32 GB vram to speed up the training time. 

Our final training checkpoint is saved as best_obje