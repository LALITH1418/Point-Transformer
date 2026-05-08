import os
import urllib.request
from zipfile import ZipFile


urls = {
    "images": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip",
    "calibration": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip",
    "labels": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip",
    "lidar": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip",
}

save_folder = "kitti_3d_object_detection_dataset"
os.makedirs(save_folder, exist_ok=True)

def fetch_and_unzip(url, folder):
    fname = url.split("/")[-1]
    zip_path = os.path.join(folder, fname)

    if not os.path.exists(zip_path):
        print(f"Downloading {fname}...")
        urllib.request.urlretrieve(url, zip_path)
        print(f"Downloaded {fname}")
    else:
        print(f"{fname} already downloaded.")

    print(f"Extracting {fname}...")
    with ZipFile(zip_path, 'r') as zf:
        zf.extractall(folder)
    print(f"Extracted {fname}")

for name, url in urls.items():
    fetch_and_unzip(url, save_folder)

print("KITTI dataset downloaded and extracted successfully.")
