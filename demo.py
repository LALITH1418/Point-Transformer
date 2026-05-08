import os
import sys
import math
import argparse
import numpy as np
import torch
import open3d as o3d
import matplotlib.pyplot as plt

from Networks.network import ObjectDetectionModel


# constants
NUM_CLASSES = 3
CLASS_NAMES = ['vehicle', 'pedestrian', 'cyclist']
NUM_POINTS = 5000
BBOX_NORM = np.array([5.0, 3.0, 10.0, 40.0, 3.0, 70.0, math.pi], dtype=np.float32)

KITTI_CLASS_MAP = {
    'Car': 0, 'Van': 0, 'Truck': 0,
    'Pedestrian': 1, 'Person_sitting': 1,
    'Cyclist': 2,
}

COLOR_PRED = [1.0, 0.15, 0.15]  # red
COLOR_GT = [0.1, 0.85, 0.1]     # green

BOX_EDGES = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]]


# read kitti calibration file
def load_calib(cal_file):
    cal = {}
    with open(cal_file) as f:
        for line in f:
            if line.strip():
                key, *vals = line.split()
                cal[key.rstrip(':')] = np.array(vals, dtype=np.float32)
    return cal


# camera frame -> velodyne frame
def cam_to_velo(cal):
    T = cal['Tr_velo_to_cam'].reshape(3, 4)
    T4 = np.vstack([T, [0, 0, 0, 1]])
    return np.linalg.inv(T4)


# velodyne -> rectified camera
def rect_transform(cal):
    Tr = cal['Tr_velo_to_cam'].reshape(3, 4)
    R0 = cal.get('R0_rect', np.eye(3, dtype=np.float32).ravel()).reshape(3, 3)
    return R0 @ Tr[:, :3], R0 @ Tr[:, 3]


def load_raw_pts(bin_file):
    return np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)


# transform to camera frame, drop points behind camera, sample to fixed size
def sample_camera_frame(pts_raw, R, t, num_points=NUM_POINTS):
    xyz = pts_raw[:, :3] @ R.T + t
    xyz = xyz[xyz[:, 2] > 0]
    n = xyz.shape[0]
    if n == 0:
        return np.zeros((num_points, 3), dtype=np.float32)
    idx = np.random.choice(n, num_points, replace=(n < num_points))
    return xyz[idx].astype(np.float32)


# parse all valid objects from label file
def parse_gt_labels(lbl_file):
    objs = []
    with open(lbl_file) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 15:
                continue
            cls_name = parts[0]
            if cls_name not in KITTI_CLASS_MAP:
                continue
            try:
                _, _, _, _, _, _, _, h, w, l, x, y, z, ry = map(float, parts[1:15])
            except ValueError:
                continue
            if h <= 0 or w <= 0 or l <= 0:
                continue
            objs.append((KITTI_CLASS_MAP[cls_name], h, w, l, x, y, z, ry))
    return objs


# 8 corners of a 3d box in camera frame
def box_corners_cam(h, w, l, x, y, z, ry):
    corners = np.array([
        [ l/2,  h/2,  w/2], [ l/2,  h/2, -w/2],
        [-l/2,  h/2, -w/2], [-l/2,  h/2,  w/2],
        [ l/2, -h/2,  w/2], [ l/2, -h/2, -w/2],
        [-l/2, -h/2, -w/2], [-l/2, -h/2,  w/2],
    ])
    R = np.array([
        [ np.cos(ry), 0, np.sin(ry)],
        [          0, 1,          0],
        [-np.sin(ry), 0, np.cos(ry)],
    ])
    return (R @ corners.T).T + np.array([x, y, z])


# draw boxes on the camera image
def show_image(img_file, cal, pred_box, gt_objects, frame_id):
    img = plt.imread(img_file)
    P2 = cal.get('P2', cal.get('P0')).reshape(3, 4)

    fig, ax = plt.subplots(figsize=(13, 4))
    fig.canvas.manager.set_window_title(f"Frame {frame_id} — camera image")
    ax.imshow(img)
    ax.set_title("red = prediction     green = ground truth", fontsize=9)
    ax.axis('off')

    def draw_box_2d(h, w, l, x, y, z, ry, color):
        corners = box_corners_cam(h, w, l, x, y, z, ry)
        proj = P2 @ np.hstack([corners, np.ones((8, 1))]).T
        depths = proj[2]
        uv = (proj[:2] / np.maximum(depths, 1e-6)).T
        for i, j in BOX_EDGES:
            if depths[i] > 0 and depths[j] > 0:
                ax.plot([uv[i, 0], uv[j, 0]], [uv[i, 1], uv[j, 1]],
                        color=color, lw=1.5)

    h, w, l, x, y, z_c, ry = pred_box
    draw_box_2d(h, w, l, x, y, z_c, ry, 'red')
    if gt_objects:
        for (_, gh, gw, gl, gx, gy, gz, gry) in gt_objects:
            draw_box_2d(gh, gw, gl, gx, gy, gz, gry, 'lime')

    fig.tight_layout()
    plt.pause(0.001)
    return fig


# build a 3d box wireframe for open3d
def bbox_lineset(h, w, l, x, y, z, ry, T_c2v, color):
    corners = np.array([
        [ l/2,  h/2,  w/2], [ l/2,  h/2, -w/2],
        [-l/2,  h/2, -w/2], [-l/2,  h/2,  w/2],
        [ l/2, -h/2,  w/2], [ l/2, -h/2, -w/2],
        [-l/2, -h/2, -w/2], [-l/2, -h/2,  w/2],
    ]).T
    R = np.array([
        [ np.cos(ry), 0, np.sin(ry)],
        [          0, 1,          0],
        [-np.sin(ry), 0, np.cos(ry)],
    ])
    corners = T_c2v @ np.vstack([R @ corners + np.array([[x], [y], [z]]),
                                  np.ones((1, 8))])
    corners = corners[:3].T

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(corners)
    ls.lines = o3d.utility.Vector2iVector(BOX_EDGES)
    ls.paint_uniform_color(color)
    return ls


# show the point cloud + boxes in 3d
def show_frame(pts_raw, pred_box, pred_cls, conf, T_c2v, gt_objects, frame_id):
    # only forward-facing points
    fwd = pts_raw[pts_raw[:, 0] > 0]

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(fwd[:, :3])

    # color by depth
    z = fwd[:, 2]
    z_norm = np.clip((z - z.min()) / (z.max() - z.min() + 1e-6), 0, 1)
    cloud.colors = o3d.utility.Vector3dVector(plt.cm.viridis(z_norm)[:, :3])

    geoms = [cloud]

    h, w, l, x, y, z_c, ry = pred_box
    geoms.append(bbox_lineset(h, w, l, x, y, z_c, ry, T_c2v, COLOR_PRED))

    if gt_objects:
        for (_, gh, gw, gl, gx, gy, gz, gry) in gt_objects:
            geoms.append(bbox_lineset(gh, gw, gl, gx, gy, gz, gry, T_c2v, COLOR_GT))

    # center the view on the prediction
    center_velo = (T_c2v @ np.array([x, y, z_c, 1.0]))[:3]

    has_gt = bool(gt_objects)
    legend = "RED = prediction"
    legend += "   GREEN = ground truth" if has_gt else "   (no labels for this split)"
    title = f"Frame {frame_id} | pred: {CLASS_NAMES[pred_cls]} ({conf:.0%}) | {legend}"

    o3d.visualization.draw_geometries(
        geoms,
        window_name=title,
        width=1280, height=720,
        zoom=0.25,
        front=[-1.0, 0.0, -0.4],
        lookat=center_velo.tolist(),
        up=[0.0, 0.0, 1.0],
    )


# run model on one frame
@torch.no_grad()
def infer(model, pc_cam, device):
    x = torch.from_numpy(pc_cam).unsqueeze(0).to(device)
    cls_logits, bbox_pred = model(x)

    cls_id = int(cls_logits.argmax(dim=1).item())
    conf = float(torch.softmax(cls_logits, dim=1)[0, cls_id])

    # convert sin/cos back to angle
    sin_ry = bbox_pred[0, 6].item()
    cos_ry = bbox_pred[0, 7].item()
    ry_norm = math.atan2(sin_ry, cos_ry) / math.pi
    box7 = np.concatenate([bbox_pred[0, :6].cpu().numpy(), [ry_norm]])
    box7 *= BBOX_NORM

    return cls_id, conf, box7


def main():
    ap = argparse.ArgumentParser(description="Point-Transformer 3D detection demo")
    ap.add_argument('--checkpoint', default='best_object_detection_model_run6_200epochs.pth')
    ap.add_argument('--data_root', default='kitti_3d_object_detection_dataset')
    ap.add_argument('--split', default='training', help='training / test / testing')
    ap.add_argument('--start', type=int, default=0, help='starting frame index')
    args = ap.parse_args()

    velodyne_dir = os.path.join(args.data_root, args.split, 'velodyne')
    calib_dir = os.path.join(args.data_root, args.split, 'calib')
    label_dir = os.path.join(args.data_root, args.split, 'label_2')
    image_dir = os.path.join(args.data_root, args.split, 'image_2')
    has_labels = os.path.isdir(label_dir)
    has_images = os.path.isdir(image_dir)

    if not os.path.isdir(velodyne_dir):
        print(f"ERROR: velodyne directory not found: {velodyne_dir}")
        sys.exit(1)

    frame_ids = sorted(f[:-4] for f in os.listdir(velodyne_dir) if f.endswith('.bin'))
    if not frame_ids:
        print(f"No .bin files found in {velodyne_dir}")
        sys.exit(1)
    print(f"Found {len(frame_ids)} frames  |  labels: {'yes' if has_labels else 'no'}  |  images: {'yes' if has_images else 'no'}")

    # keep matplotlib non-blocking so it doesn't freeze the open3d window
    plt.ion()

    # load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  |  Loading: {args.checkpoint}")
    model = ObjectDetectionModel(num_classes=NUM_CLASSES, feature_dim=64).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    print("Model ready.\n")

    idx = max(0, min(args.start, len(frame_ids) - 1))

    while True:
        fid = frame_ids[idx]
        bin_file = os.path.join(velodyne_dir, fid + '.bin')
        cal_file = os.path.join(calib_dir, fid + '.txt')

        cal = load_calib(cal_file)
        T_c2v = cam_to_velo(cal)
        R, t = rect_transform(cal)
        pts_raw = load_raw_pts(bin_file)
        pc_cam = sample_camera_frame(pts_raw, R, t)

        cls_id, conf, box7 = infer(model, pc_cam, device)

        gt_objects = None
        if has_labels:
            lbl_file = os.path.join(label_dir, fid + '.txt')
            if os.path.exists(lbl_file):
                gt_objects = parse_gt_labels(lbl_file)

        # print prediction info
        h, w, l, x, y, z_c, ry = box7
        print(f"[{idx+1:>4}/{len(frame_ids)}] frame {fid}")
        print(f"  pred : {CLASS_NAMES[cls_id]}  (conf {conf:.1%})")
        print(f"  box  : h={h:.2f}  w={w:.2f}  l={l:.2f}  "
              f"x={x:.2f}  y={y:.2f}  z={z_c:.2f}  ry={ry:.2f} rad")
        if gt_objects:
            gt_summary = "  gt   : " + "  |  ".join(
                f"{CLASS_NAMES[c]}  h={gh:.2f} w={gw:.2f} l={gl:.2f} "
                f"x={gx:.2f} y={gy:.2f} z={gz:.2f} ry={gry:.2f}"
                for c, gh, gw, gl, gx, gy, gz, gry in gt_objects
            )
            print(gt_summary)
        print("  → Close the Open3D window, then type a command below.")

        # show image first, then 3d view
        img_fig = None
        if has_images:
            img_file = os.path.join(image_dir, fid + '.png')
            if os.path.exists(img_file):
                img_fig = show_image(img_file, cal, box7, gt_objects, fid)

        show_frame(pts_raw, box7, cls_id, conf, T_c2v, gt_objects, fid)

        if img_fig is not None:
            plt.close(img_fig)

        # handle user input
        cmd = input("  n=next  p=prev  q=quit  or frame index: ").strip().lower()
        if cmd == 'q':
            print("Bye.")
            break
        elif cmd == 'p':
            idx = max(0, idx - 1)
        elif cmd.isdigit():
            idx = min(int(cmd), len(frame_ids) - 1)
        else:
            idx = min(idx + 1, len(frame_ids) - 1)


if __name__ == '__main__':
    main()
