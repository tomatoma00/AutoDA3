import glob
import os

import numpy as np
import torch
import trimesh

from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.export.glb import (
    _compute_alignment_transform_first_cam_glTF_center_by_points,
    _depths_to_world_points_with_colors,
    get_conf_thresh,
)


def downsample_points(
    points: np.ndarray,
    colors: np.ndarray,
    sample_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    colors = colors[finite]

    if 0 < sample_ratio < 1.0 and points.shape[0] > 0:
        num_samples = int(points.shape[0] * sample_ratio)
        num_samples = max(1, num_samples)
        idx = np.random.choice(points.shape[0], num_samples, replace=False)
        points = points[idx]
        colors = colors[idx]

    return points, colors


def load_images_and_poses(data_dir: str) -> tuple[list[str], np.ndarray]:
    """Load frame-*.color.png and corresponding frame-*.pose.txt as (N, 4, 4)."""
    image_pattern = os.path.join(data_dir, "frame-*.color.png")
    image_paths = sorted(glob.glob(image_pattern))

    if not image_paths:
        raise FileNotFoundError(f"No images found with pattern: {image_pattern}")

    poses = []
    for image_path in image_paths:
        image_name = os.path.basename(image_path)
        # frame-000000.color.png -> frame-000000.pose.txt
        pose_name = image_name.replace(".color.png", ".pose.txt")
        pose_path = os.path.join(data_dir, pose_name)

        if not os.path.exists(pose_path):
            raise FileNotFoundError(f"Missing pose file for image {image_name}: {pose_path}")

        pose = np.loadtxt(pose_path, dtype=np.float32)
        if pose.shape != (4, 4):
            raise ValueError(
                f"Pose file {pose_path} has shape {pose.shape}, expected (4, 4)"
            )

        poses.append(pose)

    poses_np = np.stack(poses, axis=0)
    return image_paths, poses_np


def convert_poses_to_w2c(poses: np.ndarray, pose_format: str) -> np.ndarray:
    if pose_format == "w2c":
        return poses.astype(np.float32, copy=False)
    if pose_format == "c2w":
        return np.linalg.inv(poses).astype(np.float32)
    raise ValueError(f"Unsupported pose_format: {pose_format}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthAnything3.from_pretrained("da3ckpt", local_files_only=True)
    model = model.to(device=device)

    # example_path = "/amax/shenchentao/ffwdgs/Depth-Anything-3/assets/examples/seq"
    example_path = "/amax/shenchentao/ffwdgs/Depth-Anything-3/datawithgt/redkitchen"
    output_ply = os.path.join(example_path, "stitched_point_cloud.ply")
    sample_ratio = 0.1
    black_bg_threshold = 16
    pose_format = "c2w"

    images, gt_poses = load_images_and_poses(example_path)
    gt_w2c = convert_poses_to_w2c(gt_poses, pose_format=pose_format)
    print(f"Loaded {len(images)} images")
    print(f"GT poses shape: {gt_poses.shape}")
    print(f"Using pose_format={pose_format}, inference extrinsics shape: {gt_w2c.shape}")
    K = np.array(
        [
            [525.0, 0.0, 319.5],
            [0.0, 525.0, 239.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    K = np.repeat(K[None, :, :], len(images), axis=0)
    prediction = model.inference(images, extrinsics=gt_w2c, intrinsics=K)

    # prediction.processed_images : [N, H, W, 3] uint8 array
    print(prediction.processed_images.shape)
    # prediction.depth            : [N, H, W] float32 array
    print(prediction.depth.shape)
    # prediction.conf             : [N, H, W] float32 array
    print(prediction.conf.shape)
    # prediction.extrinsics       : [N, 3, 4] float32 array (opencv/colmap w2c)
    print(prediction.extrinsics.shape)
    # prediction.intrinsics       : [N, 3, 3] float32 array
    print(prediction.intrinsics.shape)

    if prediction.conf is None:
        raise ValueError("prediction.conf is None, cannot generate robust stitched point cloud")

    conf_for_threshold = prediction.conf.copy()
    conf_for_threshold[(prediction.processed_images < black_bg_threshold).all(axis=-1)] = 1.0

    class _Pred:
        pass

    pred_stub = _Pred()
    pred_stub.conf = conf_for_threshold
    conf_threshold = get_conf_thresh(
        pred_stub,
        sky_mask=None,
        conf_thresh=1.05,
        conf_thresh_percentile=40.0,
        ensure_thresh_percentile=90.0,
    )
    print(f"Confidence threshold: {conf_threshold:.4f}")

    points_world, colors = _depths_to_world_points_with_colors(
        prediction.depth,
        prediction.intrinsics,
        prediction.extrinsics,
        prediction.processed_images,
        prediction.conf,
        conf_threshold,
    )

    # Force-remove black background points from exported cloud.
    non_black = ~(colors < black_bg_threshold).all(axis=-1)
    points_world = points_world[non_black]
    colors = colors[non_black]

    global_align = _compute_alignment_transform_first_cam_glTF_center_by_points(
        prediction.extrinsics[0], points_world
    )
    points_world = trimesh.transform_points(points_world, global_align)

    points_world, colors = downsample_points(points_world, colors, sample_ratio=sample_ratio)

    os.makedirs(os.path.dirname(output_ply), exist_ok=True)
    trimesh.PointCloud(points_world, colors=colors).export(output_ply)
    print(f"Exported stitched point cloud: {output_ply}")
    print(f"Point count: {points_world.shape[0]}")


if __name__ == "__main__":
    main()