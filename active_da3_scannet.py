import os
import torch
import glob
import numpy as np
import math
from dataclasses import dataclass
from typing import Dict, List
from PIL import Image
from utils.loss_utils import l1_loss, ssim
import sys
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from utils.loss_utils import ssim
# from lpipsPyTorch import lpips, lpips_func
from resnet_train import ResNet50Regressor
from publicfunction import qualitycheck, robust_normalize_gpu
from torchvision.utils import save_image
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from depth_anything_3.api import DepthAnything3
from depth_anything_3.model.utils.gs_renderer import run_renderer_in_chunk_w_trj_mode
from depth_anything_3.utils.export.gs import export_to_gs_ply
from depth_anything_3.utils.export.glb import (
    _compute_alignment_transform_first_cam_glTF_center_by_points,
    _depths_to_world_points_with_colors,
    get_conf_thresh,
)
import random
random.seed(0)
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = False
except ImportError:
    TENSORBOARD_FOUND = False
from utils.cluster_manager import ClusterStateManager

csm = ClusterStateManager()


@dataclass
class SimpleCamera:
    uid: int
    image_name: str
    image_path: str
    image_width: int
    image_height: int
    K: np.ndarray
    E: np.ndarray
    inference_image: np.ndarray
    original_image: torch.Tensor
    camera_center: torch.Tensor

def _bytes_to_gb(num_bytes: int) -> float:
    return float(num_bytes) / (1024.0 ** 3)


def report_cuda_memory(iteration: int):
    if not torch.cuda.is_available():
        print(f"[ITER {iteration}] CUDA unavailable, skip VRAM report")
        return

    current_alloc = torch.cuda.memory_allocated(device)
    peak_alloc = torch.cuda.max_memory_allocated(device)
    print(
        f"[ITER {iteration}] VRAM current={_bytes_to_gb(current_alloc):.3f} GB, "
        f"peak={_bytes_to_gb(peak_alloc):.3f} GB"
    )

class ActiveSceneLite:
    def __init__(self, cameras: List[SimpleCamera], model_path: str, init_trainidx:list):
        self.model_path = model_path
        self.train_cameras = {1.0: cameras}
        self.test_cameras = {1.0: []}
        self.all_train_set = set(range(len(cameras)))
        if init_trainidx is None:
            self.train_idxs = list(range(0,len(cameras),8))
        else:
            self.train_idxs = init_trainidx
        self.candidate_views_filter = None

    def getTrainCameras(self, scale=1.0):
        return [self.train_cameras[scale][i] for i in self.train_idxs]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def get_candidate_set(self):
        candidate_set = sorted(list(self.all_train_set - set(self.train_idxs)))
        if self.candidate_views_filter is not None:
            candidate_set = list(filter(self.candidate_views_filter, candidate_set))
        return candidate_set

    def getCandidateCameras(self, scale=1.0):
        return [self.train_cameras[scale][i] for i in self.get_candidate_set()]


def _qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = qvec
    return np.array(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qw * qz, 2 * qx * qz + 2 * qw * qy],
            [2 * qx * qy + 2 * qw * qz, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qw * qx],
            [2 * qx * qz - 2 * qw * qy, 2 * qy * qz + 2 * qw * qx, 1 - 2 * qx * qx - 2 * qy * qy],
        ],
        dtype=np.float32,
    )


def _load_image_tensor(image_path: str) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def convert_poses_to_w2c(
    poses: np.ndarray,
    pose_format: str,
    camera_coord_system: str = "opencv",
) -> np.ndarray:
    if pose_format == "w2c":
        w2c = poses.astype(np.float32, copy=False)
    elif pose_format == "c2w":
        w2c = np.linalg.inv(poses).astype(np.float32)
    else:
        raise ValueError(f"Unsupported pose_format: {pose_format}")

    if camera_coord_system == "opencv":
        return w2c
    if camera_coord_system == "opengl":
        flip = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
        return (flip[None, :, :] @ w2c).astype(np.float32)
    raise ValueError(f"Unsupported camera_coord_system: {camera_coord_system}")


def preprocess_single_image_and_k(
    image_path: str,
    k: np.ndarray,
    target_width: int = 504,
) -> tuple[np.ndarray, np.ndarray]:
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    scale = target_width / float(orig_w)
    target_height = max(1, int(round(orig_h * scale)))
    if orig_w != target_width:
        interp = Image.Resampling.LANCZOS if scale < 1.0 else Image.Resampling.BICUBIC
        img = img.resize((target_width, target_height), interp)

    img_np = np.asarray(img, dtype=np.uint8)
    k_new = k.astype(np.float32, copy=True)
    k_new[0, 0] *= scale
    k_new[1, 1] *= scale
    k_new[0, 2] *= scale
    k_new[1, 2] *= scale
    return img_np, k_new


def normalize_w2c_like_da3(extrinsics: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    if extrinsics.ndim != 3 or extrinsics.shape[1:] != (4, 4):
        raise ValueError(f"extrinsics must be (N, 4, 4), got {extrinsics.shape}")
    ex = extrinsics.astype(np.float32, copy=True)
    transform = np.linalg.inv(ex[:1])
    ex_norm = ex @ transform
    c2w = np.linalg.inv(ex_norm)
    translations = c2w[:, :3, 3]
    dists = np.linalg.norm(translations, axis=-1)
    median_dist = max(float(np.median(dists)), 1e-1)
    ex_norm[:, :3, 3] /= max(median_dist, eps)
    return ex_norm.astype(np.float32)


def get_da3_normalization_params(extrinsics: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, float]:
    if extrinsics.ndim != 3 or extrinsics.shape[1:] != (4, 4):
        raise ValueError(f"extrinsics must be (N, 4, 4), got {extrinsics.shape}")
    ex = extrinsics.astype(np.float32, copy=False)
    transform = np.linalg.inv(ex[0]).astype(np.float32)
    ex_norm = ex @ transform[None, :, :]
    c2w = np.linalg.inv(ex_norm)
    translations = c2w[:, :3, 3]
    dists = np.linalg.norm(translations, axis=-1)
    median_dist = max(float(np.median(dists)), 1e-1)
    median_dist = max(median_dist, eps)
    return transform, median_dist


def apply_da3_normalization(extrinsics: np.ndarray, transform: np.ndarray, median_dist: float) -> np.ndarray:
    if extrinsics.ndim != 3 or extrinsics.shape[1:] != (4, 4):
        raise ValueError(f"extrinsics must be (N, 4, 4), got {extrinsics.shape}")
    ex = extrinsics.astype(np.float32, copy=True)
    ex = ex @ transform[None, :, :]
    ex[:, :3, 3] /= float(median_dist)
    return ex

def load_images_and_poses(data_dir: str) -> tuple[list[str], np.ndarray]:
    """Load frame-*.color.png and corresponding frame-*.pose.txt as (N, 4, 4)."""
    import glob

    image_pattern = os.path.join(data_dir, "*.jpg")
    image_paths = sorted(glob.glob(image_pattern))

    if not image_paths:
        raise FileNotFoundError(f"No images found with pattern: {image_pattern}")

    poses = []
    for image_path in image_paths:
        image_name = os.path.basename(image_path)
        pose_name = image_name.replace(".jpg", ".txt")
        pose_path = os.path.join(data_dir, pose_name)

        if not os.path.exists(pose_path):
            raise FileNotFoundError(f"Missing pose file for image {image_name}: {pose_path}")

        pose = np.loadtxt(pose_path, dtype=np.float32)
        if pose.shape != (4, 4):
            raise ValueError(f"Pose file {pose_path} has shape {pose.shape}, expected (4, 4)")

        poses.append(pose)

    poses_np = np.stack(poses, axis=0)
    return image_paths, poses_np

def build_active_scene_lite(dataset, model_path: str, init_trainidx:list) -> ActiveSceneLite:
    datapath = dataset.source_path
    image_paths, poses_np = load_images_and_poses(datapath)
    poses_np = convert_poses_to_w2c(poses_np, pose_format="c2w", camera_coord_system="opencv")
    cameras=[]
    base_k = np.array(
        [
            [1165.723022, 0.0, 649.094971],
            [0.0, 1165.738037, 484.765015],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    for i in range(len(image_paths)):
        image_np, K = preprocess_single_image_and_k(image_paths[i], base_k, target_width=504)
        image_tensor = torch.from_numpy(image_np.astype(np.float32) / 255.0).permute(2, 0, 1)
        image_h = int(image_np.shape[0])
        image_w = int(image_np.shape[1])
        cameras.append(
            SimpleCamera(
                uid=i,
                image_name=os.path.splitext(os.path.basename(image_paths[i]))[0],
                image_path=image_paths[i],
                image_width=image_w,
                image_height=image_h,
                K = K,
                E  = poses_np[i],
                inference_image=image_np,
                original_image=image_tensor,
                camera_center=None,
            )
        )
    return ActiveSceneLite(cameras=cameras, model_path=model_path,init_trainidx=init_trainidx)

def downsample_points(points: np.ndarray, colors: np.ndarray, sample_ratio: float):
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    colors = colors[finite]

    if 0 < sample_ratio < 1.0 and points.shape[0] > 0:
        num_samples = max(1, int(points.shape[0] * sample_ratio))
        idx = np.random.choice(points.shape[0], num_samples, replace=False)
        points = points[idx]
        colors = colors[idx]

    return points, colors


def resolve_image_path(source_path: str, image_name: str):
    image_root = source_path
    candidates = sorted(glob.glob(os.path.join(image_root, f"{image_name}.*")))
    if len(candidates) == 0:
        return None
    return candidates[0]


def camera_to_w2c_and_k(cam):
    w2c = cam.E.astype(np.float32)
    k = cam.K.astype(np.float32)
    return w2c, k


@torch.no_grad()
def build_da3_prediction(scene, dataset, da3_model, selected_idxs):
    selected_cams = [scene.train_cameras[1.0][i] for i in selected_idxs]

    images = []
    w2c_list = []
    k_list = []
    for cam in selected_cams:
        w2c, k = camera_to_w2c_and_k(cam)
        images.append(cam.inference_image)
        w2c_list.append(w2c)
        k_list.append(k)

    w2c_np = np.stack(w2c_list, axis=0)
    k_np = np.stack(k_list, axis=0)
    da3_transform, da3_median_dist = get_da3_normalization_params(w2c_np)

    prediction = da3_model.inference(
        images,
        extrinsics=w2c_np,
        intrinsics=k_np,
        infer_gs=True,
        ref_view_strategy="first",
    )

    if prediction.gaussians is None:
        raise RuntimeError("DA3 inference did not produce Gaussian parameters. Please check DA3 checkpoint and inputs.")
    return prediction, da3_transform, da3_median_dist


@torch.no_grad()
def render_prediction_batch(prediction, cams, da3_transform, da3_median_dist, chunk_size=4,trj_mode = "original"):
    if len(cams) == 0:
        return None, None

    h = int(cams[0].image_height)
    w = int(cams[0].image_width)
    for cam in cams:
        if int(cam.image_height) != h or int(cam.image_width) != w:
            raise ValueError("All cameras in one render batch must have the same image shape.")

    w2c_list, k_list = [], []
    for cam in cams:
        w2c, k = camera_to_w2c_and_k(cam)
        w2c_list.append(w2c)  # Use full 4x4 matrix, same as in build_da3_prediction
        k_list.append(k)

    w2c_np = np.stack(w2c_list, axis=0)
    w2c_np = apply_da3_normalization(w2c_np, da3_transform, da3_median_dist)

    gs_world = prediction.gaussians
    tgt_extrs = torch.from_numpy(w2c_np).unsqueeze(0).to(gs_world.means)
    tgt_intrs = torch.from_numpy(np.stack(k_list, axis=0)).unsqueeze(0).to(gs_world.means)

    color, depth = run_renderer_in_chunk_w_trj_mode(
        gaussians=gs_world,
        extrinsics=tgt_extrs,
        intrinsics=tgt_intrs,
        image_shape=(h, w),
        chunk_size=chunk_size,
        trj_mode=trj_mode,
        use_sh=True,
        color_mode="RGB+ED",
        enable_tqdm=False,
    )
    
    # Return [V, 3, H, W] and [V, H, W]
    return color[0], depth[0]


@torch.no_grad()
def evaluate_ff_da3(scene, prediction, da3_transform, da3_median_dist, iteration, tb_writer=None, log_every_image=False):
    torch.cuda.empty_cache()
    # lpips_metric = lpips_func("cuda", net_type="vgg")
    eval_rgb_root = os.path.join(scene.model_path, "eval_rgb", f"iter_{iteration:06d}")

    train_cams = scene.getTrainCameras()
    test_cams = scene.getTestCameras()
    sampled_train = [train_cams[idx % len(train_cams)] for idx in range(5, 30, 5)] if len(train_cams) > 0 else []

    validation_configs = (
        {"name": "test", "cameras": test_cams},
        {"name": "train", "cameras": sampled_train},
    )

    for config in validation_configs:
        cameras = config["cameras"]
        if not cameras:
            continue
        split_save_dir = os.path.join(eval_rgb_root, config["name"])
        os.makedirs(split_save_dir, exist_ok=True)

        l1_test = 0.0
        psnr_test = 0.0
        ssim_test = 0.0
        # lpips_test = 0.0

        
        colors, _ = render_prediction_batch(
            prediction,
            [viewpoint for idx,viewpoint in enumerate(cameras)],
            da3_transform,
            da3_median_dist,
            chunk_size=4,
        )
        for idx, viewpoint in enumerate(cameras):    
            image = torch.clamp(colors[idx], 0.0, 1.0)
            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

            render_name = f"{idx:03d}_{viewpoint.image_name}.png"
            save_image(image, os.path.join(split_save_dir, render_name))

            if tb_writer and ((idx < 5) or log_every_image):
                tb_writer.add_images(config["name"] + "_view_{}/render".format(idx), image[None], global_step=iteration)
                tb_writer.add_images(config["name"] + "_view_{}/ground_truth".format(idx), gt_image[None], global_step=iteration)

            l1_test += l1_loss(image, gt_image).mean().double()
            psnr_test += psnr(image, gt_image).mean().double()
            ssim_test += ssim(image, gt_image).mean().double()
            # lpips_metric.to(image.device)
            # lpips_test += lpips_metric(image, gt_image).mean().double()

        denom = len(cameras)
        l1_test /= denom
        psnr_test /= denom
        ssim_test /= denom
        # lpips_test /= denom

        print(
            "\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {}".format(
                iteration, config["name"], l1_test, psnr_test, ssim_test, 0.0
            )
        )

        if tb_writer:
            tb_writer.add_scalar(config["name"] + "/loss_viewpoint - l1_loss", l1_test, iteration)
            tb_writer.add_scalar(config["name"] + "/loss_viewpoint - psnr", psnr_test, iteration)
            tb_writer.add_scalar(config["name"] + "/loss_viewpoint - ssim", ssim_test, iteration)
            # tb_writer.add_scalar(config["name"] + "/loss_viewpoint - lpips", lpips_test, iteration)

    torch.cuda.empty_cache()


@torch.no_grad()
def save_da3_prediction_ply(prediction, model_path, iteration):
    export_root = os.path.join(model_path, f"ff_da3_iter_{iteration:06d}")
    os.makedirs(export_root, exist_ok=True)
    export_to_gs_ply(prediction, export_dir=export_root, gs_views_interval=1)
    print(f"Saved DA3 Gaussian PLY to {os.path.join(export_root, 'gs_ply')}")


def training(dataset,testing_iterations, saving_iterations, args):
    # init
    tb_writer = prepare_output_and_logger(dataset)
    scene = build_active_scene_lite(dataset, args.model_path,init_trainidx=[0,15,30,45,60])

    uqmodel = ResNet50Regressor(pretrained=True).to(device)
    uqmodel.eval()
    state_dict = torch.load('ssimruns/best1473.pth', map_location="cpu")
    uqmodel.load_state_dict(state_dict)
    da3_model = DepthAnything3.from_pretrained("/amax/shenchentao/ffwdgs/Depth-Anything-3/da3ckpt", local_files_only=True)
    da3_model = da3_model.to(device=device)
    da3_model.eval()

    # Active View Selection
    print(f"init cam:,",[cam.image_name for cam in scene.getTrainCameras()])
    # Build initial feed-forward reconstruction from initial views.
    prediction, da3_transform, da3_median_dist = build_da3_prediction(scene, dataset, da3_model, scene.train_idxs)

    if 0 in testing_iterations:
        evaluate_ff_da3(
            scene,
            prediction,
            da3_transform,
            da3_median_dist,
            iteration=0,
            tb_writer=tb_writer,
            log_every_image=args.log_every_image,
        )

    progress_bar = tqdm(range(1, 11), desc="FF-DA3 active reconstruction")
    for iteration in progress_bar:
        if iteration in testing_iterations:
            evaluate_ff_da3(
                scene,
                prediction,
                da3_transform,
                da3_median_dist,
                iteration=iteration,
                tb_writer=tb_writer,
                log_every_image=args.log_every_image,
            )
        if iteration in saving_iterations:
            save_da3_prediction_ply(prediction, args.model_path, iteration)

        print(f"\n[ITER {iteration}] Selecting a new view with DA3 forward rendering")

        candidate_indices = scene.get_candidate_set()
        candidate_cams = scene.getCandidateCameras()
        if len(candidate_cams) == 0:
            print("No candidate cameras left. Stop view selection.")
            break

        outpath = f"iteration{iteration}"
        uqsavedir = f"{args.model_path}/{outpath}/uqs"
        os.makedirs(uqsavedir, exist_ok=True)

        B = 16
        all_best_score = -1.0
        all_best_global_idx = -1

        for batchidx in range(0, len(candidate_cams), B):
            batch_cams = candidate_cams[batchidx:batchidx + B]
            if len(batch_cams)==1:
                batch_color, batch_depth = render_prediction_batch(
                    prediction,
                    batch_cams,
                    da3_transform,
                    da3_median_dist,
                    chunk_size=min(4, len(batch_cams)),
                    trj_mode="wander",
                )
            else:
                batch_color, batch_depth = render_prediction_batch(
                    prediction,
                    batch_cams,
                    da3_transform,
                    da3_median_dist,
                    chunk_size=min(4, len(batch_cams)),
                )

            batch_img = torch.clamp(batch_color, 0.0, 1.0)
            depth_norm = [robust_normalize_gpu(batch_depth[i]) for i in range(batch_depth.shape[0])]
            batch_depth_uq = torch.stack(depth_norm, dim=0).unsqueeze(1)

            b_max, uq_max = qualitycheck(
                uqmodel,
                batch_img,
                batch_depth_uq,
                uqsavedir,
                [f"{c.image_name}.jpg" for c in batch_cams],
                batch_size=B,
                iter=iteration,
            )

            uq_max_value = float(uq_max)
            if uq_max_value > all_best_score:
                all_best_score = uq_max_value
                all_best_global_idx = batchidx + int(b_max)

        chosen_dataset_idx = candidate_indices[all_best_global_idx]
        chosen_cam = scene.train_cameras[1.0][chosen_dataset_idx]
        scene.train_idxs.append(chosen_dataset_idx)

        print(f"[ITER {iteration}] selected view: {chosen_cam.image_name}, idx={chosen_dataset_idx}, score={all_best_score:.6f}")

        with open(f"{args.model_path}/viewnumber.txt", "w") as f:
            for idx in scene.train_idxs:
                f.write(f"{idx}\n")

        # Rebuild Gaussian prediction after adding each selected view.
        prediction, da3_transform, da3_median_dist = build_da3_prediction(scene, dataset, da3_model, scene.train_idxs)

        if iteration in testing_iterations:
            evaluate_ff_da3(
                scene,
                prediction,
                da3_transform,
                da3_median_dist,
                iteration=iteration,
                tb_writer=tb_writer,
                log_every_image=args.log_every_image,
            )

        if iteration in saving_iterations:
            save_da3_prediction_ply(prediction, args.model_path, iteration)

        report_cuda_memory(iteration)

        if csm.should_exit():
            print("Cluster manager requests exit; stopping after saving selected view indices.")
            return

        

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene, renderFunc, renderArgs, before_selection=False, log_every_image=False):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations or before_selection:
        print(f"Running evaluation for iteration: {iteration}")
        torch.cuda.empty_cache()
        #lpips = lpips_func("cuda", net_type='vgg')
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                # lpips_test = 0.0

                # log_images = {}
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and ((idx < 5) or log_every_image):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(idx), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(idx), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()
                    # lpips.to(image.device)
                    # lpips_test += lpips(image, gt_image).mean().double()


                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])

                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {}".format(iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)
                log_dict = {config['name'] + '/l1_loss': l1_test, config['name'] + '/psnr': psnr_test,
                            config['name'] + '/ssim': ssim_test, config['name'] + '/lpips': lpips_test,}

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

import socket
from contextlib import closing

def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]

if __name__ == "__main__":
    # 
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    #lp = ModelParams(parser)
    
    # parser.add_argument('--ip', type=str, default="127.0.0.1")
    # parser.add_argument('--port', type=int, default=6009)
    # parser.add_argument('--debug_from', type=int, default=-1)
    # parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("-source_path",required=True,type=str)
    parser.add_argument("-model_path",required=True,type=str)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[0,5,10,15,20])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[10,20])
    parser.add_argument("--quiet", action="store_true")
    
    # Flags for view selections
    parser.add_argument("--filter_out_grad", nargs="+", type=str, default=["rotation"])
    parser.add_argument("--log_every_image", action="store_true", help="log every images during traing")
    parser.add_argument("--da3_ckpt", type=str, default="da3ckpt", help="DepthAnything3 checkpoint path or HF id")
    parser.add_argument("--da3_local_files_only", action="store_true", help="Only load DA3 weights from local files")
    parser.add_argument("--da3_sample_ratio", type=float, default=0.1, help="Downsample ratio for DA3 stitched points")
    parser.add_argument("--da3_black_bg_threshold", type=int, default=16, help="Black background threshold for DA3 point filtering")

    args = parser.parse_args(sys.argv[1:])
    #safe_state(args.quiet, seed=0)
    #torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(args, [0,5,10,15,20], [10,20],args)

    # All done
    print("\nTraining complete.")

#python active_da3.py -source_path datawithgt/redkitchen/ -model_path datawithgt/redkitchenout