#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from scene.cameras import Camera
import numpy as np
from gaussian_renderer import render
from scipy.spatial.transform import Rotation as RSci
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

def get_camera(existing_cameras, R, T):
    device = existing_cameras[0].data_device
    
    new_camera = Camera(
        colmap_id=existing_cameras[0].colmap_id,
        R=R,
        T=T,
        FoVx=existing_cameras[0].FoVx,
        FoVy=existing_cameras[0].FoVy,
        image=existing_cameras[0].original_image,
        gt_alpha_mask=None,
        image_name=existing_cameras[0].image_name,
        uid=f"new_{np.random.randint(10000)}",
        trans=existing_cameras[0].trans,
        scale=existing_cameras[0].scale, 
        data_device="cuda"
    )

    return new_camera

def generate_new_cameras(existing_cameras, num_new_cameras=5):
    # Placeholder for the new cameras
    new_cameras = []
    device = existing_cameras[0].data_device
    
    # Step 1: Analyze existing cameras to determine bounds for position and orientation
    positions = [cam.camera_center for cam in existing_cameras]
    positions = torch.stack(positions).to(device)
    pos_min, pos_max = positions.min(dim=0)[0], positions.max(dim=0)[0]
    
    for _ in range(num_new_cameras):
        # Step 2: Generate a new position as a torch tensor, then convert to numpy array for consistency
        random_pos = (torch.rand(3).to(device) * (pos_max - pos_min)) + pos_min
        random_pos_numpy = random_pos.cpu().numpy()  # Convert to numpy array
        
        # Generate a random orientation (rotation around the y-axis) as a numpy array
        angle = np.random.uniform(-np.pi, np.pi)
        random_rotation = RSci.from_euler('y', angle).as_matrix()

        # Create a new Camera object with numpy arrays for R and T
        new_camera = Camera(
            colmap_id=existing_cameras[0].colmap_id,
            R=random_rotation,  # This is a numpy array
            T=-np.dot(random_rotation, random_pos_numpy.reshape(3, )),  # Use numpy operations
            FoVx=existing_cameras[0].FoVx,
            FoVy=existing_cameras[0].FoVy,
            image=existing_cameras[0].original_image,
            gt_alpha_mask=None,
            image_name=existing_cameras[0].image_name,
            uid=f"new_{np.random.randint(10000)}",
            trans=existing_cameras[0].trans,
            scale=existing_cameras[0].scale, 
            data_device="cuda"
        )
        
        new_cameras.append(new_camera)
    
    return new_cameras

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, no_gt):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    makedirs(render_path, exist_ok=True)

    if not no_gt:
        gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
        makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        if not no_gt:
            gt = view.original_image[0:3, :, :]
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, custom_cams: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if custom_cams:
            # render only looks at image height, image width, fovx, fovy, world_view_Transform, full_proj_transform, and camera_center
            random_cams = generate_new_cameras(scene.getTrainCameras())
            render_set(dataset.model_path, "random", scene.loaded_iter, random_cams, gaussians, pipeline, background, True)
        else:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, False)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--custom_cams", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.custom_cams)