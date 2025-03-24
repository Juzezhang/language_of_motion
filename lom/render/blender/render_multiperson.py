import math
import os
import sys
import smplx
import bpy
import numpy as np

from .camera import Camera
from .floor import get_trajectory, plot_floor_multiperson, show_traj
from .sampler import get_frameidx
from .scene import setup_scene  # noqa
from .tools import delete_objs, load_numpy_vertices_into_blender, style_detect
from .vertices import prepare_vertices
from lom.utils.joints import smplh_to_mmm_scaling_factor


def prune_begin_end(data, perc):
    to_remove = int(len(data) * perc)
    if to_remove == 0:
        return data
    return data[to_remove:-to_remove]


def render_current_frame(path):
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(use_viewport=True, write_still=True)


def render_multiperson(npydata_list,
                       frames_folder,
                       *,
                       mode,
                       model_path,
                       faces_path,
                       gt=False,
                       exact_frame=None,
                       num=8,
                       downsample=True,
                       canonicalize=True,
                       always_on_floor=False,
                       denoising=True,
                       oldrender=True,
                       res="high",
                       init=True,
                       accelerator='gpu',
                       device=[0]):
    """
    Render multiple npydata inputs in a single Blender scene.

    Args:
        npydata_list (list): List of npydata inputs.
        frames_folder (str): Output folder for rendered frames or sequence.
        Other parameters are similar to the original function.
    """
    if init:
        # Setup the scene (lights / render engine / resolution etc)
        setup_scene(res=res,
                    denoising=denoising,
                    oldrender=oldrender,
                    accelerator=accelerator,
                    device=device)

    # Detect styles for all npydata
    all_data = []
    for npydata in npydata_list:
        is_mesh, is_smplx, jointstype = style_detect(npydata)
        if not is_mesh:
            npydata = npydata * smplh_to_mmm_scaling_factor

        if is_smplx:
            smplx_model_male = smplx.create(model_path,
                                            model_type='smplx',
                                            gender='neutral_2020',
                                            ext='npz',
                                            num_betas=10,
                                            flat_hand_mean=True,
                                            use_pca=False)
            faces_path = smplx_model_male.faces

        if downsample and not is_mesh:
            npydata = npydata[::8]

        if is_mesh:
            from .meshes import Meshes
            data = Meshes(npydata,
                          gt=gt,
                          mode=mode,
                          faces_path=faces_path,
                          canonicalize=canonicalize,
                          always_on_floor=always_on_floor,
                          is_smplx=is_smplx)
        else:
            from .joints import Joints
            data = Joints(npydata,
                          gt=gt,
                          mode=mode,
                          canonicalize=canonicalize,
                          always_on_floor=always_on_floor,
                          jointstype=jointstype)

        all_data.append(data)

    # Combine trajectories and plot floor
    combined_trajectory = np.concatenate([data.trajectory for data in all_data], axis=0)
    show_traj(combined_trajectory)
    plot_floor_multiperson(np.concatenate([data.data for data in all_data], axis=0), big_plane=False)

    # # Initialize the camera
    # camera = Camera(first_root=all_data[0].get_root(0), mode=mode, is_mesh=True)
    # Calculate the center of all individuals
    all_roots = [data.get_root(0) for data in all_data]
    all_roots = np.array(all_roots)
    center_root = np.mean(all_roots, axis=0)  # Calculate the center of all roots

    # Initialize the camera with the calculated center
    # camera = Camera(first_root=center_root, mode=mode, is_mesh=True)



    # Process frames for all npydata
    combined_imported_obj_names = []
    for frameidx in range(len(all_data[0])):
        # Load and render all objects in the current frame
        current_obj_names = []
        for data in all_data:
            mat = data.mat
            objname = data.load_in_blender(frameidx, mat)
            current_obj_names.extend(objname)

        combined_imported_obj_names.extend(current_obj_names)
        name = f"{str(frameidx).zfill(4)}"
        path = os.path.join(frames_folder, f"frame_{name}.png")
        render_current_frame(path)

        # Remove objects for the next frame
        delete_objs(current_obj_names)

    bpy.ops.wm.save_as_mainfile(filepath=frames_folder.replace('.png', '.blend').replace('_frames', '.blend'))

    # Remove all objects created
    delete_objs(combined_imported_obj_names)
    delete_objs(["Plane", "myCurve", "Cylinder"])

    return frames_folder