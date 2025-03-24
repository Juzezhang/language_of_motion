import bpy
from .materials import floor_mat
import numpy as np


def get_trajectory(data, is_mesh):
    if is_mesh:
        # mean of the vertices
        trajectory = data[:, :, [0, 1]].mean(1)
    else:
        # get the root joint
        trajectory = data[:, 0, [0, 1]]
    return trajectory


def plot_floor(data, big_plane=True):
    # Create a floor
    minx, miny, _ = data.min(axis=(0, 1))
    maxx, maxy, _ = data.max(axis=(0, 1))
    minz = 0

    location = ((maxx + minx)/2, (maxy + miny)/2, 0)
    # a little bit bigger
    scale = (1.08*(maxx - minx)/2, 1.08*(maxy - miny)/2, 1)

    bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=location, scale=(1, 1, 1))

    bpy.ops.transform.resize(value=scale, orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL',
                             constraint_axis=(False, True, False), mirror=True, use_proportional_edit=False,
                             proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False,
                             use_proportional_projected=False, release_confirm=True)
    obj = bpy.data.objects["Plane"]
    obj.name = "SmallPlane"
    obj.data.name = "SmallPlane"

    if not big_plane:
        obj.active_material = floor_mat(color=(0.2, 0.2, 0.2, 1))
    else:
        obj.active_material = floor_mat(color=(0.1, 0.1, 0.1, 1))

    if big_plane:
        location = ((maxx + minx)/2, (maxy + miny)/2, -0.01)
        bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=location, scale=(1, 1, 1))

        bpy.ops.transform.resize(value=[2*x for x in scale], orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL',
                                 constraint_axis=(False, True, False), mirror=True, use_proportional_edit=False,
                                 proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False,
                                 use_proportional_projected=False, release_confirm=True)

        obj = bpy.data.objects["Plane"]
        obj.name = "BigPlane"
        obj.data.name = "BigPlane"
        obj.active_material = floor_mat(color=(0.2, 0.2, 0.2, 1))


def plot_floor_multiperson(data_list, big_plane=True):
    """
    Create a floor dynamically sized to encompass all mesh data.

    Args:
        data_list (list): List of numpy arrays, each representing mesh data.
                          Expected shape: [num_vertices, 3].
        big_plane (bool): Whether to create an additional larger plane below the main one.
    """
    # Concatenate all mesh data
    combined_data = np.concatenate(data_list, axis=0)

    if combined_data.ndim != 2 or combined_data.shape[1] != 3:
        raise ValueError(f"Expected a 2D array with shape [num_vertices, 3], but got {combined_data.shape}")

    # Calculate the bounding box
    minx, miny, _ = combined_data.min(axis=0)
    maxx, maxy, _ = combined_data.max(axis=0)

    # Center and scale of the floor
    location = ((maxx + minx) / 2, (maxy + miny) / 2, 0)
    scale = (1.08 * (maxx - minx) / 2, 1.08 * (maxy - miny) / 2, 1)

    # Add the main floor plane
    bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=location, scale=(1, 1, 1))
    bpy.ops.transform.resize(value=scale, orient_type='GLOBAL',
                             orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
                             orient_matrix_type='GLOBAL',
                             constraint_axis=(False, True, False), mirror=True,
                             use_proportional_edit=False,
                             proportional_edit_falloff='SMOOTH', proportional_size=1,
                             use_proportional_connected=False,
                             use_proportional_projected=False, release_confirm=True)

    obj = bpy.data.objects["Plane"]
    obj.name = "SmallPlane"
    obj.data.name = "SmallPlane"

    # Assign material for the main floor
    obj.active_material = floor_mat(color=(0.2, 0.2, 0.2, 1) if not big_plane else (0.1, 0.1, 0.1, 1))

    # Add an additional big plane if required
    if big_plane:
        location = ((maxx + minx) / 2, (maxy + miny) / 2, -0.01)
        bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=location,
                                         scale=(1, 1, 1))
        bpy.ops.transform.resize(value=[2 * x for x in scale], orient_type='GLOBAL',
                                 orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
                                 orient_matrix_type='GLOBAL',
                                 constraint_axis=(False, True, False), mirror=True,
                                 use_proportional_edit=False,
                                 proportional_edit_falloff='SMOOTH', proportional_size=1,
                                 use_proportional_connected=False,
                                 use_proportional_projected=False, release_confirm=True)

        obj = bpy.data.objects["Plane"]
        obj.name = "BigPlane"
        obj.data.name = "BigPlane"
        obj.active_material = floor_mat(color=(0.2, 0.2, 0.2, 1))


def show_traj(coords):
    pass
    # create the Curve Datablock
    # curveData = bpy.data.curves.new('myCurve', type='CURVE')
    # curveData.dimensions = '3D'
    # curveData.resolution_u = 2

    # # map coords to spline
    # polyline = curveData.splines.new('POLY')
    # polyline.points.add(len(coords)-1)
    # for i, coord in enumerate(coords):
    #     x, y = coord
    #     polyline.points[i].co = (x, y, 0.001, 1)

    # # create Object
    # curveOB = bpy.data.objects.new('myCurve', curveData)
    # curveData.bevel_depth = 0.01

    # bpy.context.collection.objects.link(curveOB)
