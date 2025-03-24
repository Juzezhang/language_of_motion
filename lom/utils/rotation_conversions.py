# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import functools
from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np


"""
The transformation matrices returned from the functions in this file assume
the points on which the transformation will be applied are column vectors.
i.e. the R matrix is structured as

    R = [
            [Rxx, Rxy, Rxz],
            [Ryx, Ryy, Ryz],
            [Rzx, Rzy, Rzz],
        ]  # (3, 3)

This matrix can be applied to column vectors by post multiplication
by the points e.g.

    points = [[0], [1], [2]]  # (3 x 1) xyz coordinates of a point
    transformed_points = R * points

To apply the same matrix to points which are row vectors, the R matrix
can be transposed and pre multiplied by the points:

e.g.
    points = [[0, 1, 2]]  # (1 x 3) xyz coordinates of a point
    transformed_points = points * R.transpose(1, 0)
"""


def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))



def quaternion_to_matrix_np(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: A numpy array of shape (..., 4) representing quaternions with the real part first.

    Returns:
        A numpy array of shape (..., 3, 3) representing the corresponding rotation matrices.
    """
    # Unpack quaternion components: real part (r) and imaginary parts (i, j, k)
    r = quaternions[..., 0]
    i = quaternions[..., 1]
    j = quaternions[..., 2]
    k = quaternions[..., 3]

    # Compute the scaling factor: two_s = 2.0 / (sum of squares of quaternion components)
    two_s = 2.0 / np.sum(quaternions**2, axis=-1)

    # Compute the elements of the rotation matrix based on the quaternion formula:
    # First row of the rotation matrix
    m00 = 1 - two_s * (j**2 + k**2)
    m01 = two_s * (i * j - k * r)
    m02 = two_s * (i * k + j * r)
    
    # Second row of the rotation matrix
    m10 = two_s * (i * j + k * r)
    m11 = 1 - two_s * (i**2 + k**2)
    m12 = two_s * (j * k - i * r)
    
    # Third row of the rotation matrix
    m20 = two_s * (i * k - j * r)
    m21 = two_s * (j * k + i * r)
    m22 = 1 - two_s * (i**2 + j**2)
    
    # Stack the rows to form the rotation matrix of shape (..., 3, 3)
    matrix = np.stack([
        np.stack([m00, m01, m02], axis=-1),
        np.stack([m10, m11, m12], axis=-1),
        np.stack([m20, m21, m22], axis=-1)
    ], axis=-2)
    
    return matrix


def _copysign(a, b):
    """
    Return a tensor where each element has the absolute value taken from the,
    corresponding element of a, with sign taken from the corresponding
    element of b. This is like the standard copysign floating-point operation,
    but is not careful about negative 0 and NaN.

    Args:
        a: source tensor.
        b: tensor whose signs will be used, of the same shape as a.

    Returns:
        Tensor of the same shape as a with the signs of b.
    """
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)


def _sqrt_positive_part(x):
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quaternion(matrix):
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")
    m00 = matrix[..., 0, 0]
    m11 = matrix[..., 1, 1]
    m22 = matrix[..., 2, 2]
    o0 = 0.5 * _sqrt_positive_part(1 + m00 + m11 + m22)
    x = 0.5 * _sqrt_positive_part(1 + m00 - m11 - m22)
    y = 0.5 * _sqrt_positive_part(1 - m00 + m11 - m22)
    z = 0.5 * _sqrt_positive_part(1 - m00 - m11 + m22)
    o1 = _copysign(x, matrix[..., 2, 1] - matrix[..., 1, 2])
    o2 = _copysign(y, matrix[..., 0, 2] - matrix[..., 2, 0])
    o3 = _copysign(z, matrix[..., 1, 0] - matrix[..., 0, 1])
    return torch.stack((o0, o1, o2, o3), -1)


def _axis_angle_rotation(axis: str, angle):
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    if axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    if axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(euler_angles, convention: str):
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = map(_axis_angle_rotation, convention, torch.unbind(euler_angles, -1))
    return functools.reduce(torch.matmul, matrices)


def _angle_from_tan(
    axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
):
    """
    Extract the first or third Euler angle from the two members of
    the matrix which are positive constant times its sine and cosine.

    Args:
        axis: Axis label "X" or "Y or "Z" for the angle we are finding.
        other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
            convention.
        data: Rotation matrices as tensor of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
        tait_bryan: Whether the first and third axes in the convention differ.

    Returns:
        Euler Angles in radians for each matrix in data as a tensor
        of shape (...).
    """

    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])


def _index_from_letter(letter: str):
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2


def matrix_to_euler_angles(matrix, convention: str):
    """
    Convert rotations given as rotation matrices to Euler angles in radians.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        convention: Convention string of three uppercase letters.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")
    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = torch.asin(
            matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        )
    else:
        central_angle = torch.acos(matrix[..., i0, i0])

    o = (
        _angle_from_tan(
            convention[0], convention[1], matrix[..., i2], False, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
        ),
    )
    return torch.stack(o, -1)


def random_quaternions(
    n: int, dtype: Optional[torch.dtype] = None, device=None, requires_grad=False
):
    """
    Generate random quaternions representing rotations,
    i.e. versors with nonnegative real part.

    Args:
        n: Number of quaternions in a batch to return.
        dtype: Type to return.
        device: Desired device of returned tensor. Default:
            uses the current device for the default tensor type.
        requires_grad: Whether the resulting tensor should have the gradient
            flag set.

    Returns:
        Quaternions as tensor of shape (N, 4).
    """
    o = torch.randn((n, 4), dtype=dtype, device=device, requires_grad=requires_grad)
    s = (o * o).sum(1)
    o = o / _copysign(torch.sqrt(s), o[:, 0])[:, None]
    return o


def random_rotations(
    n: int, dtype: Optional[torch.dtype] = None, device=None, requires_grad=False
):
    """
    Generate random rotations as 3x3 rotation matrices.

    Args:
        n: Number of rotation matrices in a batch to return.
        dtype: Type to return.
        device: Device of returned tensor. Default: if None,
            uses the current device for the default tensor type.
        requires_grad: Whether the resulting tensor should have the gradient
            flag set.

    Returns:
        Rotation matrices as tensor of shape (n, 3, 3).
    """
    quaternions = random_quaternions(
        n, dtype=dtype, device=device, requires_grad=requires_grad
    )
    return quaternion_to_matrix(quaternions)


def random_rotation(
    dtype: Optional[torch.dtype] = None, device=None, requires_grad=False
):
    """
    Generate a single random 3x3 rotation matrix.

    Args:
        dtype: Type to return
        device: Device of returned tensor. Default: if None,
            uses the current device for the default tensor type
        requires_grad: Whether the resulting tensor should have the gradient
            flag set

    Returns:
        Rotation matrix as tensor of shape (3, 3).
    """
    return random_rotations(1, dtype, device, requires_grad)[0]


def standardize_quaternion(quaternions):
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def quaternion_raw_multiply(a, b):
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)


def quaternion_multiply(a, b):
    """
    Multiply two quaternions representing rotations, returning the quaternion
    representing their composition, i.e. the versor with nonnegative real part.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions of shape (..., 4).
    """
    ab = quaternion_raw_multiply(a, b)
    return standardize_quaternion(ab)


def quaternion_invert(quaternion):
    """
    Given a quaternion representing rotation, get the quaternion representing
    its inverse.

    Args:
        quaternion: Quaternions as tensor of shape (..., 4), with real part
            first, which must be versors (unit quaternions).

    Returns:
        The inverse, a tensor of quaternions of shape (..., 4).
    """

    return quaternion * quaternion.new_tensor([1, -1, -1, -1])


def quaternion_apply(quaternion, point):
    """
    Apply the rotation given by a quaternion to a 3D point.
    Usual torch rules for broadcasting apply.

    Args:
        quaternion: Tensor of quaternions, real part first, of shape (..., 4).
        point: Tensor of 3D points of shape (..., 3).

    Returns:
        Tensor of rotated points of shape (..., 3).
    """
    if point.size(-1) != 3:
        raise ValueError(f"Points are not in 3D, f{point.shape}.")
    real_parts = point.new_zeros(point.shape[:-1] + (1,))
    point_as_quaternion = torch.cat((real_parts, point), -1)
    out = quaternion_raw_multiply(
        quaternion_raw_multiply(quaternion, point_as_quaternion),
        quaternion_invert(quaternion),
    )
    return out[..., 1:]


def axis_angle_to_matrix(axis_angle):
    """
    Convert rotations given as axis/angle to rotation matrices.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


    """
    Convert rotations given as axis/angle to rotation matrices using Rodrigues' formula.

    Args:
        axis_angle (Tensor): Rotations in axis-angle form, shape (..., 3).

    Returns:
        Tensor: Rotation matrices of shape (..., 3, 3).
    """
    # Compute the angle (magnitude) and unit axis
    angle = torch.norm(axis_angle, dim=-1, keepdim=True)  # (..., 1)
    axis = axis_angle / (angle + 1e-8)  # Normalize axis (..., 3)

    # Compute sine and cosine of the angle
    cos_theta = torch.cos(angle)  # (..., 1)
    sin_theta = torch.sin(angle)  # (..., 1)
    one_minus_cos = 1 - cos_theta  # (..., 1)

    # Extract components
    x, y, z = axis[..., 0:1], axis[..., 1:2], axis[..., 2:3]  # Each (..., 1)

    # Compute the cross-product matrix elements
    # Using einsum to handle arbitrary batch dimensions
    zeros = torch.zeros_like(x)
    
    # Construct the skew-symmetric matrix K properly
    K = torch.zeros((*axis_angle.shape[:-1], 3, 3), device=axis_angle.device, dtype=axis_angle.dtype)
    K[..., 0, 1] = -z.squeeze(-1)
    K[..., 0, 2] = y.squeeze(-1)
    K[..., 1, 0] = z.squeeze(-1)
    K[..., 1, 2] = -x.squeeze(-1)
    K[..., 2, 0] = -y.squeeze(-1)
    K[..., 2, 1] = x.squeeze(-1)

    # Correctly expand identity matrix to match batch size
    batch_size = axis.shape[:-1]  # Get batch dimensions
    identity = torch.eye(3, device=axis.device, dtype=axis.dtype).expand(*batch_size, 3, 3)  # (..., 3, 3)

    # Compute final rotation matrix using the Rodrigues formula
    R = identity + sin_theta[..., None] * K + one_minus_cos[..., None] * torch.matmul(K, K)  # (..., 3, 3)

    return R


def matrix_to_axis_angle(matrix):
    """
    Convert rotations given as rotation matrices to axis/angle.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


def axis_angle_to_quaternion(axis_angle):
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = 0.5 * angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions


def axis_angle_to_quaternion_np(axis_angle):
    """
    Convert rotations given in axis-angle representation to quaternions.

    Args:
        axis_angle: A numpy array of shape (..., 3), where the magnitude of each vector 
                    represents the rotation angle in radians, and the direction of the vector 
                    represents the rotation axis.

    Returns:
        A numpy array of shape (..., 4) representing quaternions with the real part first.
    """
    # Compute the Euclidean norm along the last dimension (rotation angles)
    angles = np.linalg.norm(axis_angle, axis=-1, keepdims=True)
    half_angles = 0.5 * angles
    eps = 1e-6
    # Identify small angles to avoid numerical issues
    small_angles = np.abs(angles) < eps
    sin_half_angles_over_angles = np.empty_like(angles)

    # For non-small angles, compute sin(half_angle)/angle directly
    sin_half_angles_over_angles[~small_angles] = np.sin(half_angles[~small_angles]) / angles[~small_angles]
    # For small angles, use a Taylor series approximation:
    # sin(x/2)/x ≈ 1/2 - (x^2)/48
    sin_half_angles_over_angles[small_angles] = 0.5 - (angles[small_angles]**2) / 48.0

    # Construct the quaternion by concatenating the real part (cos(half_angle))
    # with the imaginary part (axis_angle * sin(half_angle)/angle)
    quaternions = np.concatenate([np.cos(half_angles), axis_angle * sin_half_angles_over_angles], axis=-1)
    return quaternions



def quaternion_to_axis_angle(quaternions):
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles


def quaternion_to_axis_angle_np(quaternions):
    """
    Convert rotations given as quaternions to axis-angle representation.

    Args:
        quaternions: A numpy array of shape (..., 4) representing quaternions with the real part first.

    Returns:
        A numpy array of shape (..., 3) representing rotations in axis-angle form,
        where the magnitude of each vector is the rotation angle in radians (anticlockwise)
        around the vector's direction.
    """
    # Compute the Euclidean norm of the imaginary components (last three elements)
    norms = np.linalg.norm(quaternions[..., 1:], axis=-1, keepdims=True)
    
    # Compute the half angles using arctan2: arctan2(norm, real_part)
    half_angles = np.arctan2(norms, quaternions[..., :1])
    
    # Calculate the full rotation angles
    angles = 2 * half_angles
    
    eps = 1e-6
    # Identify small angles to avoid numerical instability
    small_angles = np.abs(angles) < eps
    sin_half_angles_over_angles = np.empty_like(angles)
    
    # For non-small angles, compute sin(half_angle)/angle directly
    sin_half_angles_over_angles[~small_angles] = np.sin(half_angles[~small_angles]) / angles[~small_angles]
    # For small angles, use a Taylor series approximation:
    # sin(x/2)/x ≈ 1/2 - (x^2)/48
    sin_half_angles_over_angles[small_angles] = 0.5 - (angles[small_angles]**2) / 48.0
    
    # Convert the quaternion to axis-angle representation:
    # axis_angle = (imaginary part) / (sin(half_angle)/angle)
    axis_angle = quaternions[..., 1:] / sin_half_angles_over_angles
    
    return axis_angle


def axis_angle_to_6d(axis_angle):
    """
    Convert rotations given as axis/angle to 6D rotation representation by Zhou et al. [1].

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        6D rotation representation, of size (..., 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    # Convert to rotation matrix first, then to 6D representation
    matrices = axis_angle_to_matrix(axis_angle)
    return matrix_to_rotation_6d(matrices)


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalisation per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    return matrix[..., :2, :].clone().reshape(*matrix.size()[:-2], 6)


def rotation_6d_to_axis_angle(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to axis-angle representation.

    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        axis-angle representation, of size (*, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    # First convert 6D representation to rotation matrix
    matrices = rotation_6d_to_matrix(d6)
    
    # Then convert rotation matrix to axis-angle
    return matrix_to_axis_angle(matrices)


def axis_angle_to_6d_np(axis_angle):
    """Convert axis-angle to 6D rotation representation (NumPy version)."""
    # Convert to rotation matrix first, then to 6D representation
    matrices = axis_angle_to_matrix_np(axis_angle)
    return matrix_to_rotation_6d_np(matrices)

def axis_angle_to_matrix_np(axis_angle):
    """
    Convert rotations given as axis/angle to rotation matrices.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return quaternion_to_matrix_np(axis_angle_to_quaternion_np(axis_angle))



def matrix_to_rotation_6d_np(matrix):
    """
    Converts rotation matrices to the 6D rotation representation proposed by Zhou et al. [1]
    by taking the first two rows of the matrix and flattening them. Note that the 6D 
    representation is not unique.

    Args:
        matrix: A numpy array of shape (..., 3, 3), representing a batch of rotation matrices.

    Returns:
        A numpy array of shape (..., 6), representing the 6D rotation representation.

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    "On the Continuity of Rotation Representations in Neural Networks."
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    # Extract the first two rows of the rotation matrix
    rotation_6d = matrix[..., :2, :].reshape(*matrix.shape[:-2], 6)
    return rotation_6d