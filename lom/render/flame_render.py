"""FLAME face mesh rendering for LoM (PyTorch3D backend).

The renderer (`RenderMesh`) and the PyAV video writer (`write_video`) are adapted from ViBES
  Copyright (c) Xuangeng Chu (xg.chu@outlook.com)
  ViBES `models/utils/renderer_utils.py` (RenderMesh) and `models/utils/utils_videos.py` (write_video)
so LoM's co-speech face output reproduces the exact ViBES look (HardPhongShader, teal mesh).

This module is intentionally self-contained: it depends only on torch, pytorch3d, av and smplx
(plus LoM's pure-torch `rotation_conversions`), so it can run in a PyTorch3D-equipped environment
(e.g. the ViBES conda env) without LoM's heavier model dependencies. The LoM model env does NOT
ship pytorch3d/av — generate the 112-D face feature there, then render it here.

Face feature layout (112-D), as produced by the ViBES 1x face VQ:
    [0:6]    head pose   (6-D rotation)
    [6:12]   jaw pose    (6-D rotation)
    [12:112] expression  (100 FLAME expression coeffs)

Example
-------
    from lom.render.flame_render import render_face_sequence
    render_face_sequence(rec_face, 'model_files/FLAME2020', 'face.mp4',
                         audio=waveform, fps=25, sample_rate=16000)
"""
import os
import numpy as np
import torch
import torch.nn as nn

from lom.utils.rotation_conversions import rotation_6d_to_axis_angle

try:  # pytorch3d is only needed for rendering; import lazily so the LoM model env can still import this file
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import (
        PerspectiveCameras, BlendParams, RasterizationSettings, PointLights,
        TexturesVertex, HardPhongShader, MeshRasterizer, MeshRenderer, Materials,
    )
    _HAS_PYTORCH3D = True
except Exception:  # pragma: no cover - depends on environment
    _HAS_PYTORCH3D = False

try:
    import av
    _HAS_AV = True
except Exception:  # pragma: no cover
    _HAS_AV = False


# ---------------------------------------------------------------------------
# Mesh renderer (adapted from ViBES models/utils/renderer_utils.py: RenderMesh)
# ---------------------------------------------------------------------------
class RenderMesh(nn.Module):
    """PyTorch3D HardPhong renderer for a fixed-topology mesh (FLAME), ViBES teal look."""

    def __init__(self, image_size, obj_filename=None, faces=None, scale=1.0):
        super().__init__()
        if not _HAS_PYTORCH3D:
            raise ImportError(
                "RenderMesh requires pytorch3d, which is not installed in this environment. "
                "Run the face render in a pytorch3d-equipped env (e.g. the ViBES conda env)."
            )
        self.ori_size = image_size
        self.image_size = int(image_size)
        self.scale = scale
        if faces is None:
            raise NotImplementedError('Must provide `faces`.')
        self.faces = faces if isinstance(faces, torch.Tensor) else torch.tensor(faces.astype(np.int32))
        self.raster_settings = RasterizationSettings(image_size=self.image_size, blur_radius=0.0, faces_per_pixel=1)

    def _build_cameras(self, transform_matrix, focal_length, device):
        batch_size = transform_matrix.shape[0]
        screen_size = torch.tensor([self.image_size, self.image_size], device=device).float()[None].repeat(batch_size, 1)
        cameras_kwargs = {
            'principal_point': torch.zeros(batch_size, 2, device=device).float(),
            'focal_length': focal_length, 'image_size': screen_size, 'device': device,
        }
        return PerspectiveCameras(**cameras_kwargs, R=transform_matrix[:, :3, :3], T=transform_matrix[:, :3, 3])

    @torch.no_grad()
    def forward(self, vertices, cameras=None, transform_matrix=None, focal_length=None):
        if cameras is None and transform_matrix is not None:
            cameras = self._build_cameras(transform_matrix, focal_length, device=vertices.device)
        if cameras is None and transform_matrix is None:
            transform_matrix = torch.tensor(
                [[[-1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 2 * self.scale]]],
                dtype=torch.float32, device=vertices.device,
            )
            cameras = self._build_cameras(transform_matrix, 12.0, device=vertices.device)
        faces = self.faces[None].repeat(vertices.shape[0], 1, 1)
        verts_rgb = torch.ones_like(vertices) * torch.tensor([17, 168, 205])[None, None].type_as(vertices) / 255.0
        textures = TexturesVertex(verts_features=verts_rgb.to(vertices.device))
        mesh = Meshes(verts=vertices.to(vertices.device), faces=faces.to(vertices.device), textures=textures)
        lights = PointLights(location=[[0.0, 1.0, 3.0]], device=vertices.device)
        materials = Materials(device=vertices.device, specular_color=[[0.6, 0.6, 0.6]], shininess=10.0)
        blend_params = BlendParams(background_color=(1.0, 1.0, 1.0))
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=self.raster_settings),
            shader=HardPhongShader(cameras=cameras, materials=materials, lights=lights,
                                   blend_params=blend_params, device=vertices.device),
        )
        render_results = renderer(mesh).permute(0, 3, 1, 2)
        images = render_results[:, :3]
        alpha_images = render_results[:, 3:]
        images = torch.nn.functional.interpolate(images, (self.ori_size, self.ori_size), mode='area')
        return images * 255, alpha_images


# ---------------------------------------------------------------------------
# Video writer (adapted from ViBES models/utils/utils_videos.py: write_video)
# ---------------------------------------------------------------------------
def write_video(video_frames, output_path, fps, audio_samples=None, sample_rate=None, acodec="aac"):
    """Encode (N, 3, H, W) RGB frames (0..255) to H.264 mp4, optionally muxing a 1-D waveform."""
    if not _HAS_AV:
        raise ImportError("write_video requires PyAV (`pip install av`).")
    assert video_frames.ndim == 4, "Input frames should be a 4D array."
    assert video_frames.shape[1] == 3, "Input frames should have 3 channels (RGB)."
    if isinstance(video_frames, torch.Tensor):
        video_frames = video_frames.cpu().numpy()
    if video_frames.dtype != np.uint8:
        video_frames = video_frames.astype(np.uint8)
    _, _, height, width = video_frames.shape
    container = av.open(output_path, mode="w")
    stream = container.add_stream("h264", rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "18"}
    audio_stream = None
    if audio_samples is not None:
        audio_stream = container.add_stream("aac", rate=sample_rate)
        audio_stream.format = "fltp"

    for frame in video_frames:
        frame = np.array(frame.tolist())  # force a contiguous standard array (avoid numpy object arrays)
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        frame = frame.transpose(1, 2, 0)
        video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
        for packet in stream.encode(video_frame):
            container.mux(packet)

    if audio_samples is not None:
        if isinstance(audio_samples, torch.Tensor):
            audio_samples = audio_samples.cpu().numpy()
        assert audio_samples.ndim == 1, "Input audio samples should be a 1D array."
        num_samples_per_frame = int(sample_rate // fps)
        for i in range(0, audio_samples.shape[0], num_samples_per_frame):
            chunk = np.array(audio_samples[i:i + num_samples_per_frame].tolist(), dtype=np.float32)
            if chunk.shape[0] < num_samples_per_frame:
                chunk = np.pad(chunk, (0, num_samples_per_frame - chunk.shape[0]), mode="constant")
            audio_frame = av.AudioFrame.from_ndarray(chunk[None], format="fltp", layout="mono")
            audio_frame.rate = sample_rate
            for packet in audio_stream.encode(audio_frame):
                container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)
    if audio_stream is not None:
        for packet in audio_stream.encode():
            container.mux(packet)
    container.close()


# ---------------------------------------------------------------------------
# High-level helper: 112-D face feature -> FLAME mesh -> rendered mp4
# ---------------------------------------------------------------------------
def render_face_sequence(rec_face, flame_dir, out_path, audio=None, *, fps=25, sample_rate=16000,
                         image_size=512, num_expression=100, flame_batch=100, max_frames=None,
                         device='cuda'):
    """Render a 112-D FLAME face feature sequence to an mp4 with ViBES's renderer.

    Args:
        rec_face: (N, 112) or (1, N, 112) array/tensor: head6D[0:6] + jaw6D[6:12] + expr[12:112].
        flame_dir: path to the FLAME2020 model dir (passed to smplx.FLAME).
        out_path: output .mp4 path.
        audio: optional 1-D waveform (float) to mux, at `sample_rate` Hz.
    Returns: number of frames written.
    """
    from smplx import FLAME

    rec_face = torch.as_tensor(np.asarray(rec_face), dtype=torch.float32)
    if rec_face.ndim == 3:
        rec_face = rec_face[0]
    if max_frames is not None:
        rec_face = rec_face[:max_frames]
    rec_face = rec_face.to(device)
    n = rec_face.shape[0]

    head = rotation_6d_to_axis_angle(rec_face[:, 0:6])
    jaw = rotation_6d_to_axis_angle(rec_face[:, 6:12])
    exp = rec_face[:, 12:12 + num_expression]

    flame = FLAME(flame_dir, num_expression_coeffs=num_expression, ext='pkl', batch_size=flame_batch).to(device)
    faces = torch.tensor(flame.faces.astype(np.int32), dtype=torch.int64)
    renderer = RenderMesh(image_size=image_size, faces=faces, scale=1.0)

    frames = []
    for i in range(0, n, flame_batch):
        cbs = min(flame_batch, n - i)
        if cbs != flame_batch:
            flame = FLAME(flame_dir, num_expression_coeffs=num_expression, ext='pkl', batch_size=cbs).to(device)
        kw = dict(global_orient=head[i:i + cbs], expression=exp[i:i + cbs], jaw_pose=jaw[i:i + cbs])
        with torch.no_grad():
            try:  # ViBES's smplx fork takes `shape=`; upstream smplx takes `betas=`
                out = flame(shape=torch.zeros(cbs, 100, device=device), **kw)
            except TypeError:
                out = flame(betas=torch.zeros(cbs, flame.num_betas, device=device), **kw)
        verts = out['vertices'] if isinstance(out, dict) else out.vertices
        for v in verts.detach():
            rgb = renderer(v[None])[0]      # (1, 3, H, W), 0..255
            frames.append(rgb.cpu()[0])
    frames = torch.stack(frames)            # (N, 3, H, W)

    audio_clip = None
    if audio is not None:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        audio_clip = audio[:int(frames.shape[0] / fps * sample_rate)]

    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    write_video(frames, out_path, fps, audio_clip, sample_rate, "aac")
    return frames.shape[0]
