import moviepy.editor as mp
import moviepy.video.fx.all as vfx
import os
import imageio


def mask_png(frames):
    for frame in frames:
        im = imageio.imread(frame)
        im[im[:, :, 3] < 1, :] = 255
        imageio.imwrite(frame, im[:, :, 0:3])
    return


class Video:
    def __init__(self, frame_path: str, fps: float = 12.5, res="high"):
        frame_path = str(frame_path)
        # Guard against fps=None propagating to moviepy (TypeError in ffmpeg writer).
        self.fps = fps if fps else 30.0

        self._conf = {"codec": "libx264",
                      "fps": self.fps,
                      "audio_codec": "aac",
                      "temp_audiofile": "temp-audio.m4a",
                      "remove_temp": True}

        if res == "low":
            bitrate = "500k"
        else:
            bitrate = "5000k"

        self._conf = {"bitrate": bitrate,
                      "fps": self.fps}

        # Load video
        # video = mp.VideoFileClip(video1_path, audio=False)
        # Load with frames
        frames = [os.path.join(frame_path, x)
                  for x in sorted(os.listdir(frame_path))]

        # mask background white for videos
        mask_png(frames)

        self.frames = frames
        video = mp.ImageSequenceClip(frames, fps=self.fps)
        self.video = video
        self.duration = video.duration

    def add_text(self, text):
        # needs ImageMagick
        video_text = mp.TextClip(text,
                                 font='Amiri',
                                 color='white',
                                 method='caption',
                                 align="center",
                                 size=(self.video.w, None),
                                 fontsize=30)
        video_text = video_text.on_color(size=(self.video.w, video_text.h + 5),
                                         color=(0, 0, 0),
                                         col_opacity=0.6)
        # video_text = video_text.set_pos('bottom')
        video_text = video_text.set_pos('top')

        self.video = mp.CompositeVideoClip([self.video, video_text])

    def save(self, out_path):
        out_path = str(out_path)
        # NOTE: moviepy's write_videofile silently loses the `fps` argument when
        # run inside Blender's embedded Python (a moviepy/`decorator`-lib bug that
        # passes fps=None to the ffmpeg writer -> TypeError). Write the frames with
        # imageio instead, which is robust in that environment.
        try:
            writer = imageio.get_writer(
                out_path, fps=self.fps, codec="libx264",
                quality=8, macro_block_size=1, pixelformat="yuv420p")
            for f in self.frames:
                im = imageio.imread(f)
                if im.ndim == 3 and im.shape[-1] == 4:
                    im = im[:, :, :3]
                writer.append_data(im)
            writer.close()
        except Exception:
            # Fallback to moviepy (works in a normal Python interpreter).
            self.video.subclip(0, self.duration).write_videofile(
                out_path, **self._conf)
