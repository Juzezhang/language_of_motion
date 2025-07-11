import os
import random
import shutil
import sys
import natsort
from pathlib import Path
from argparse import ArgumentParser

try:
    import bpy

    sys.path.append(os.path.dirname(bpy.data.filepath))

    # local packages
    sys.path.append(os.path.expanduser("~/.local/lib/python3.9/site-packages"))
except ImportError:
    raise ImportError(
        "Blender is not properly installed or not launch properly. See README.md to have instruction on how to install and use blender."
    )


# Monkey patch argparse such that
# blender / python / hydra parsing works
def parse_args(self, args=None, namespace=None):
    if args is not None:
        return self.parse_args_bak(args=args, namespace=namespace)
    try:
        idx = sys.argv.index("--")
        args = sys.argv[idx + 1:]  # the list after '--'
    except ValueError as e:  # '--' not in the list:
        args = []
    return self.parse_args_bak(args=args, namespace=namespace)


setattr(ArgumentParser, 'parse_args_bak', ArgumentParser.parse_args)
setattr(ArgumentParser, 'parse_args', parse_args)

from lom.config import parse_args


def render_cli() -> None:
    # parse options
    cfg = parse_args(phase="render")  # parse config file
    # cfg.FOLDER = cfg.RENDER.FOLDER

    DIR_all = os.listdir(cfg.RENDER.DIR)

    for DIR_sub in DIR_all:

        print(DIR_sub)
        DIR_i = os.path.join(cfg.RENDER.DIR, DIR_sub)
        if not os.path.isdir(DIR_i):
            continue

        if cfg.RENDER.INPUT_MODE.lower() == "npy":
            output_dir = Path(os.path.dirname(cfg.RENDER.NPY))
            paths = [cfg.RENDER.NPY]
        elif cfg.RENDER.INPUT_MODE.lower() == "dir":
            output_dir = Path(DIR_i)
            paths = []
            file_list = natsort.natsorted(os.listdir(DIR_i))
            # begin_id = random.randrange(0, len(file_list))
            begin_id = 0
            file_list = file_list[begin_id:] + file_list[:begin_id]

            # render mesh npy first
            for item in file_list:
                if item.endswith("_mesh.npy"):
                    paths.append(os.path.join(DIR_i, item))

            # then render joint npy
            for item in file_list:
                if item.endswith(".npy") and not item.endswith("_mesh.npy"):
                    paths.append(os.path.join(DIR_i, item))

            print(f"begin to render for {paths[0]}")

        import numpy as np
        from lom.render.blender import render_multiperson
        from lom.render.video import Video

        init = True
        data_list = []
        for path in paths:
            try:
                data = np.load(path)
                # print(data.shape)
                if data.shape[0] == 1:
                    data = data[0]
                data_list.append(data)
            except FileNotFoundError:
                print(f"{path} not found")
                continue


        # check existed mp4 or under rendering
        if cfg.RENDER.MODE == "video":
            if os.path.exists(paths[0].replace(".npy", ".mp4")) or os.path.exists(
                    paths[0].replace(".npy", "_frames")) or  os.path.exists( os.path.join(cfg.RENDER.DIR, str(DIR_sub) + '.mp4')):
                continue
                print(f"npy is rendered or under rendering {path}")
                exit()
        else:
            # check existed png
            if os.path.exists(paths[0].replace(".npy", ".png")):
                print(f"npy is rendered or under rendering {path}")
                exit()

        if cfg.RENDER.MODE == "video":
            frames_folder = os.path.join(
                output_dir,
                paths[0].replace(".npy", "_frames").split('/')[-1])
            os.makedirs(frames_folder, exist_ok=True)
        else:
            frames_folder = os.path.join(
                output_dir,
                paths[0].replace(".npy", ".png").split('/')[-1])


        if cfg.RENDER.MODE == "video":
            frames_folder = os.path.join(
                output_dir,
                paths[0].replace(".npy", "_frames").split("/")[-1])
        else:
            frames_folder = os.path.join(
                output_dir,
                paths[0].replace(".npy", ".png").split("/")[-1])

        out = render_multiperson(
            data_list,
            frames_folder,
            canonicalize=cfg.RENDER.CANONICALIZE,
            exact_frame=cfg.RENDER.EXACT_FRAME,
            num=cfg.RENDER.NUM,
            mode=cfg.RENDER.MODE,
            model_path=cfg.RENDER.MODEL_PATH,
            faces_path=cfg.RENDER.FACES_PATH,
            downsample=cfg.RENDER.DOWNSAMPLE,
            always_on_floor=cfg.RENDER.ALWAYS_ON_FLOOR,
            oldrender=cfg.RENDER.OLDRENDER,
            res=cfg.RENDER.RES,
            init=init,
            gt=cfg.RENDER.GT,
            accelerator=cfg.ACCELERATOR,
            device=cfg.DEVICE,
        )

        init = False

        if cfg.RENDER.MODE == "video":
            shutil.copytree(frames_folder, frames_folder+'_img')
            if cfg.RENDER.DOWNSAMPLE:
                video = Video(frames_folder, fps=cfg.RENDER.FPS)
            else:
                video = Video(frames_folder, fps=cfg.RENDER.FPS)

            # vid_path = frames_folder.replace("_frames", ".mp4")
            vid_path = os.path.join(cfg.RENDER.DIR, str(DIR_sub) + '.mp4')
            video.save(out_path=vid_path)
            shutil.rmtree(frames_folder)
            print(f"remove tmp fig folder and save video in {vid_path}")

        else:
            print(f"Frame generated at: {out}")


if __name__ == "__main__":
    render_cli()
