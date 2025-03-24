import torch
import rich
import pickle
import numpy as np


def lengths_to_mask(lengths):
    max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(
        len(lengths), max_len) < lengths.unsqueeze(1)
    return mask

# padding to max length in one batch
def collate_tensors(batch):
    if isinstance(batch[0], np.ndarray):
        batch = [torch.tensor(b).float() for b in batch]

    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch), ) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas

def lom_collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    split_name = batch[0]["split_name"]
    if split_name == 'vq':
        adapted_batch = {
            "pose": collate_tensors([b["pose"].float() for b in notnone_batches]),
            "face": collate_tensors([b["face"].float() for b in notnone_batches]),
            "hand": collate_tensors([b["hand"].float() for b in notnone_batches]),
            "upper": collate_tensors([b["upper"].float() for b in notnone_batches]),
            "lower": collate_tensors([b["lower"].float() for b in notnone_batches]),
            "shape": collate_tensors([b["shape"].float() for b in notnone_batches]),
            "trans": collate_tensors([b["trans"].float() for b in notnone_batches]),
            "motion_len": [b["motion_len"] for b in notnone_batches],
            "id_name": [b["id_name"] for b in notnone_batches],
            "dataset_name": [b["dataset_name"] for b in notnone_batches],
        }
    elif split_name == 'test':
    # if "face" in notnone_batches[0]:
        adapted_batch = {
            "face": collate_tensors([b["face"].float() for b in notnone_batches]),
            "hand": collate_tensors([b["hand"].float() for b in notnone_batches]),
            "lower": collate_tensors([b["lower"].float() for b in notnone_batches]),
            "upper": collate_tensors([b["upper"].float() for b in notnone_batches]),
            "tar_pose": collate_tensors([b["tar_pose"].float() for b in notnone_batches]),
            "tar_beta":  collate_tensors([b["tar_beta"].float() for b in notnone_batches]),
            "tar_trans": collate_tensors([b["tar_trans"].float() for b in notnone_batches]),
            "tar_exps": collate_tensors([b["tar_exps"].float() for b in notnone_batches]),
            "audio_token": collate_tensors([b["audio_token"].float() for b in notnone_batches]),
            "raw_audio": [b["raw_audio"] for b in notnone_batches],
            "m_tokens_len": collate_tensors([b["m_tokens_len"] for b in notnone_batches]),
            "a_tokens_len": [b["a_tokens_len"] for b in notnone_batches],
        }
    else:
        adapted_batch = {
            "face_token": collate_tensors([b["face_token"].float() for b in notnone_batches]),
            "hand_token": collate_tensors([b["hand_token"].float() for b in notnone_batches]),
            "lower_token": collate_tensors([b["lower_token"].float() for b in notnone_batches]),
            "upper_token": collate_tensors([b["upper_token"].float() for b in notnone_batches]),
            "audio_token": collate_tensors([b["audio_token"].float() for b in notnone_batches]),
            "tasks":  [b["tasks"] for b in notnone_batches],
            "m_tokens_len": [b["m_tokens_len"] for b in notnone_batches],
            "a_tokens_len": [b["a_tokens_len"] for b in notnone_batches],
            "text": [b["text"] for b in notnone_batches],
            "emotion_label":[b["emotion_label"] for b in notnone_batches],
        }

    return adapted_batch


def load_pkl(path, description=None, progressBar=False):
    if progressBar:
        with rich.progress.open(path, 'rb', description=description) as file:
            data = pickle.load(file)
    else:
        with open(path, 'rb') as file:
            data = pickle.load(file)
    return data
