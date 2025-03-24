import os
import numpy as np
import pickle
from os.path import join


def load_all_tokens(token_root):
    token_dict = {}

    count = 0
    for root, dirs, files in os.walk(token_root):
        # if count >= 100:
        #     break
        for file in files:
            if file.endswith('.txt'):
                txt_file_path = join(root, file)
                speaker_id, chapter_id, file_id, part_id = parse_file_info(file)
                if speaker_id not in token_dict:
                    token_dict[speaker_id] = {}
                if chapter_id not in token_dict[speaker_id]:
                    token_dict[speaker_id][chapter_id] = {}
                if file_id not in token_dict[speaker_id][chapter_id]:
                    token_dict[speaker_id][chapter_id][file_id] = {}
                # open file
                with open(txt_file_path, 'r') as f:
                    # read content
                    content = f.read()
                token_dict[speaker_id][chapter_id][file_id][part_id] = content
                count +=1

    return token_dict


def parse_file_info(file_name):
    parts = file_name.replace('.txt', '').split('-')
    speaker_id = parts[0]
    chapter_id = parts[1]
    file_id = parts[2].split('_')[0]
    part_id = parts[2].split('_')[-1]
    return speaker_id, chapter_id, file_id, part_id


def save_token_dict(token_dict, output_path):
    with open(output_path, 'wb') as f:
        pickle.dump(token_dict, f)


def main():
    token_root = '/nas/nas_32/AI-being/zhangjz/exp_motion/datasets/LibriSpeech/Quantized_LibriSpeech_Hubert_Split/train-other-500'
    output_path = '/nas/nas_32/AI-being/zhangjz/exp_motion/datasets/LibriSpeech/Quantized_LibriSpeech_Hubert_Split/train-other-500-texts.pkl'

    token_dict = load_all_tokens(token_root)
    save_token_dict(token_dict, output_path)
    print(f"All tokens saved to {output_path}")


if __name__ == "__main__":
    main()