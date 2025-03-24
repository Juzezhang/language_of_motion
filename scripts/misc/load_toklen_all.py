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
            if file.endswith('.npy'):
                token_file_path = join(root, file)
                speaker_id, chapter_id, file_id = parse_file_info(file)

                if speaker_id not in token_dict:
                    token_dict[speaker_id] = {}
                if chapter_id not in token_dict[speaker_id]:
                    token_dict[speaker_id][chapter_id] = {}

                token_dict[speaker_id][chapter_id][file_id] = np.load(token_file_path)
                count +=1

    return token_dict


def parse_file_info(file_name):
    parts = file_name.replace('.npy', '').split('-')
    speaker_id = parts[0]
    chapter_id = parts[1]
    file_id = parts[2]
    return speaker_id, chapter_id, file_id


def save_token_dict(token_dict, output_path):
    with open(output_path, 'wb') as f:
        pickle.dump(token_dict, f)


def main():
    token_root = '/data/data/LibriSpeech/Quantized_LibriSpeech_Encodec/train-other-500'
    output_path = '/data/data/LibriSpeech/Quantized_LibriSpeech_Encodec/train-other-500.pkl'

    token_dict = load_all_tokens(token_root)
    save_token_dict(token_dict, output_path)
    print(f"All tokens saved to {output_path}")


if __name__ == "__main__":
    main()