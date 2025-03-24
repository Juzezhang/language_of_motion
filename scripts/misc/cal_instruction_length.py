import json
import os
import textgrid as tg
from tqdm import tqdm
from transformers import AutoTokenizer
from mGPT.data.mixed_dataset.utils.split_transcript import split_and_merge_sentences

max_token_instruction = 52
root_path = '/nas/nas_32/AI-being/zhangjz/exp_motion/datasets/beat2_original/raw_data/beat_v2.0.0/beat_english_v2.0.0/textgrid/'
text_list = os.listdir(root_path)

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained('./deps/flan-t5-base', legacy=True)

token_lengths = []
transcript_token_lengths = []
audio_token_lengths = []
times_list = []
for text_name in tqdm(text_list):

    text_path = os.path.join(root_path, text_name)
    tgrid = tg.TextGrid.fromFile(text_path)
    paragraphs = split_and_merge_sentences(tgrid[0].intervals, max_duration=8.0)
    for i, (paragraph, start_time, end_time) in enumerate(paragraphs):
        times_list.append(end_time - start_time)

        if (end_time-start_time)>8.1:
            pass

        audio_start = int(start_time * 16000 / 320.)
        audio_end = int(end_time * 16000 / 320.)
        audio_token_length = audio_end - audio_start
        audio_token_lengths.append(audio_token_length)
        # 对文本进行编码
        encoded_input = tokenizer(paragraph, return_tensors="pt")
        # 获取 token 数量
        transcript_token_length = len(encoded_input["input_ids"][0])
        transcript_token_lengths.append(transcript_token_length)
        total_token_length = max_token_instruction + audio_token_length + transcript_token_length
        token_lengths.append(total_token_length)


max(times_list)
max(transcript_token_lengths)
max(token_lengths)
pass