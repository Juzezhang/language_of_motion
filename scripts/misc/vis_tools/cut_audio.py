from pydub import AudioSegment

# 读取音频文件
audio_path = "/nas/nas_32/AI-being/zhangjz/exp_motion/datasets/beat2_original/beat_v2.0.0/beat_english_v2.0.0/wave16k/2_scott_0_3_3.wav"  # 替换为你的音频文件路径
audio = AudioSegment.from_file(audio_path)

# 裁剪从第19秒开始到音频结束的部分
start_time_ms = 200/30 * 1000  # 将秒转换为毫秒
stop_time_ms = 400/30 * 1000  # 将秒转换为毫秒
trimmed_audio = audio[start_time_ms:stop_time_ms]

# 保存裁剪后的音频
output_path = "/nas/nas_32/AI-being/zhangjz/exp_motion/paper_result/videos/audios/2_scott_0_3_3_cut.wav"  # 替换为输出路径
trimmed_audio.export(output_path, format="wav")
print(f"裁剪后的音频已保存到: {output_path}")