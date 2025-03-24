import os
import shutil

# 源文件夹路径
source_dir = "/nas/nas_38/AI-being/libin/PantoMatrix/SynTalker/test_npz/obj/2_scott_0_73_73/"

# 目标文件夹路径
target_dir = "/nas/nas_32/AI-being/zhangjz/exp_motion/paper_result/videos/a2m_supp/syntalker/"
os.makedirs(target_dir, exist_ok=True)  # 确保目标文件夹存在

# 要复制的文件列表
files_to_copy = [
    "000570.obj",
    "000585.obj",
    "000600.obj",
    "000615.obj",
    "000630.obj",
    "000645.obj",
    "000660.obj",
    "000675.obj",
    "000690.obj",
    "000705.obj",
    "000720.obj",
    "000735.obj",
    "000750.obj",
]

# 遍历文件列表并复制文件
for file_name in files_to_copy:
    source_file = os.path.join(source_dir, file_name)
    target_file = os.path.join(target_dir, file_name)

    if os.path.exists(source_file):
        shutil.copy(source_file, target_file)
        # shutil.copytree(source_file, target_file)

        print(f"Copied: {file_name} to {target_dir}")
    else:
        print(f"File not found: {file_name}")

print("File copy process completed.")