import os
import shutil

# 定义源文件夹路径
source_folder1 = "/nas/nas_32/AI-being/zhangjz/exp_motion/datasets/beat2/beat2_text_segmented"
# source_folder2 = "/nas/nas_32/AI-being/zhangjz/exp_motion/datasets/HumanML3D/texts"

# 定义目标文件夹路径
destination_folder = "/nas/nas_32/AI-being/zhangjz/exp_motion/datasets/beat2_amass/texts"

# 如果目标文件夹不存在，则创建它
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# 复制一个文件夹中的所有子文件到目标文件夹
def copy_files_from_folder(source_folder, destination_folder):
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            # 构造文件的完整路径
            source_file = os.path.join(root, file)
            # 将文件复制到目标文件夹
            shutil.copy2(source_file, destination_folder)
            # print(f"Copied {source_file} to {destination_folder}")

# 复制两个源文件夹的文件
copy_files_from_folder(source_folder1, destination_folder)
# copy_files_from_folder(source_folder2, destination_folder)

print("All files copied successfully!")
#
# # 导入必要的库
# import os
#
# # 定义文件路径
# file1 = "/nas/nas_32/AI-being/zhangjz/exp_motion/datasets/beat2/test.txt"
# file2 = "/nas/nas_32/AI-being/zhangjz/exp_motion/datasets/HumanML3D/test.txt"
# output_file = "/nas/nas_32/AI-being/zhangjz/exp_motion/datasets/beat2_amass/test.txt"
#
# # 读取并合并文件
# with open(output_file, 'w') as outfile:
#     # 读取第一个文件并写入
#     with open(file1, 'r') as infile1:
#         outfile.write(infile1.read())
#
#     # 在两个文件内容之间添加换行符（可选）
#     outfile.write("\n")
#
#     # 读取第二个文件并写入
#     with open(file2, 'r') as infile2:
#         outfile.write(infile2.read())
#
# print(f"文件已成功合并并保存为 {output_file}")
