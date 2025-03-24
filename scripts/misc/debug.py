import numpy as np


# pose_file_beat2 = "/simurgh/u/juze/datasets/BEAT2/beat_english_v2.0.0/smplxflame_30/10_kieks_0_10_10.npz"
# pose_data_beat2 = np.load(pose_file_beat2, allow_pickle=True)

# print(pose_data_beat2.files)  

pose_file_amass = "/simurgh/u/juze/datasets/AMASS/amass_data_align/001578.npz"
pose_data_amass = np.load(pose_file_amass, allow_pickle=True)

print("AMASS data keys:", pose_data_amass.files)
print("Current betas shape:", pose_data_amass['betas'].shape)

# Extract all the data from the original file
data_dict = {}
for key in pose_data_amass.files:
    data_dict[key] = pose_data_amass[key]

# Pad betas with zeros to make it 300-dimensional
original_betas = data_dict['betas']
padded_betas = np.zeros(300)
padded_betas[:len(original_betas)] = original_betas
data_dict['betas'] = padded_betas
data_dict['num_betas'] = 300  # Update the num_betas field if it exists

# Create a new output directory if it doesn't exist

np.savez('001578_new', **data_dict)

# print(f"Saved new file with 300-dim betas to: {output_file}")
print(f"New betas shape: {data_dict['betas'].shape}")


