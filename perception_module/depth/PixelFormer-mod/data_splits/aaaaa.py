file = '/Users/imrankabir/Desktop/research/semantic_seg_audio_description/PixelFormer-mod/data_splits/images_017_list.txt'

new_lines = []

with open(file, 'r') as f:
    data = f.readlines()

for line in data:
    n_l = f"{line.split()[0]} {line.split()[0]} {line.split()[1]}"
    new_lines.append(n_l)

with open(file, 'w') as f:
    f.write("\n".join(new_lines))