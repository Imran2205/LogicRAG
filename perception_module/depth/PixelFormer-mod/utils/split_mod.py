file_p = r"C:\Users\admin\Desktop\desktop\gta\PixelFormer-mod\data_splits\eigen_train_files_with_gt_kgl.txt"

with open(file_p, 'r') as f:
    data = f.readlines()

new_data = []
for line in data:
    new_line = line.replace(f'/{line.split("/")[1]}/{line.split("/")[2]}/',
                            f'/{line.split("/")[1]}/{line.split("/")[2]}/{line.split("/")[1]}/{line.split("/")[2]}/')
    new_data.append(new_line)

with open(file_p.replace('.txt', '2.txt'), 'w') as f:
    f.write("".join(new_data))
