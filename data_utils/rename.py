# import os

# l = os.listdir('/home/ashutosh/dataset_v2')
# for i in range(len(l)):
#     os.rename(f'/home/ashutosh/dataset_v2/{l[i]}', f'/home/ashutosh/dataset_v2/{i+1}')

# write code to rename file in all subfolders to orig.ext
import os
import shutil

path = '/home/ashutosh/eval_dataset'
for folder in os.listdir(path):
    for file in os.listdir(os.path.join(path, folder)):
        file_name, ext = file.split('.')
        shutil.move(os.path.join(path, folder, file), os.path.join(path, folder, 'orig.'+ext))