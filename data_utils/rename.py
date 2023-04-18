import os

l = os.listdir('/home/ashutosh/dataset_v2')
for i in range(len(l)):
    os.rename(f'/home/ashutosh/dataset_v2/{l[i]}', f'/home/ashutosh/dataset_v2/{i+1}')