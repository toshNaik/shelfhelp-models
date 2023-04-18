import os
import tqdm
from PIL import Image

image_list = os.listdir('/home/ashutosh/dataset_v2')
to_be_removed = set() 
for image in tqdm.tqdm(image_list):
    try:
        l = os.listdir(os.path.join('/home/ashutosh/dataset', image))
        if l == None or l == []: continue
        path = os.path.join('/home/ashutosh/dataset', image, l[0])
        img = Image.open(path).convert('RGB')
        img.close()
    except Exception as e:
        to_be_removed.add(image)

image_list = [image for image in image_list if image not in to_be_removed]
# for image in tqdm.tqdm(image_list):
#     try:
#         path = os.path.join('/home/ashutosh/dataset', image, 'image_1.jpg')
#         img = Image.open(path).convert('RGB')
#         img.close()
#     except Exception as e:
#         image_list.remove(image)

print(len(image_list))