import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from pl_bolts.models.self_supervised import SimCLR, SimSiam
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# load augmented images with prefix 00
images_aug = []
images = []
image_path = '/home/ashutosh/eval_dataset'
for folder in os.listdir(image_path):
    for file in os.listdir(os.path.join(image_path, folder)):
        if file.startswith('00'):
            images_aug.append(os.path.join(image_path, folder, file))
        else:
            images.append(os.path.join(image_path, folder, file))

def load_images(image_list):
    batch = []
    for image in image_list:
        img = Image.open(image).convert('RGB')
        img = img.resize((224, 224))
        img = np.asarray(img)
        batch.append(img)
    batch = np.stack(batch)
    return batch


batch_aug = load_images(images_aug)
batch_ref = load_images(images)

# normalize images and load as tensor
batch_aug = batch_aug.reshape(-1, 3, 224, 224).astype('float32') / 255.
batch_aug = torch.from_numpy(batch_aug)

batch_ref = batch_ref.reshape(-1, 3, 224, 224).astype('float32') / 255.
batch_ref = torch.from_numpy(batch_ref)
# transforms.functional.normalize(images_aug, [0.5], [0.5], inplace=True)

# pass the batch of augmented images to the model
weight_path = '/home/ashutosh/simclr/tb_logs/simsiam_tb_logs/version_1/checkpoints/epoch=77-step=13650.ckpt'
# simclr = SimCLR.load_from_checkpoint(weight_path)
# simclr.eval()
simsiam = SimSiam.load_from_checkpoint(weight_path)
with torch.no_grad():
    feats = simsiam(batch_aug)
    # z = simclr.projection(feats)
    feats_ref = simsiam(batch_ref)
    # z_ref = simclr.projection(feats_ref)

del batch_aug
del batch_ref

def multi_cosine_similarity(a, b):
    num = b@a.T
    den = np.outer(np.linalg.norm(b, axis=1), np.linalg.norm(a, axis=1))
    return num/den

row_labels = []
for image in images:
    row_labels.append(os.sep.join(os.path.normpath(image).split(os.sep)[-2:]))

col_labels = ['0']
for image in images_aug:
    col_labels.append(os.sep.join(os.path.normpath(image).split(os.sep)[-2:]))

row_labels = np.array([row_labels], dtype='object')
col_labels = np.array([col_labels], dtype='object')

feature_sim = multi_cosine_similarity(feats, feats_ref)
print(feature_sim.shape)
final = np.concatenate([row_labels.T, feature_sim], axis=1)
final = np.concatenate([col_labels, final], axis=0)
np.savetxt("simsiam_features.csv", final, delimiter=',', fmt='%s')

# z_sim = multi_cosine_similarity(z, z_ref)
# print(z_sim.shape)
# final = np.concatenate([row_labels.T, z_sim], axis=1)
# final = np.concatenate([col_labels, final], axis=0)
# np.savetxt("simclr_z.csv", final, delimiter=",", fmt='%s')
