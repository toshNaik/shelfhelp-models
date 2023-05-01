import os
import numpy as np
from PIL import Image
import imgaug
import imgaug.augmenters as iaa

# set seed for reproducibility
imgaug.seed(12)

# define augmentations
aug = iaa.Sequential([
    iaa.Sometimes(1.0, iaa.Sequential([
        iaa.Multiply((0.5, 1.5)),
        iaa.LinearContrast((0.5, 2.0)),
        iaa.AddToHueAndSaturation((-10, 10)),
    ])),
    iaa.Sometimes(1.0, iaa.OneOf([
        iaa.GaussianBlur((0, 3.0)),
        iaa.AverageBlur(k=(1, 4)),
        iaa.MedianBlur(k=(1, 5)),
    ])),
    iaa.Sometimes(1.0, iaa.OneOf([
        iaa.Affine(rotate=(-10, 10)),
        iaa.Affine(shear=(-10, 10)),
    ])),
])

# load images. there's one image in each folder
images = []
image_path = '/home/ashutosh/eval_dataset'
for folder in os.listdir(image_path):
    for file in os.listdir(os.path.join(image_path, folder)):
        if file.startswith('orig'):
            images.append(os.path.join(image_path, folder, file))

# Repeat 5 times to get 5 augmented images for each image
for i in range(5):
    # for each image augment it and save it
    for image in images:
        img = Image.open(image).convert('RGB')
        img = img.resize((224, 224))
        img = np.asarray(img)
        img = aug(image=img)
        img = Image.fromarray(img)

        # get the path of the folder
        folder_path = os.path.dirname(image)
        # get the extension of the image
        _, ext = os.path.basename(image).split('.')
        image_name = f'00{i}_aug.' + ext
        # save the image
        img.save(os.path.join(folder_path, image_name))
