import torch
from torch import optim
import tqdm
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os 
from torchvision.io import read_image
from PIL import Image
from pl_bolts.models.self_supervised import SimSiam
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

######## Transformations #########
class ContrastiveTransformations:
    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for i in range(self.n_views)]

# These transformations are as described in the SimCLR paper
contrast_transforms = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        #####
        transforms.Resize(size=500),
        #####
        transforms.RandomResizedCrop(size=(224,224)),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2, hue=0.1)], p=0.8),
        # transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=21),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

######### Loading dataset ##########
class ImageDataset(Dataset):
  def __init__(self, image_list, transform=None):
    self.image_list = image_list
    self.transform = transform
    
  def __len__(self):
    return len(self.image_list)

  def __getitem__(self, idx):
    img_path = self.image_list[idx]
    image = Image.open(img_path)
    image = image.convert('RGB')
    if self.transform:
      image = self.transform(image)
    return image

# Loading image paths
final_image_path_list = []
image_list = os.listdir('/home/ashutosh/dataset_v2')
for i in range(len(image_list)):
  img = os.listdir(os.path.join('/home/ashutosh/dataset_v2', image_list[i]))
  if img == [] or img == None:
    continue
  final_image_path_list.append(os.path.join('/home/ashutosh/dataset_v2', image_list[i], img[0]))

# Filter corrupted files from list
to_be_removed = set()
for image in tqdm.tqdm(final_image_path_list):
  try:
    img = Image.open(image).convert('RGB')
    img.close()
  except Exception as e:
    to_be_removed.add(image)
final_image_path_list = [image for image in final_image_path_list if image not in to_be_removed]
print(f'Total number of images: {len(final_image_path_list)}')


unlabeled_data_train = ImageDataset(final_image_path_list[:11200], ContrastiveTransformations(contrast_transforms, n_views=2))
training_loader = DataLoader(unlabeled_data_train, batch_size=64, num_workers=16)

unlabeled_data_val = ImageDataset(final_image_path_list[11200:], ContrastiveTransformations(contrast_transforms, n_views=2))
val_loader = DataLoader(unlabeled_data_val, batch_size=64, num_workers=16)

print('$$$$$$$$$$$$$$$')
print(len(final_image_path_list))
print('$$$$$$$$$$$$$$$')

if __name__ == '__main__':
  simsiam = SimSiam()
  simsiam.eval()
  lr_monitor = LearningRateMonitor(logging_interval="step")
  model_checkpoint = ModelCheckpoint(save_last=True, save_top_k=1, monitor="train_loss")
  callbacks = [model_checkpoint, lr_monitor]

  tb_logger = TensorBoardLogger('tb_logs', name='simsiam_tb_logs')
  trainer = pl.Trainer(
      max_epochs = 100,
      accelerator = 'gpu',
      enable_checkpointing = True,
      logger=tb_logger,
      callbacks=callbacks,
  )

  trainer.fit(simsiam, train_dataloaders=training_loader, val_dataloaders=val_loader)
  print(model_checkpoint.best_model_path)
  print(model_checkpoint.best_model_score)
