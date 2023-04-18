import torch
from torch import optim
import tqdm
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt
from torchmetrics.classification import MulticlassAccuracy
import os 
from torchvision.io import read_image
from PIL import Image
from torch.nn import functional as F
from pl_bolts.models.self_supervised import SimCLR
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

######## Transformations #########
contrast_transforms = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.Resize(size=224),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2, hue=0.1)], p=0.8),
        transforms.GaussianBlur(kernel_size=21),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

######### Loading dataset ##########
freiburg_dataset = datasets.ImageFolder(root='/home/ashutosh/freiburg_groceries_dataset/', transform = contrast_transforms)
train_set, val_set = random_split(freiburg_dataset, [0.85, 0.15])
train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
val_loader = DataLoader(val_set, batch_size=128, shuffle=True)

####### Classifier Model #########
class Classifier(pl.LightningModule):
    def __init__(self, input_dim=2048, hidden_dim=1024, output_dim=25, learning_rate = 0.01):
        super().__init__()
        weight_path = '/home/ashutosh/simclr/tb_logs/simclr_tb_logs/version_8/checkpoints/epoch=96-step=8536.ckpt'
        # weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt'
        simclr = SimCLR.load_from_checkpoint(
            weight_path,
            strict=False,
            optimizer='adam',
        )
        simclr.freeze()
        resnet_encoder = simclr.encoder.eval()
        self.encoder = resnet_encoder
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(self.hidden_dim, self.output_dim, bias=False),
        )
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss  = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss, on_step = False, on_epoch = True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        accuracy = MulticlassAccuracy(num_classes=25).to('cuda')
        acc = accuracy(y_hat, y)
        self.log("validation_loss", loss, on_step = False, on_epoch = True)
        self.log("validation_accuracy", acc, on_step = False, on_epoch = True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = self.learning_rate, weight_decay = 1e-4)

    def forward(self, x):
        feat = self.encoder(x)[-1]
        x = self.model(feat)
        return x

if __name__ == '__main__':

    classifier = Classifier() #.load_from_checkpoint('/home/ashutosh/simclr/tb_logs/classifier_tb_logs/version_0/checkpoints/epoch=2-step=99.ckpt')
    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_checkpoint = ModelCheckpoint(save_last=True, save_top_k=1, monitor="validation_loss")
    callbacks = [model_checkpoint, lr_monitor]

    tb_logger = TensorBoardLogger('tb_logs', name='classifier_tb_logs')
    trainer = pl.Trainer(
        max_epochs = 100,
        accelerator = 'gpu',
        enable_checkpointing = True,
        logger = tb_logger,
        callbacks = callbacks,
    )

    trainer.fit(classifier, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print(model_checkpoint.best_model_path)
    print(model_checkpoint.best_model_score)
