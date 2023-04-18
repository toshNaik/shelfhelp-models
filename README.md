# Shelfhelp Models
### This repository contains the training script for some of the models to improve CV performance in the project [Shelfhelp](https://shivendraagrawal.github.io/projects/shelfhelp/). This uses [pytorch lightning](https://www.pytorchlightning.ai/index.html) and [pytorch lightning bolts](https://github.com/Lightning-Universe/lightning-boltshttps://www.pytorchlightning.ai/bolts). 

## Introduction:
### It is impossible to train an object classifier to identify the sheer number products that are found in a grocery store. Shelfhelp uses a model that converts product images in the scene to feature vectors and then compares it with the feature vector of the user-provided image to find the closest match. The model used for this approach is the encoder part of an autoencoder which leverages training data to learn relevant image features.
### "[Siamese networks](https://towardsdatascience.com/a-friendly-introduction-to-siamese-networks-85ab17522942)" are a class of networks that are well-suited for this use case. This repo (mostly) contains the training scripts for this class of networks. 

## Files and Models:
#### simclr_train.py: Contains the training script for the simclr model. 
[SimCLR](https://arxiv.org/pdf/2002.05709.pdf) is a framework for self-supervised training of models to learn visual representations. Some key takeaways from the paper:
- The authors identified a set of augmentations that result in good performance of the model. 
- They identified that an additional projection head *g(.)* before defining contrastive loss improves the representation of the previous layer. For instance, say *h* and *h\'* are the two representation vectors, applying contrastive loss between *g(h)* and *g(h\')* improves representation quality of *h* and *h\'* than applying contrastive loss directly on *h* and *h\'*.
- They found normalized cross-entropy works best when compared to other contrastive loss functions.

#### byol_train.py: Contains the training script for another siamese like network BYOL. 
[BYOL](https://arxiv.org/pdf/2006.07733.pdf) uses a target and an online network both with very similar architectures. The weights of the target network are a moving average of the online network. The authors achieve good performance without the use of negative pairs for training. 

#### broad_classifier.py Contains code for a classifier model to distinguish between classes of grocery products. 
The model is trained on the Freiburg groceries dataset. It uses a resnet-50 base pretrained using SimCLR followed by a classifier head.