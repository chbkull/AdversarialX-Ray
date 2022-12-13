#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from src.model import DenseNet121
from src.dataset import ChestXrayDataSet, CLASS_NAMES
import torchvision.transforms as transforms
from PIL import Image

from tqdm import tqdm

from torchattacks import PGD, FGSM, BIM, DeepFool, SparseFool, TPGD


labels = pd.read_csv("data/labels/cleaned.csv")
X, Y = labels.iloc[:, 0], labels.iloc[:, 1]
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.1, shuffle=True, random_state=42)


N_CLASSES = len(CLASS_NAMES)
DATA_DIR = "data/images"
BATCH_SIZE = 1


train_dataset = ChestXrayDataSet(data_dir=DATA_DIR, X=X_train, Y=Y_train,
                                 transform=transforms.Compose([
                                     transforms.Resize(256),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [
                                                          0.229, 0.224, 0.225]),
                                     transforms.RandomHorizontalFlip()
                                 ]))

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=0, pin_memory=True)

test_dataset = ChestXrayDataSet(data_dir=DATA_DIR, X=X_test, Y=Y_test,
                                transform=transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [
                                                         0.229, 0.224, 0.225])
                                ]))

test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                         shuffle=True, num_workers=0, pin_memory=True)


CKPT_TRAINED_PATH = "model-trained.pth"

cudnn.benchmark = True  # Fixed input size, enables tuning for optimal use

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# initialize and load the model
model = DenseNet121(N_CLASSES).to(device)

if os.path.isfile(CKPT_TRAINED_PATH):
    print("=> loading checkpoint")
    checkpoint = torch.load(CKPT_TRAINED_PATH)
    # Load directly into the module else the model gets screwed up
    # https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/15
    model.load_state_dict(checkpoint, strict=True)
    print("=> loaded checkpoint")
else:
    print("=> no checkpoint found")


# switch to evaluate mode
model.eval()


# Adversarial Examples Generator
def adversarial_examples_generator(model, atk_name="pgd"):

    if atk_name == "pgd":
        atk = PGD(model, eps=8/255, alpha=2/255, steps=10, random_start=True)
    if atk_name == "fgsm":
        atk = FGSM(model, eps=8/255)
    elif atk_name == "bim":
        atk = BIM(model, eps=8/255, alpha=2/255, steps=10)
    elif atk_name == "deepfool":
        atk = DeepFool(model, steps=75, overshoot=0.07)
    elif atk_name == "sparsefool":
        atk = SparseFool(model, steps=1, lam=1)
    elif atk_name == "tpgd":
        atk = TPGD(model, eps=8/255, alpha=2/255, steps=10)

    atk.set_normalization_used(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    filenameAndLabel = []

    for i, (x, y, z) in tqdm(enumerate(test_loader)):
        adv_images = atk(x, y)
        adv_images = adv_images.cpu()
        filename = z[0].split("\\")[1].split(".")[0] + "_" + atk_name + ".png"

        MEAN = torch.tensor([0.485, 0.456, 0.406])
        STD = torch.tensor([0.229, 0.224, 0.225])
    
        x = adv_images * STD[:, None, None] + MEAN[:, None, None]
        x = x.numpy().transpose(2, 3, 1, 0).squeeze(3)
        
        # save using PIL
        im = Image.fromarray((x * 255).astype(np.uint8))
        im.save("data/images/{}".format(filename))
        filenameAndLabel.append([filename, CLASS_NAMES[torch.argmax(y)]])

    pd.DataFrame(filenameAndLabel, columns=["Image Index", "Finding Labels"]).to_csv(
        "data/labels_" + atk_name + ".csv", index=False)


if __name__ == "__main__":
    adversarial_examples_generator(model, atk_name="pgd")
    adversarial_examples_generator(model, atk_name="fgsm")
    adversarial_examples_generator(model, atk_name="bim")
    adversarial_examples_generator(model, atk_name="deepfool")
    adversarial_examples_generator(model, atk_name="sparsefool")
    adversarial_examples_generator(model, atk_name="tpgd")


# To combine all the csv files into one

# import os
# maincsv = pd.DataFrame()

# for i in os.listdir('./adv_images/'):
#     m = pd.read_csv('./adv_images/'+i)
#     maincsv = maincsv.append(m)

# print(maincsv.shape)

# m = pd.read_csv('./data/labels/cleaned.csv')

# # Append the first two columns of m to maincsv
# maincsv = maincsv.append(m.iloc[:, :2])

# print(maincsv.shape)

# maincsv.to_csv('./data/labels/adv_images.csv', index=False)