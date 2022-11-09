import os
import torch
from torch.utils.data import Dataset
from PIL import Image

CLASS_NAMES = [ "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia",
    "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]

def binary_encoding(indices, classes):
    x = [0] * classes
    if indices is not None:
        for index in indices:
            x[index] = 1
    return x

class ChestXrayDataSet(Dataset):
    def __init__(self, data_dir, X, Y, transform=None):
        image_names = []
        labels = []

        for image_name, label in zip(X, Y):
            res = os.path.join(data_dir, image_name)

            # Only using subset, skip over images not downloaded
            if not os.path.exists(res):
                continue

            image_names.append(res)

            if label == "No Finding":
                indices = None
            else:
                indices = [CLASS_NAMES.index(l) for l in label.split('|')]

            labels.append(binary_encoding(indices, 14))

        self.image_names = image_names
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.image_names)