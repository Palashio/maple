import torch
import torch.nn as nn
import torch.nn.functional as F


class Dataset(torch.utils.data.Dataset):
    def __init__(self, orig, segment):
        self.orig = orig
        self.segment = segment

    def __len__(self):
        return len(self.orig)

    def __getitem__(self, index):
        image = (sitk.ReadImage(self.orig[index]))
        image = sitk.GetArrayFromImage(image)
        image = torch.from_numpy(normalize(image)).type(torch.float32)

        image = torch.reshape(image, shape=(1, 64, 64, 64))

        mask = (sitk.ReadImage(self.segment[index]))
        mask = sitk.GetArrayFromImage(mask)
        mask = torch.from_numpy(mask).type(torch.float32)

        mask = torch.reshape(mask, shape=(1, 64, 64, 64))

        return image, mask
