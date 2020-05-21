import fnmatch
import math
import os
import sys
import time
from operator import itemgetter

import gc
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms




print('starting test read')
im = Image.open('data/horse.jpg').convert('RGB')
print('img read!')


def affine(theta, img_size, patch_aff):
    grid = F.affine_grid(theta, [theta.shape[0], 3, img_size, img_size])
    affine_result = F.grid_sample(patch_aff, grid).unsqueeze(0)
    return affine_result


class PatchTransformer(nn.Module):
    """PatchTransformer: transforms batch of patches

    Module providing the functionality necessary to transform a batch of patches, randomly adjusting brightness and
    contrast, adding random amount of noise, and rotating randomly. Resizes patches according to as size based on the
    batch of labels, and pads them to the dimension of an image.

    Module providing the functionality necessary to transform a list of patches, put them at the location
    defined by a list of location.
    
    Output the img which is patched

    """

    def __init__(self):
        super(PatchTransformer, self).__init__()

    def forward(self, adv_patch_list, patch_location_list, img_size, img_clean):

        # clamp the range of patch pixel
        for patch_item_index in range(len(adv_patch_list)):
            patch_item = adv_patch_list[patch_item_index]
            # adv_patch_list[patch_item_index] = torch.clamp(patch_item, 0.000001, 0.99999)

        assert len(adv_patch_list) == len(patch_location_list)

        # prepare FloatTensor that will be used in affine
        theta = torch.cuda.FloatTensor(len(adv_patch_list), 2, 3).fill_(0)
        patch_aff_total = torch.cuda.FloatTensor()
        patch_mask_aff_total = torch.cuda.FloatTensor()


        for i in range(len(adv_patch_list)):

            patch_height = adv_patch_list[i].shape[1]
            patch_width = adv_patch_list[i].shape[2]
            assert patch_height <= img_size
            assert patch_width <= img_size

            # left_up_corner
            patch_x = int(patch_location_list[i][0]*img_size)
            patch_y = int(patch_location_list[i][1]*img_size)
            assert patch_x <= img_size
            assert patch_y <= img_size

            # Use affine translation to put the patch to the desired location

            # Prepare theta, affine matrix for i-th
            theta[i, 0, 0] = 1
            theta[i, 0, 1] = 0
            theta[i, 0, 2] = -patch_location_list[i][0]

            theta[i, 1, 0] = 0
            theta[i, 1, 1] = 1
            theta[i, 1, 2] = -patch_location_list[i][1]

            # Pre-process for affine
            # Complete the patch to the same size as img size, so the affine precess won't be wrong
            patch_aff = adv_patch_list[i]
            patch_aff = torch.cat([patch_aff, torch.cuda.FloatTensor(3, img_size-patch_height, patch_width).fill_(0)],
                                  1)
            patch_aff = torch.cat([patch_aff, torch.cuda.FloatTensor(3, img_size, img_size-patch_width).fill_(0)],
                                  2)
            patch_aff = patch_aff.unsqueeze(0)

            # Create a mask to cut-off the black edge.
            # The black edge is caused by Bilinear interpolation when affine.
            patch_mask_aff_ones = torch.ones_like(patch_aff)
            patch_mask_aff_zeros = torch.zeros_like(patch_aff)
            patch_mask_aff = torch.where(patch_aff > 0, patch_mask_aff_ones, patch_mask_aff_zeros)

            # storage the patch_aff and patch_mask_aff to a large multi-channel tensor
            patch_mask_aff_total = torch.cat([patch_mask_aff_total, patch_mask_aff], dim=0)
            patch_aff_total = torch.cat([patch_aff_total, patch_aff], dim=0)

        # use theta and patch_aff_total to affine all patches.
        # Each patch has its own result
        patch_affine2 = affine(theta, img_size, patch_aff_total)

        # Use mask to cut off the black edge
        patch_mask_affine2 = affine(theta, img_size, patch_mask_aff_total)
        patch_mask_affine2_black = torch.cuda.FloatTensor(patch_mask_affine2.size()).fill_(0)
        patch_mask_affine2 = torch.where((patch_mask_affine2 == 1), patch_mask_affine2, patch_mask_affine2_black)

        # apply cut-off
        patch_affine2 = patch_affine2 * patch_mask_affine2
        
        # apply patches to the image
        advs = torch.unbind(patch_affine2, 1)
        for adv in advs:
            img_patched = torch.where((adv == 0), img_clean, adv)  # zzj:注意存在边缘擦除的可能性

        return img_patched


class PatchApplier(nn.Module):
    """PatchApplier: applies adversarial patches to images.

    Module providing the functionality necessary to apply a patch to all detections in all images in the batch.

    """

    def __init__(self):
        super(PatchApplier, self).__init__()

    def forward(self, img_batch, adv_batch):
        advs = torch.unbind(adv_batch, 1)
        for adv in advs:
            # img_batch = torch.where((adv == 0), img_batch, adv)0.000001
            img_batch = torch.where((adv == 0), img_batch, adv)  # zzj:注意存在边缘擦除的可能性
        return img_batch

'''
class PatchGenerator(nn.Module):
    """PatchGenerator: network module that generates adversarial patches.

    Module representing the neural network that will generate adversarial patches.

    """

    def __init__(self, cfgfile, weightfile, img_dir, lab_dir):
        super(PatchGenerator, self).__init__()
        self.yolo = Darknet(cfgfile).load_weights(weightfile)
        self.dataloader = torch.utils.data.DataLoader(InriaDataset(img_dir, lab_dir, shuffle=True),
                                                      batch_size=5,
                                                      shuffle=True)
        self.patchapplier = PatchApplier()
        self.nmscalculator = NMSCalculator()
        self.totalvariation = TotalVariation()

    def forward(self, *input):
        pass
'''

class InriaDataset(Dataset):
    """InriaDataset: representation of the INRIA person dataset.

    Internal representation of the commonly used INRIA person dataset.
    Available at: http://pascal.inrialpes.fr/data/human/

    Attributes:
        len: An integer number of elements in the
        img_dir: Directory containing the images of the INRIA dataset.
        lab_dir: Directory containing the labels of the INRIA dataset.
        img_names: List of all image file names in img_dir.
        shuffle: Whether or not to shuffle the dataset.

    """

    def __init__(self, img_dir, lab_dir, max_lab, imgsize, shuffle=True):
        n_png_images = len(fnmatch.filter(os.listdir(img_dir), '*.png'))
        n_jpg_images = len(fnmatch.filter(os.listdir(img_dir), '*.jpg'))
        n_images = n_png_images + n_jpg_images
        n_labels = len(fnmatch.filter(os.listdir(lab_dir), '*.txt'))
        assert n_images == n_labels, "Number of images and number of labels don't match"
        self.len = n_images
        self.img_dir = img_dir
        self.lab_dir = lab_dir
        self.imgsize = imgsize
        self.img_names = fnmatch.filter(os.listdir(img_dir), '*.png') + fnmatch.filter(os.listdir(img_dir), '*.jpg')
        self.shuffle = shuffle
        self.img_paths = []
        for img_name in self.img_names:
            self.img_paths.append(os.path.join(self.img_dir, img_name))
        self.lab_paths = []
        for img_name in self.img_names:
            lab_path = os.path.join(self.lab_dir, img_name).replace('.jpg', '.txt').replace('.png', '.txt')
            self.lab_paths.append(lab_path)
        self.max_n_labels = max_lab

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        assert idx <= len(self), 'index range error'
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        lab_path = os.path.join(self.lab_dir, self.img_names[idx]).replace('.jpg', '.txt').replace('.png', '.txt')
        image = Image.open(img_path).convert('RGB')
        if os.path.getsize(lab_path):       #check to see if label file contains data.
            label = np.loadtxt(lab_path)
        else:
            label = np.ones([5])

        label = torch.from_numpy(label).float()
        if label.dim() == 1:
            label = label.unsqueeze(0)

        image, label = self.pad_and_scale(image, label)
        transform = transforms.ToTensor()
        image = transform(image)
        label = self.pad_lab(label)
        return image, label

    def pad_and_scale(self, img, lab):
        """

        Args:
            img:

        Returns:

        """
        w,h = img.size
        if w==h:
            padded_img = img
        else:
            dim_to_pad = 1 if w<h else 2
            if dim_to_pad == 1:
                padding = (h - w) / 2
                padded_img = Image.new('RGB', (h,h), color=(127,127,127))
                padded_img.paste(img, (int(padding), 0))
                lab[:, [1]] = (lab[:, [1]] * w + padding) / h
                lab[:, [3]] = (lab[:, [3]] * w / h)
            else:
                padding = (w - h) / 2
                padded_img = Image.new('RGB', (w, w), color=(127,127,127))
                padded_img.paste(img, (0, int(padding)))
                lab[:, [2]] = (lab[:, [2]] * h + padding) / w
                lab[:, [4]] = (lab[:, [4]] * h  / w)
        resize = transforms.Resize((self.imgsize,self.imgsize))
        padded_img = resize(padded_img)     #choose here
        return padded_img, lab

    def pad_lab(self, lab):
        pad_size = self.max_n_labels - lab.shape[0]
        if(pad_size>0):
            padded_lab = F.pad(lab, (0, 0, 0, pad_size), value=1)
        else:
            padded_lab = lab
        return padded_lab

if __name__ == '__main__':
   pass
