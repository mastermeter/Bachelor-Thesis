import copy
import os
import random
import time
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


class SwissDatasetTrainConGeo(Dataset):

    def __init__(self, df, data_folder,
                 transforms_query1,
                 transforms_query2,
                 transforms_reference1,
                 transforms_reference2,
                 ground_cutting=0,
                 prob_flip=0.0,
                 prob_rotate=0.0,
                 debug=True,

                ):

        self.df = df.reset_index(drop=True)
        self.data_folder = data_folder

        self.transforms_query1 = transforms_query1
        self.transforms_query2 = transforms_query2
        self.transforms_reference1 = transforms_reference1
        self.transforms_reference2 = transforms_reference2

        self.ground_cutting = ground_cutting
        self.prob_flip = prob_flip
        self.prob_rotate = prob_rotate

        self.debug = debug

        self.debug_folder = "./debug_transforms"
        if self.debug and not os.path.exists(self.debug_folder):
            os.makedirs(self.debug_folder)

    def align_and_crop_ground(self, img_pil, heading):

        w, h = img_pil.size

        shift_pixels= int((heading % 360) / 360.0 * w)
        if shift_pixels > 0:
            img_np = np.array(img_pil)
            img_np= np.roll(img_np, shift=-shift_pixels, axis=1)
            img_pil = Image.fromarray(img_np)
        
        if self.ground_cutting > 0 and self.ground_cutting < h:
            img_pil = img_pil.crop((0,0, w, h - self.ground_cutting))
        
        return img_pil


    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        aerial_path = os.path.join(self.data_folder, row['aerial_path'])
        ground_path = os.path.join(self.data_folder, row['ground_path'])
        heading = float(row["heading"])
        image_id = row['image_id']

        img_aerial = Image.open(aerial_path).convert('RGB')
        img_ground_raw = Image.open(ground_path).convert('RGB')

        img_ground = self.align_and_crop_ground(img_ground_raw, heading)

        q1 = self.transforms_query1(img_ground)
        q2 = self.transforms_query2(img_ground.copy())

        r1 = self.transforms_reference1(img_aerial)
        r2 = self.transforms_reference2(img_aerial.copy())

        if self.debug and idx < 5:
            try:
                TF.to_pil_image(q1).save(os.path.join(self.debug_folder, f"id_{image_id}_tensor_q1_augmented.png"))
                TF.to_pil_image(q2).save(os.path.join(self.debug_folder, f"id_{image_id}_tensor_q2_augmented.png"))
                TF.to_pil_image(r1).save(os.path.join(self.debug_folder, f"id_{image_id}_tensor_r1_augmented.png"))
            except Exception:
                pass

        return [q1, q2, r1, r2]
    
class SwissDatasetEval(Dataset):
    def __init__(self, df, data_folder, img_type, transforms, ground_cutting=0):

        self.df = df.reset_index(drop=True)
        self.data_folder = data_folder
        self.img_type = img_type
        self.transforms = transforms
        self.ground_cutting = ground_cutting
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        if self.img_type == "reference":
            img_path = os.path.join(self.data_folder, row["aerial_path"])
            img = Image.open(img_path).convert('RGB')
        else :
            img_path = os.path.join(self.data_folder, row["ground_path"])
            heading = float(row['heading'])
            img_raw = Image.open(img_path).convert('RGB')

            w, h = img_raw.size

            shift_pixels= int((heading % 360) / 360.0 * w)
            if shift_pixels > 0:
                img_np = np.array(img_raw)
                img_np= np.roll(img_np, shift=-shift_pixels, axis=1)
                img_raw = Image.fromarray(img_np)
            
            if self.ground_cutting > 0 and self.ground_cutting < h:
                img_raw = img_raw.crop((0,0, w, h - self.ground_cutting))
            
            img = img_raw
            

        img_tensor = self.transforms(img)
        return img_tensor, idx

