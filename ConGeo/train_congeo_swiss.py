import os
import time
import math
import shutil
import sys
import torch
import pickle
import csv
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, get_cosine_schedule_with_warmup

from congeo.dataset.swiss import TODO
from congeo.transforms import get_transforms_train_congeo, get_transforms_val
from congeo.utils import setup_system, Logger
from congeo.trainer import train_contrast_congeo
from congeo.evaluate.vigor import evaluate, calc_sim
from congeo.loss import InfoNCE
from congeo.model import TimmModel_ConGeo


@dataclass
class Configuration:

    model: str = 'convnext_base.fb_in22k_ft_in1k_384'

    img_size: int = 384

    # Training
    mixed_precision: bool = True
    seed = 1
    epoch: int = 60
    batch_size: int = 32
    verbose: bool = True

    # Eval
    batch_size: int = 16
    eval_every_n_epoch: int = 1
    normalize_features: bool = True

    # Optimizer
    clip_grad = 100.
    decay_exclue_bias: bool = False
    grad_checkpointing: bool = False

    # Loss
    label_smoothing: float = 0.1

    # Learning Rate
    lr: float = 0.0001
    scheduler: str = "cosine"
    warmup_epochs : int = 1
    lr_end: float = 0.0001 #not used with scheduler = "cosine"

    data_folder = "/mnt/swiss_dataset"
    
    #Augment Images
    prob_rotate: float = 0.75
    prob_flip: float = 0.5      # Same condition as Vigor

    model_path: str = "./swiss_congeo"

    num_workers: int = 0 if os.name == 'nt' else 4

    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    cudnn_benchmark: bool = True

    cudnn_deterministic: bool = False
    train_fov: float = 180
    fov: float = 90
    random_fov: bool = False


config = Configuration()

if __name__ == '__main__':

    model_path = "{}/{}/{}".format(config.model_path,
                                       config.model,
                                       time.strftime("%H%M%S"))

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    shutil.copyfile(os.path.basename(__file__), "{}/train.py".format(model_path))

    # Redirect print to both console and log file
    sys.stdout = Logger(os.path.join(model_path, 'log.txt'))

    setup_system(seed=config.seed,
                 cudnn_benchmark=config.cudnn_benchmark,
                 cudnn_deterministic=config.cudnn_deterministic)
    
    # Model

    print("\nModel: {}".format(config.model))

    model = TimmModel_ConGeo(config.model,
                          pretrained=True,
                          img_size=config.img_size)
    
    data_config = model.get_config()
    print(data_config)
    mean = data_config["mean"]
    std = data_config["std"]
    img_size = config.img_size
    train_fov = config.train_fov
    fov = config.fov

    image_size_sat = (img_size, img_size)

    new_width = img_size*2
    new_height = int(((1024 - 2 * config.ground_cutting) / 2048) * new_width)
    img_size_ground = (new_height, new_width)

    model = model.to(config.device)

    print("\nImage Size Sat:", image_size_sat)
    print("Image Size Ground:", img_size_ground)
    print("Mean: {}".format(mean))
    print("Std:  {}\n".format(std)) 