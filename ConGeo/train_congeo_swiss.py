import csv
import math
import os
import pickle
import shutil
import sys
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import (get_constant_schedule_with_warmup,
                          get_cosine_schedule_with_warmup,
                          get_polynomial_decay_schedule_with_warmup)

from congeo.dataset.swiss import SwissDatasetEval, SwissDatasetTrainConGeo
from congeo.evaluate.vigor import calc_sim, evaluate
from congeo.loss import InfoNCE
from congeo.model import TimmModel_ConGeo
from congeo.trainer import train_contrast_congeo
from congeo.transforms import get_transforms_train_congeo, get_transforms_val
from congeo.utils import Logger, setup_system


@dataclass
class Configuration:

    model: str = 'convnext_base.fb_in22k_ft_in1k_384'
    img_size: int = 384

    # Training
    mixed_precision: bool = True
    seed = 1
    epochs: int = 60
    batch_size: int = 32
    verbose: bool = True

    # Eval
    batch_size_eval: int = 16
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
    ground_cutting = 60         # Cut only ground 
    
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


class LossTracker(torch.nn.Module):
    def __init__(self, base_loss_fn):
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.batch_losses = []
    
    def forward(self, query_features, reference_features, *args, **kwargs):
        loss = self.base_loss_fn(query_features, reference_features, *args, **kwargs)
        self.batch_losses.append(loss.item())
        return loss

def calculate_val_loss(model, dataloader, loss_function, device):
    
    model.eval()

    sums = {"total": 0.0, "l1_sol_sat": 0.0, "l2_sol_sol" : 0.0, "l3_sat_sat" : 0.0, "l4_sat_sol" : 0.0}
    steps = 0

    with torch.no_grad():
        for batch in dataloader:
            q1, q2, r1, r2 = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)
            
            features_q1, features_q2 = model(q1), model(q2)
            features_r1, features_r2 = model(r1), model(r2)

            logit_scale = model.module.logit_scale.exp() if isinstance(model, torch.nn.DataParallel) else model.logit_scale.exp()

            l1 = loss_function.base_loss_fn(features_q1, features_r1, logit_scale)
            l2 = loss_function.base_loss_fn(features_q1, features_q2, logit_scale)
            l3 = loss_function.base_loss_fn(features_r1, features_r2, logit_scale)
            l4 = loss_function.base_loss_fn(features_r1, features_q2, logit_scale)

            batch_loss = l1 + 0.5*l2 + 0.5*l3 + 0.25*l4

            sums["total"] += batch_loss.item()
            sums["l1_sol_sat"] += l1.item()
            sums["l2_sol_sol"] += l2.item()
            sums["l3_sat_sat"] += l3.item()
            sums["l4_sat_sol"] += l4.item()
            steps += 1
    if steps > 0:
            return {k: v / steps for k, v in sums.items()}
    return {k: 0.0 for k in sums.keys()}

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

    #--------------------------------------------------#
    # Data splitting                                   #
    #--------------------------------------------------#

    csv_path = os.path.join(config.data_folder, "metadata.csv")
    df_metadata = pd.read_csv(csv_path)

    df_train, df_temp = train_test_split(df_metadata, test_size=0.40, random_state=config.seed)
    df_val, df_test = train_test_split(df_temp, test_size=0.50, random_state=config.seed)

    print("Dataset split :")
    print(f" - Train        : {len(df_train)}")
    print(f" - Validation   : {len(df_val)}")
    print(f" - Test         : {len(df_test)}")


    #--------------------------------------------------#
    # Dataloaders                                      #
    #--------------------------------------------------#

    sat_train1, sat_train2, ground_train1, ground_train2 = get_transforms_train_congeo(
        image_size_sat, img_size_ground, mean=mean, std=std, fov=config.train_fov
    )
    sat_val, ground_val = get_transforms_val(
        image_size_sat, img_size_ground, mean=mean, std=std, ground_cutting=config.ground_cutting
    )


    # Training Dataloader
    train_dataset = SwissDatasetTrainConGeo(df=df_train, data_folder=config.data_folder,
                                            transforms_query1=ground_train1, transforms_query2=ground_train2,
                                            transforms_reference1=sat_train1, transforms_reference2=sat_train2,
                                            ground_cutting=config.ground_cutting,
                                            prob_flip=config.prob_flip, prob_rotate=config.prob_rotate)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=True, pin_memory=True)

    # Validation Dataloader
    val_loss_dataset = SwissDatasetTrainConGeo(df=df_val, data_folder=config.data_folder,
                                               transforms_query1=ground_val, transforms_query2=ground_val,
                                               transforms_reference1=sat_val, transforms_reference2=sat_val,
                                               ground_cutting=config.ground_cutting,
                                               prob_flip=0.0, prob_rotate=0.0)
    val_loss_dataloader = DataLoader(val_loss_dataset, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=False, pin_memory=True)

    # Test Dataloader
    query_dataset_test = SwissDatasetEval(df=df_test, data_folder=config.data_folder, img_type="query", transforms=ground_val, ground_cutting=config.ground_cutting)
    reference_dataset_test = SwissDatasetEval(df=df_test, data_folder=config.data_folder, img_type="reference", transforms=sat_val)
    
    query_dataloader_test = DataLoader(query_dataset_test, batch_size=config.batch_size_eval, num_workers=config.num_workers, shuffle=False, pin_memory=True)
    reference_dataloader_test = DataLoader(reference_dataset_test, batch_size=config.batch_size_eval, num_workers=config.num_workers, shuffle=False, pin_memory=True)

    #--------------------------------------------------#
    # Loss, Optimizer, Scheduler                       #
    #--------------------------------------------------#

    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    info_nce_loss = InfoNCE(loss_function=loss_fn, device=config.device)
    loss_function = LossTracker(info_nce_loss)

    scaler = GradScaler(init_scale=2.**10) if config.mixed_precision else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    train_steps = len(train_dataloader) * config.epochs
    warmup_steps = len(train_dataloader) * config.warmup_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_training_steps=train_steps, num_warmup_steps=warmup_steps)

    #--------------------------------------------------#
    # Training                                         #
    #--------------------------------------------------#

    history_file = os.path.join(model_path, "swiss_training_detailed_history.csv")
    with open(history_file, mode="w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_loss_avg", "lr",
            "val_loss_total", "val_l1_sol_sat", "val_l2_sol_sol", "val_l3_sat_sat", "val_l4_sat_sol",
            "recall_1", "recall_5", "recall_10", "hit_rate"
        ])
    
    best_score = 0.0

    for epoch in range(1, config.epochs + 1):
        print("\n{}[Epoch: {}]{}".format(30*"-", epoch, 30*"-"))
        loss_function.batch_losses = []

        train_loss_average, batch_data = train_contrast_congeo(
            config, model, dataloader=train_dataloader, loss_function=loss_function,
            optimizer=optimizer, scheduler=scheduler, scaler=scaler
        )

        if epoch % config.eval_every_n_epoch == 0:
            print(f"\n{30*'-'}[Detailed Evaluation]{30*'-'}")

            val_losses = calculate_val_loss(model, val_loss_dataloader, loss_function, config.device)
            
            print(f"Epoch {epoch} - Val Total Loss: {val_losses['total']:.4f}")
            print(f"  └─ L1 (Sol-Sat): {val_losses['l1_sol_sat']:.4f} | L2 (Sol-Sol): {val_losses['l2_sol_sol']:.4f}")
            
            r1_test, test_hit_rate = evaluate(
                config=config, model=model, reference_dataloader=reference_dataloader_test,
                query_dataloader=query_dataloader_test, ranks=[1, 5, 10], step_size=1000, cleanup=True
            )
            
            current_lr = optimizer.param_groups[0]['lr']
            with open(history_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch, train_loss_average, current_lr,
                    val_losses['total'], val_losses['l1_sol_sat'], val_losses['l2_sol_sol'], val_losses['l3_sat_sat'], val_losses['l4_sat_sol'],
                    r1_test[0], r1_test[1], r1_test[2], test_hit_rate
                ])

            if r1_test[0] > best_score:
                best_score = r1_test[0]
                save_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
                torch.save(save_dict, '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, r1_test[0]))
    
    #--------------------------------------------------#
    # Graphs                                           #
    #--------------------------------------------------#


    df_hist = pd.read_csv(history_file)

    fig1, ax_loss = plt.subplots(figsize=(12, 6))
    ax_loss.set_xlabel('Epochs')
    ax_loss.set_ylabel('Loss', color='tab:red')
    l_train = ax_loss.plot(df_hist['epoch'], df_hist['train_loss_avg'], color='tab:red', alpha=0.5, label='Train Loss')
    l_test = ax_loss.plot(df_hist['epoch'], df_hist['val_loss_total'], color='tab:orange', marker='o', label='Val Loss')
    ax_loss.grid(True, linestyle=':', alpha=0.5)

    ax_lr = ax_loss.twinx()
    ax_lr.set_ylabel('Learning Rate', color='tab:green')
    l_lr = ax_lr.plot(df_hist['epoch'], df_hist['lr'], color='tab:green', alpha=0.7, label='LR')
    
    lines = l_train + l_test + l_lr
    ax_loss.legend(lines, [l.get_label() for l in lines], loc='upper right')
    plt.title('Swiss Dataset: Loss & LR Evolution')
    plt.savefig(os.path.join(model_path, "swiss_learning_curve_loss_lr.png"))

    fig2, ax_perf = plt.subplots(figsize=(10, 6))
    ax_perf.set_xlabel('Epochs')
    ax_perf.set_ylabel('Score (%)')
    ax_perf.plot(df_hist['epoch'], df_hist['recall_1'], color='tab:blue', marker='s', label='Recall@1')
    ax_perf.plot(df_hist['epoch'], df_hist['recall_5'], color='teal', marker='^', label='Recall@5')
    ax_perf.plot(df_hist['epoch'], df_hist['recall_10'], color='purple', marker='x', label='Recall@10')
    ax_perf.plot(df_hist['epoch'], df_hist['hit_rate'], color='magenta', marker='o', linestyle='--', label='Hit Rate')
    ax_perf.set_ylim(-5, 105)
    ax_perf.grid(True, linestyle=':', alpha=0.5)
    ax_perf.legend(loc='lower right')
    plt.title('Swiss Dataset: Evaluation Metrics')
    plt.savefig(os.path.join(model_path,"swiss_learning_curve_metrics.png"))