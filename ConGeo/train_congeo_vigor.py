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

from congeo.dataset.vigor import VigorDatasetEval, VigorDatasetTrainConGeo
from congeo.transforms import get_transforms_train_congeo, get_transforms_val
from congeo.utils import setup_system, Logger
from congeo.trainer import train_contrast_congeo
from congeo.evaluate.vigor import evaluate, calc_sim
from congeo.loss import InfoNCE
from congeo.model import TimmModel_ConGeo


@dataclass
class Configuration:
    
    # Model
    model: str = 'convnext_base.fb_in22k_ft_in1k_384'
    
    # Override model image size
    img_size: int = 384
    
    # Training 
    mixed_precision: bool = True
    seed = 1
    epochs: int = 60
    batch_size: int = 32        # keep in mind real_batch_size = 2 * batch_size
    verbose: bool = True
    gpu_ids: tuple = (0,)   # GPU ids for training
    
    
    # Similarity Sampling
    custom_sampling: bool = False   # use custom sampling instead of random
    gps_sample: bool = False        # use gps sampling
    sim_sample: bool = False       # use similarity sampling
    neighbour_select: int = 64     # max selection size from pool
    neighbour_range: int = 128     # pool size for selection
    gps_dict_path: str = "./data/VIGOR/gps_dict_same.pkl"   # gps_dict_cross.pkl | gps_dict_same.pkl
 
    # Eval
    batch_size_eval: int = 16
    eval_every_n_epoch: int = 1      # eval every n Epoch
    normalize_features: bool = True

    # Optimizer 
    clip_grad = 100.                 # None | float
    decay_exclue_bias: bool = False
    grad_checkpointing: bool = False # Gradient Checkpointing
    
    # Loss
    label_smoothing: float = 0.1
    
    # Learning Rate
    lr: float = 0.0001                  # 1 * 10^-4 for ViT | 1 * 10^-1 for CNN
    scheduler: str = "cosine"          # "polynomial" | "cosine" | "constant" | None
    warmup_epochs: int = 1
    lr_end: float = 0.0001             #  only for "polynomial"
    
    # Dataset
    data_folder = "/mnt/vigor"
    same_area: bool = True             # True: same | False: cross
    ground_cutting = 0                 # cut ground upper and lower
   
    # Augment Images
    prob_rotate: float = 0.75          # rotates the sat image and ground images simultaneously
    prob_flip: float = 0.5             # flipping the sat image and ground images simultaneously
    
    # Savepath for model checkpoints
    model_path: str = "./vigor_same_congeo"
    
    # Eval before training
    zero_shot: bool = False  
    
    # Checkpoint to start from
    checkpoint_start = None 
  
    # set num_workers to 0 if on Windows
    num_workers: int = 0 if os.name == 'nt' else 4 
    
    # train on GPU if available
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu' 
    
    # for better performance
    cudnn_benchmark: bool = True
    
    # make cudnn deterministic
    cudnn_deterministic: bool = False
    train_fov: float=180 # train fov (with random shift)
    fov: float=90 # eval fov (with random shift)
    random_fov: bool=False # if True, plase set train_fov to 360

class LossTracker(torch.nn.Module):
    def __init__(self, base_loss_fn):
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.batch_losses = [] 

    def forward(self, query_features, reference_features):
        loss = self.base_loss_fn(query_features, reference_features)
        self.batch_losses.append(loss.item())
        return loss

#-----------------------------------------------------------------------------#
# Train Config                                                                #
#-----------------------------------------------------------------------------#

config = Configuration() 


if __name__ == '__main__':

    def calculate_test_loss(model, query_dataloader, reference_dataloader, loss_function, device):
        model.eval()
        
        query_features_list = []
        reference_features_list = []
        
        with torch.no_grad():
            for img, _ in query_dataloader:
                img = img.to(device)
                query_features_list.append(model(img))
                
            for img, _ in reference_dataloader:
                img = img.to(device)
                reference_features_list.append(model(img))
                
        q_feat = torch.cat(query_features_list, dim=0)
        r_feat = torch.cat(reference_features_list, dim=0)
        
        base_loss = loss_function.base_loss_fn
        
        if isinstance(model, torch.nn.DataParallel):
            logit_scale = model.module.logit_scale.exp()
        else:
            logit_scale = model.logit_scale.exp()
            
        loss = base_loss(q_feat, r_feat, logit_scale)
        return loss.item()


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

    #-----------------------------------------------------------------------------#
    # Model                                                                       #
    #-----------------------------------------------------------------------------#
        
    print("\nModel: {}".format(config.model))


    model = TimmModel_ConGeo(config.model,
                          pretrained=True,
                          img_size=config.img_size,
                          random_fov=config.random_fov)
                          
    data_config = model.get_config()
    print(data_config)
    mean = data_config["mean"]
    std = data_config["std"]
    img_size = config.img_size
    train_fov = config.train_fov
    fov = config.fov # eval FoV

    image_size_sat = (img_size, img_size)

    new_width = img_size*2    
    new_hight = int(((1024 - 2 * config.ground_cutting) / 2048) * new_width)
    img_size_ground = (new_hight, new_width)
    
    # Activate gradient checkpointing
    if config.grad_checkpointing:
        model.set_grad_checkpointing(True)
     
    # Load pretrained Checkpoint    
    if config.checkpoint_start is not None:  
        print("Start from:", config.checkpoint_start)
        model_state_dict = torch.load(config.checkpoint_start)  
        model.load_state_dict(model_state_dict, strict=False)     

    # Data parallel
    print("GPUs available:", torch.cuda.device_count())  
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
            
    # Model to device   
    model = model.to(config.device)

    print("\nImage Size Sat:", image_size_sat)
    print("Image Size Ground:", img_size_ground)
    print("Mean: {}".format(mean))
    print("Std:  {}\n".format(std)) 


    #-----------------------------------------------------------------------------#
    # DataLoader                                                                  #
    #-----------------------------------------------------------------------------#

    # Transforms
    sat_transforms_train1, sat_transforms_train2, ground_transforms_train1, ground_transforms_train2 = get_transforms_train_congeo(image_size_sat,
                                                                   img_size_ground,
                                                                   mean=mean,
                                                                   std=std,
                                                                   fov=train_fov
                                                                   )                            
                                                                   
    # Train
    train_dataset = VigorDatasetTrainConGeo(data_folder=config.data_folder ,
                                      same_area=config.same_area,
                                      transforms_query1=ground_transforms_train1,
                                      transforms_query2=ground_transforms_train2,
                                      transforms_reference1=sat_transforms_train1,
                                      transforms_reference2=sat_transforms_train2,
                                      prob_flip=config.prob_flip,
                                      prob_rotate=config.prob_rotate,
                                      shuffle_batch_size=config.batch_size
                                      )
    
    
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  num_workers=config.num_workers,
                                  shuffle=not config.custom_sampling,
                                  pin_memory=True)
    
    
    # Eval
    sat_transforms_val, ground_transforms_val = get_transforms_val(image_size_sat,
                                                                   img_size_ground,
                                                                   mean=mean,
                                                                   std=std,
                                                                   ground_cutting=config.ground_cutting)


    # Reference Satellite Images Test
    reference_dataset_test = VigorDatasetEval(data_folder=config.data_folder ,
                                              split="test",
                                              img_type="reference",
                                              same_area=config.same_area,  
                                              transforms=sat_transforms_val,
                                              )
    
    reference_dataloader_test = DataLoader(reference_dataset_test,
                                           batch_size=config.batch_size_eval,
                                           num_workers=config.num_workers,
                                           shuffle=False,
                                           pin_memory=True)
    
    
    
    # Query Ground Images Test
    query_dataset_test = VigorDatasetEval(data_folder=config.data_folder ,
                                          split="test",
                                          img_type="query",
                                          same_area=config.same_area,      
                                          transforms=ground_transforms_val,
                                          )
    
    query_dataloader_test = DataLoader(query_dataset_test,
                                       batch_size=config.batch_size_eval,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)
    
    # Evaluation test loss
    val_query_dataset = VigorDatasetEval(
        data_folder=config.data_folder,
        split="test",
        img_type="query",
        same_area=config.same_area,
        transforms=ground_transforms_val
    )
    
    val_reference_dataset = VigorDatasetEval(
        data_folder=config.data_folder,
        split="test",
        img_type="reference",
        same_area=config.same_area,
        transforms=sat_transforms_val
    )
    
    val_query_dataloader = DataLoader(
        val_query_dataset, batch_size=config.batch_size_eval,
        num_workers=config.num_workers, shuffle=False, pin_memory=True
    )
    
    val_reference_dataloader = DataLoader(
        val_reference_dataset, batch_size=config.batch_size_eval,
        num_workers=config.num_workers, shuffle=False, pin_memory=True
    )
    
    
    print("Query Images Test:", len(query_dataset_test))
    print("Reference Images Test:", len(reference_dataset_test))
    

    #-----------------------------------------------------------------------------#
    # GPS Sample                                                                  #
    #-----------------------------------------------------------------------------#
    if config.gps_sample:
        with open(config.gps_dict_path, "rb") as f:
            sim_dict = pickle.load(f)
    else:
        sim_dict = None
    
    #-----------------------------------------------------------------------------#
    # Sim Sample + Eval on Train                                                  #
    #-----------------------------------------------------------------------------#
    
    if config.sim_sample:

        # Query Ground Images Train for simsampling
        query_dataset_train = VigorDatasetEval(data_folder=config.data_folder ,
                                               split="train",
                                               img_type="query",
                                               same_area=config.same_area,      
                                               transforms=ground_transforms_val,
                                               )
            
        query_dataloader_train = DataLoader(query_dataset_train,
                                            batch_size=config.batch_size_eval,
                                            num_workers=config.num_workers,
                                            shuffle=False,
                                            pin_memory=True)
        
        # Reference Satellite Images Train for simsampling
        reference_dataset_train = VigorDatasetEval(data_folder=config.data_folder ,
                                                   split="train",
                                                   img_type="reference",
                                                   same_area=config.same_area,  
                                                   transforms=sat_transforms_val,
                                                   )
        
        reference_dataloader_train = DataLoader(reference_dataset_train,
                                                batch_size=config.batch_size_eval,
                                                num_workers=config.num_workers,
                                                shuffle=False,
                                                pin_memory=True)
            
      
        print("\nQuery Images Train:", len(query_dataset_train))
        print("Reference Images Train (unique):", len(reference_dataset_train))
        
    
    #-----------------------------------------------------------------------------#
    # Loss                                                                        #
    #-----------------------------------------------------------------------------#

    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    info_nce_loss = InfoNCE(loss_function=loss_fn,
                            device=config.device,
                            )
    
    loss_function = LossTracker(info_nce_loss)

    if config.mixed_precision:
        scaler = GradScaler(init_scale=2.**10)
    else:
        scaler = None
        
    #-----------------------------------------------------------------------------#
    # optimizer                                                                   #
    #-----------------------------------------------------------------------------#

    if config.decay_exclue_bias:
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        optimizer_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_parameters, lr=config.lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)


    #-----------------------------------------------------------------------------#
    # Scheduler                                                                   #
    #-----------------------------------------------------------------------------#

    train_steps = len(train_dataloader) * config.epochs
    warmup_steps = len(train_dataloader) * config.warmup_epochs
       
    if config.scheduler == "polynomial":
        print("\nScheduler: polynomial - max LR: {} - end LR: {}".format(config.lr, config.lr_end))  
        scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,
                                                              num_training_steps=train_steps,
                                                              lr_end = config.lr_end,
                                                              power=1.5,
                                                              num_warmup_steps=warmup_steps)
        
    elif config.scheduler == "cosine":
        print("\nScheduler: cosine - max LR: {}".format(config.lr))   
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_training_steps=train_steps,
                                                    num_warmup_steps=warmup_steps)
        
    elif config.scheduler == "constant":
        print("\nScheduler: constant - max LR: {}".format(config.lr))   
        scheduler =  get_constant_schedule_with_warmup(optimizer,
                                                       num_warmup_steps=warmup_steps)
           
    else:
        scheduler = None
        
    print("Warmup Epochs: {} - Warmup Steps: {}".format(str(config.warmup_epochs).ljust(2), warmup_steps))
    print("Train Epochs:  {} - Train Steps:  {}".format(config.epochs, train_steps))
        
        
    #-----------------------------------------------------------------------------#
    # Zero Shot                                                                   #
    #-----------------------------------------------------------------------------#
    if config.zero_shot:
        print("\n{}[{}]{}".format(30*"-", "Zero Shot", 30*"-"))  

        r1_test = evaluate(config=config,
                           model=model,
                           reference_dataloader=reference_dataloader_test,
                           query_dataloader=query_dataloader_test, 
                           ranks=[1, 5, 10],
                           step_size=1000,
                           cleanup=True)
        
        if config.sim_sample:
            r1_train, sim_dict = calc_sim(config=config,
                                          model=model,
                                          reference_dataloader=reference_dataloader_train,
                                          query_dataloader=query_dataloader_train, 
                                          ranks=[1, 5, 10],
                                          step_size=1000,
                                          cleanup=True)
        
    #-----------------------------------------------------------------------------#
    # Shuffle                                                                     #
    #-----------------------------------------------------------------------------#            
    if config.custom_sampling:
        train_dataloader.dataset.shuffle(sim_dict,
                                         neighbour_select=config.neighbour_select,
                                         neighbour_range=config.neighbour_range)
            
    #-----------------------------------------------------------------------------#
    # Train                                                                       #
    #-----------------------------------------------------------------------------#
    start_epoch = 0   
    best_score = 0
    
    history_file = "training_history.csv"
    with open(history_file, mode="w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["step", "epoch", "train_loss", "test_loss", "lr", "recall_1", "recall_5", "recall_10", "recall_top1", "hit_rate"])
    
    global_step = 0

    for epoch in range(1, config.epochs+1):
        
        print("\n{}[Epoch: {}]{}".format(30*"-", epoch, 30*"-"))

        loss_function.batch_losses =  []

        train_loss_average, batch_data =  train_contrast_congeo(config,
                           model,
                           dataloader=train_dataloader,
                           loss_function=loss_function,
                           optimizer=optimizer,
                           scheduler=scheduler,
                           scaler=scaler)
        
        with open(history_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            for b_loss, b_lr in batch_data:
                writer.writerow([global_step, epoch, b_loss, "", b_lr, "", "", "", "", ""])
                global_step += 1
        
        print("Epoch: {}, Train Loss = {:.3f}, Lr = {:.6f}".format(epoch,
                                                                   train_loss_average,
                                                                   optimizer.param_groups[0]['lr']))
        
        # evaluate
        if (epoch % config.eval_every_n_epoch == 0 and epoch != 0) or epoch == config.epochs:
        
            print("\n{}[{}]{}".format(30*"-", "Evaluate", 30*"-"))

            test_loss = calculate_test_loss(
                model=model,
                query_dataloader=val_query_dataloader,
                reference_dataloader=val_reference_dataloader,
                loss_function=loss_function,
                device=config.device
            )
            print("Epoch: {}, Test Loss = {:.3f}".format(epoch, test_loss))
        
            r1_test, test_hit_rate = evaluate(config=config,
                               model=model,
                               reference_dataloader=reference_dataloader_test,
                               query_dataloader=query_dataloader_test, 
                               ranks=[1, 5, 10],
                               step_size=1000,
                               cleanup=True)
            current_epoch_lr = optimizer.param_groups[0]['lr']
            with open(history_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([global_step, 
                                 epoch, 
                                 train_loss_average, 
                                 test_loss, 
                                 current_epoch_lr,
                                 r1_test[0], r1_test[1], r1_test[2], r1_test[3],
                                 test_hit_rate
                                ])
            
            
            if config.sim_sample:
                r1_train, sim_dict = calc_sim(config=config,
                                              model=model,
                                              reference_dataloader=reference_dataloader_train,
                                              query_dataloader=query_dataloader_train, 
                                              ranks=[1, 5, 10],
                                              step_size=1000,
                                              cleanup=True)
            if r1_test[0] > best_score:

                best_score = r1_test[0]

                if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
                    torch.save(model.module.state_dict(), '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, r1_test[0]))
                else:
                    torch.save(model.state_dict(), '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, r1_test[0]))
                

        if config.custom_sampling:
            train_dataloader.dataset.shuffle(sim_dict,
                                             neighbour_select=config.neighbour_select,
                                             neighbour_range=config.neighbour_range)
        
        
                
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        torch.save(model.module.state_dict(), '{}/weights_end.pth'.format(model_path))
    else:
        torch.save(model.state_dict(), '{}/weights_end.pth'.format(model_path))

    # -----------------------------------------------------------------------------#
    # Graph                                                                        #
    # -----------------------------------------------------------------------------#
    
    df_hist = pd.read_csv("training_history.csv")
    df_eval = df_hist.dropna(subset=['test_loss'])


    fig1, ax_loss = plt.subplots(figsize=(12, 6))

    #Loss
    color_loss = 'tab:red'
    ax_loss.set_xlabel('Global Training Steps (Batches)')
    ax_loss.set_ylabel('Loss', color=color_loss)
    line_train = ax_loss.plot(df_hist['step'], df_hist['train_loss'], color=color_loss, alpha=0.25, label='Train Loss (Batch)')
    line_test = ax_loss.plot(df_eval['step'], df_eval['test_loss'], color='tab:orange', marker='o', linestyle='-', label='Test Loss (Epoch)')
    ax_loss.tick_params(axis='y', labelcolor=color_loss)
    ax_loss.grid(True, linestyle=':', alpha=0.5)

    #Learning rate
    ax_lr = ax_loss.twinx()
    color_lr = 'tab:green'
    ax_lr.set_ylabel('Learning Rate', color=color_lr)

    line_lr = ax_lr.plot(df_hist['step'], df_hist['lr'], color=color_lr, linestyle='-', alpha=0.7, label='Learning Rate')

    ax_lr.tick_params(axis='y', labelcolor=color_lr)
    ax_lr.yaxis.get_major_formatter().set_powerlimits((0, 0))

    lines1 = line_train + line_test + line_lr
    labels1 = [l.get_label() for l in lines1]
    ax_loss.legend(lines1, labels1, loc='upper right')
    
    plt.title('Loss Evolution & Learning Rate Schedule')
    fig1.tight_layout()
    plt.savefig("learning_curve_loss_lr.png")

    fig2, ax_perf = plt.subplots(figsize=(10, 6))

    ax_perf.set_xlabel('Global Training Steps (Batches)')
    ax_perf.set_ylabel('Score (%)')

    line_r1 = ax_perf.plot(df_eval['step'], df_eval['recall_1'], color='tab:blue', marker='s', linestyle='-', label='Recall@1')
    line_r5 = ax_perf.plot(df_eval['step'], df_eval['recall_5'], color='teal', marker='^', linestyle='-', label='Recall@5')
    line_r10 = ax_perf.plot(df_eval['step'], df_eval['recall_10'], color='purple', marker='x', linestyle='-', label='Recall@10')
    line_rtop = ax_perf.plot(df_eval['step'], df_eval['recall_top1'], color='navy', marker='d', linestyle='-', label='Recall@top1%')
    line_hit_rate = ax_perf.plot(df_eval['step'], df_eval['hit_rate'], color='magenta', marker='o', linestyle='--', linewidth=2, label='Hit Rate')

    ax_perf.set_ylim(-5, 105)
    ax_perf.grid(True, linestyle=':', alpha=0.5)
    ax_perf.legend(loc='lower right')
    
    plt.title('Evaluation Metrics (Recalls & Hit Rate)')
    fig2.tight_layout()
    plt.savefig("learning_curve_metrics.png")
    