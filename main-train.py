import os
import torch
import pickle
from torch import optim
import pandas as pd
import numpy as np
from data.BasicClass import Composition
from torchvision.ops import sigmoid_focal_loss
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import hydra
from model.DeepGlycan import DeepGlycan
from data.dataset import DGDataset
from data.collator import DGCollator
from data.prefetcher import DataPrefetcher
from data.label_generator_comp import  LabelGenerator
from data.sampler import DGBucketBatchSampler
from omegaconf import DictConfig
from datetime import datetime
currentDateAndTime = datetime.now()

try:
    ngpus_per_node = torch.cuda.device_count()
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
except ValueError:
    rank = 0
    local_rank = "cuda" if torch.cuda.is_available() else "cpu"


'''
if local_rank == 0:
    run = wandb.init(
            name='composition-predict-Gelu-RMSNorm',
            # Set the project where this run will be logged
            project="test-data",
            # Track hyperparameters and run metadata
            config={
                "learning_rate": 1e-4,
            })
'''
mono_composition = {
    'hex': Composition('C6H12O6') - Composition('H2O'),
    'hexNAc': Composition('C8H15O6N') - Composition('H2O'),
    'neuAc': Composition('C11H19O9N') - Composition('H2O'),
    'neuGc': Composition('C11H19O10N') - Composition('H2O'),
    'fuc': Composition('C6H12O5') - Composition('H2O'),
}
label_mono = ['H', 'N', 'A', 'G', 'F']
name2id = {aa:i for i, aa in enumerate(mono_composition)}
label_name2id = {aa:i for i, aa in enumerate(label_mono)}
tokenize_aa_dict = {aa: i for i, aa in enumerate(mono_composition)}
name2mass = {aa: aa_c.mass for i, (aa, aa_c) in enumerate(mono_composition.items())}

def train(model, train_dl,test_dl, optimizer, scaler, loss_fn, rank, cfg, log_file_holder):
    best_correct = 0
    best_val = 0
    
    for epoch in range(cfg.train.total_epoch):
        total_seq_num = 0
        total_word = 0
        detect_period = 100
        reward = 0
        total_loss = 0
        num_correct = 0

        for i, (encoder_input, decoder_input, seq, _, _, glycan_mass,_,_,_, label, label_mask_num) in enumerate(train_dl,start=1):
            model.train()
            with torch.autograd.set_detect_anomaly(True):
                optimizer.zero_grad(set_to_none=True)
                if dist.is_initialized():
                    pred = model.module.forward(encoder_input, decoder_input)

                else: pred = model.forward(encoder_input, decoder_input)
                loss = loss_fn(pred[label_mask_num], torch.argmax(label, dim=-1)[label_mask_num], )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item()
            reward += label.any(-1).sum()
            num_correct += (torch.argmax(pred[label_mask_num], dim=-1).reshape(-1) == torch.argmax(label[label_mask_num], dim=-1).reshape(-1)).sum()
            total_word += label_mask_num.sum()
            total_seq_num += label_mask_num.size(0)
            # print(type((reward / total_word).item()))
            # '''
            if i%detect_period == 0:
                
                log_file_holder.write(
                    f'{datetime.now().strftime("%Y/%m/%d %H:%M:%S")}, '
                    f'epoch:{epoch}, '
                    f'step:{i}, '
                    f'loss:{(total_loss / total_word).item():.6f}, '
                    f'correct_len:{(reward / total_seq_num).item():.6f}, '
                    f'inference_len:{(total_word / total_seq_num).item():.6f}, '
                    f'accuracy:{(num_correct / total_word).item():.6f}\n'
                )
        # '''
        torch.save(model.state_dict(),
                   os.path.join(os.getcwd(), cfg.model_path+str(epoch)+'.pt'))
        print('start evaluate')
        val_accu = evaluate(model, test_dl,log_file_holder)
        if val_accu > best_val:
            torch.save(model.state_dict(),
                   os.path.join(os.getcwd(), cfg.model_path))
            best_val = val_accu
        

def evaluate(model, test_dl, log_file_holder):
    total_seq_num = 0
    total_word = 0
    reward = 0
    total_loss = 0
    num_correct = 0
    for i, (encoder_input, decoder_input, seq, _, _, glycan_mass,_,_,_, label, label_mask_num) in enumerate(test_dl,start=1):
        model.eval()
        if dist.is_initialized():
            pred = model.module.forward(encoder_input, decoder_input)
        else:
            pred = model.forward(encoder_input, decoder_input)
        reward += label.any(-1).sum()
        num_correct += (
                    torch.argmax(pred[label_mask_num], dim=-1).reshape(-1) == torch.argmax(label[label_mask_num],
                                                                                           dim=-1).reshape(
                -1)).sum()
        total_word += label_mask_num.sum()
        total_seq_num += label_mask_num.size(0)
        if i % 100 == 0:
            log_file_holder.write(
                    f'{datetime.now().strftime("%Y/%m/%d %H:%M:%S")}, '
                    f'val_loss:{(total_loss/total_word).item():.6f}, '
                    f'correct_len:{(reward/total_seq_num).item():.6f}, '
                    f'inference_len:{(total_word/total_seq_num).item():.6f}, '
                    f'val_accuracy:{(num_correct/total_word).item():.6f}\n'
                )

    return num_correct / total_word

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg:DictConfig):
    train_spec_header = pd.read_csv(cfg.train_spec_header_path)
    knapsack_mask = dict()
    knapsack_mask['mass'] = np.array(list(name2mass.values()))
    id2mass = {i: m for i, m in enumerate(name2mass.values())}
    label_generator = LabelGenerator(cfg, knapsack_mask, name2mass, name2id, label_name2id)
    train_ds = DGDataset(cfg, name2id, spec_header=train_spec_header, dataset_dir_path=cfg.train_dataset_dir)
    collator = DGCollator(cfg, label_generator)
    train_sampler = DGBucketBatchSampler(cfg, train_spec_header)
    train_dl = DataLoader(train_ds,batch_sampler=train_sampler,collate_fn=collator,num_workers=8,pin_memory=True)
    train_dl = DataPrefetcher(train_dl,local_rank)
    mass_list = [0] + list(name2mass.values())
    
    model = DeepGlycan(cfg, torch.tensor(mass_list, device=local_rank), id2mass).to(local_rank)#,
    # model.load_state_dict(torch.load('save/pglyco-ethcd-simple.pt0.pt', weights_only=True))
    if dist.is_initialized(): model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.train.lr, foreach=False)
    if rank==0: log_file_holder = open('result.log','w',buffering=1)
    else: log_file_holder = None
    scaler = torch.GradScaler("cuda")
    loss_fn = torch.nn.CrossEntropyLoss()
    test_spec_header = pd.read_csv(cfg.test_spec_header_path)
    test_ds = DGDataset(cfg, name2id, spec_header=test_spec_header, dataset_dir_path=cfg.test_dataset_dir)
    test_collator = DGCollator(cfg,label_generator)
    test_sampler = DGBucketBatchSampler(cfg, test_spec_header)
    test_dl = DataLoader(test_ds, batch_sampler=test_sampler, collate_fn=test_collator, num_workers=8, pin_memory=True)
    test_dl = DataPrefetcher(test_dl, local_rank)
    train(model, train_dl, test_dl, optimizer, scaler, loss_fn, rank, cfg,log_file_holder)
    # evaluate(model, test_dl, log_file_holder)


if __name__ == '__main__':
    main()
