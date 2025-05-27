import os
import torch
import pickle
from torch import optim
import pandas as pd
import numpy as np
from BasicClass import Composition
from torchvision.ops import sigmoid_focal_loss
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import logging
from DeepGlycan import Rnova
from dataset import GenovaDataset
from collator import  GenovaCollator
from prefetcher import DataPrefetcher
from label_generator_comp import  LabelGenerator
from sampler import RnovaBucketBatchSampler

import hydra
import wandb
from omegaconf import DictConfig


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
    'xyl': Composition('C5H10O5') - Composition('H2O'),
}
label_mono = ['H', 'N', 'A', 'G', 'F', 'X']
name2id = {aa:i for i, aa in enumerate(mono_composition)}
label_name2id = {aa:i for i, aa in enumerate(label_mono)}
tokenize_aa_dict = {aa: i for i, aa in enumerate(mono_composition)}
name2mass = {aa: aa_c.mass for i, (aa, aa_c) in enumerate(mono_composition.items())}

def train(model, environment,environment_test, optimizer, scaler, loss_fn, rank, cfg):
    best_correct = 0
    best_val = 0
    for epoch in range(cfg.train.total_epoch):
        total_seq_num = 0
        total_word = 0
        detect_period = 100
        reward = 0
        total_loss = 0
        num_correct = 0
        for i, (model_input, label, label_mask_num) in enumerate(environment,start=1):
            model.train()
            with torch.autograd.set_detect_anomaly(True):
                optimizer.zero_grad(set_to_none=True)
                if dist.is_initialized():
                    pred = model.module.tgt_get(**model_input, label_mask_pad=label_mask_num)

                else: pred = model.tgt_get(**model_input, label_mask_pad=label_mask_num)
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
                print({'train_loss': (total_loss/total_word).item(),
                       'correct_len':( reward/ total_seq_num).item(),
                       'inference_len': (total_word / total_seq_num).item(),
                        'accuracy': num_correct/total_word })
        # '''
        print('start evaluate')
        val_accu = evaluate(model, environment_test)
        if val_accu > best_val:
            torch.save(model.state_dict(),
                   os.path.join(os.getcwd(), cfg.model_path))
            best_val = val_accu
        torch.save(model.state_dict(),
                   os.path.join(os.getcwd(), cfg.model_path+str(epoch)+'.pt'))

def evaluate(model, environment):
    total_seq_num = 0
    total_word = 0
    reward = 0
    total_loss = 0
    num_correct = 0
    for i, (model_input, label, label_mask_num) in enumerate(environment, start=1):
        model.eval()
        if dist.is_initialized():
            pred = model.module.tgt_get(**model_input, label_mask_pad=label_mask_num)
        else:
            pred = model.tgt_get(**model_input, label_mask_pad=label_mask_num)
        reward += label.any(-1).sum()
        num_correct += (
                    torch.argmax(pred[label_mask_num], dim=-1).reshape(-1) == torch.argmax(label[label_mask_num],
                                                                                           dim=-1).reshape(
                -1)).sum()
        total_word += label_mask_num.sum()
        total_seq_num += label_mask_num.size(0)

        print({'val_loss': (total_loss/total_word).item(),
                   'correct_len': (reward / total_seq_num).item(),
                   'inference_len': (total_word / total_seq_num).item(),
                   'val-accuracy': num_correct / total_word})

        return num_correct / total_word

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg:DictConfig):
    train_spec_header = pd.read_csv(cfg.train_spec_header_path)
    train_ds = GenovaDataset(cfg, name2id, spec_header=train_spec_header, dataset_dir_path=cfg.train_dataset_dir)
    collator = GenovaCollator(cfg)
    train_sampler = RnovaBucketBatchSampler(cfg, train_spec_header)
    train_dl = DataLoader(train_ds,batch_sampler=train_sampler,collate_fn=collator,num_workers=4,pin_memory=True)
    train_dl = DataPrefetcher(train_dl,local_rank)
    mass_list = [0] + list(name2mass.values())
    knapsack_mask = dict()
    knapsack_mask['mass'] = np.array(list(name2mass.values()))
    id2mass = {i: m for i, m in enumerate(name2mass.values())}
    model = Rnova(cfg, torch.tensor(mass_list, device=local_rank), id2mass).to(local_rank)#,

    if dist.is_initialized(): model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.train.lr, foreach=False)

    environment = LabelGenerator(cfg, model, train_dl, knapsack_mask, name2mass, name2id, label_name2id)
    scaler = GradScaler()
    loss_fn = torch.nn.CrossEntropyLoss()
    test_spec_header = pd.read_csv(cfg.test_spec_header_path)
    test_ds = GenovaDataset(cfg, name2id, spec_header=test_spec_header, dataset_dir_path=cfg.test_dataset_dir)
    collator = GenovaCollator(cfg)
    test_sampler = RnovaBucketBatchSampler(cfg, test_spec_header)
    test_dl = DataLoader(test_ds, batch_sampler=test_sampler, collate_fn=collator, num_workers=4, pin_memory=True)
    test_dl = DataPrefetcher(test_dl, local_rank)
    environment_test = LabelGenerator(cfg, model, test_dl, knapsack_mask, name2mass, name2id, label_name2id)
    train(model, environment, environment_test, optimizer, scaler, loss_fn, rank, cfg)
    # evaluate(model, environment_test)


if __name__ == '__main__':
    main()
