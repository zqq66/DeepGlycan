import collections
import os
import re
import sys
import torch
import pickle
from torch import optim
import pandas as pd
import itertools
import csv
import numpy as np
import logging
from itertools import chain
#from torchvision.ops import sigmoid_focal_loss
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from model.DeepGlycan import DeepGlycan
from data.dataset import DGDataset
from data.BasicClass import Composition
from data.collator import DGCollator
from data.prefetcher import DataPrefetcher
from data.inference import Inference_label_comp_o
from data.sampler import DGBucketBatchSampler
from data.peptide_all import peptide_search

import hydra
import json
import gzip
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

mono_composition = {
    'hex': Composition('C6H12O6') - Composition('H2O'),
    'hexNAc': Composition('C8H15O6N') - Composition('H2O'),
    'neuAc': Composition('C11H19O9N') - Composition('H2O'),
    'neuGc': Composition('C11H19O10N') - Composition('H2O'),
    'fuc': Composition('C6H12O5') - Composition('H2O'),
}
glycoCT_dict = {
    'Man': 0,
    'GlcNAc': 1,
    'NeuAc':2,
    'NeuGc': 3,
    'Fuc': 4,
}
aa_dict = {aa:i for i, aa in enumerate(mono_composition)}
# aa_dict['<pad>'] = 0
aa_dict['<bos>'] = len(aa_dict)
tokenize_aa_dict = {i:aa for i, aa in enumerate(mono_composition)}
detokenize_aa_dict = {i: aa.mass for i, aa in enumerate(mono_composition.values())}
detokenize_aa_dict[len(detokenize_aa_dict)] = 0

def evaluate(inference, rank, cfg):
    # run = wandb.init(
    #     name='mouse-male-new'+str(cfg.inference.beam_size),

    #     # Set the project where this run will be logged
    #     project="test-data",
    #     # Track hyperparameters and run metadata
    #     config={
    #         "learning_rate": 1e-4,
    #     })
    correct_comp = []
    mono_comp_dict = {'H':0, 'N':1, 'A':2, 'G':3, 'F':4}
    mono_comp_dict_reversed = {v:k for k,v in mono_comp_dict.items()}
    with open(cfg.test_spec_header_path + cfg.out_put_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        header = ['Spec', 'isotope_shift','predict', 'mass', 'predict mass','mass difference', 'score', 'Pep mass', 'predict pep']
        writer.writerow(header)
        for i, finish_pool in enumerate(inference,start=0):
            for pool in finish_pool.values():
               
                inf_seq = pool.inference_seq
                psm_idx = pool.psm_idx
                mass_diff = pool.mass_diff
                given_pep_mass = pool.pep_mass
                isotope_shift = pool.isotope_shift
                precursor_mass = pool.precursor_mass
                report_mass = pool.report_mass
                pep = pool.pep
                score = torch.mean( torch.stack(pool.score_list))
                inf_seq_r = [mono_comp_dict_reversed[i] for i in inf_seq]
                row = [psm_idx,isotope_shift,''.join(inf_seq_r), precursor_mass, report_mass, mass_diff,score.item(),given_pep_mass,pep]
                # inf_seq_r = collections.Counter(inf_seq_r)
                # inf_seq_str = ''.join(f'{key}:{count} ' for key, count in inf_seq_r.items())                
                writer.writerow(row)

                # wandb.log({'accuracy_comp':sum(correct_comp)/len(correct_comp)})

    return correct_comp
# def main_wrapper(beam_size):
@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg:DictConfig):
    # start from mgf file, fasta_file
    test_spec_header = pd.read_csv(cfg.test_spec_header_path)
    test_ds = DGDataset(cfg, name2id, spec_header=test_spec_header, dataset_dir_path=cfg.test_dataset_dir)
    test_collator = DGCollator(cfg,label_generator)
    test_sampler = DGBucketBatchSampler(cfg, test_spec_header)
    test_dl = DataLoader(test_ds, batch_sampler=test_sampler, collate_fn=test_collator, num_workers=8, pin_memory=True)
    test_dl = DataPrefetcher(test_dl, local_rank)

    mass_list = [0]+list(detokenize_aa_dict.values())[:-1]

    model = DeepGlycan(cfg, torch.tensor(mass_list,device=local_rank), detokenize_aa_dict).to(local_rank)
    new_state_dict = {}
    model_path='save/pglyco-ethcd-simple.pt' # pglyco-scehcd-plant.pt'##save/ethcd-glycan-only-rat-structured9.pt
    state_dict = torch.load(model_path, weights_only=True,
            map_location='cuda:0')  # rl-best-pos9.pt
    for key in state_dict.keys():
        new_key = key.replace('module.', '')  # Remove the 'module.' prefix
        new_state_dict[new_key] = state_dict[key]
    model.load_state_dict(new_state_dict)
    logging.info('model loaded')
    if dist.is_initialized(): model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    # optimizer = Lion(params=model.parameters(),lr=cfg.test.lr,weight_decay=cfg.test.weight_decay)
    knapsack_mask = dict()
    knapsack_mask['mass'] = np.array(list(detokenize_aa_dict.values()))[:-1]
    knapsack_mask['aa_composition'] =  np.array(list(tokenize_aa_dict.keys()))
    inference = Inference_label_comp_o(cfg, model, test_dl, aa_dict, tokenize_aa_dict, detokenize_aa_dict, knapsack_mask)
    evaluate( inference, rank, cfg)
    logging.info('evaluation done')

    

if __name__ == '__main__':
    # beam_size = int(sys.argv[1])
    # main_wrapper(beam_size)
    main()
