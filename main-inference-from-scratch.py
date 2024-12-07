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
from BasicClass import Composition
#from torchvision.ops import sigmoid_focal_loss
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from GlycopepECHO import Rnova
from dataset import GenovaDataset
from collator import GenovaCollator
from prefetcher import DataPrefetcher
from inference import Inference_label_comp_o
from sampler import RnovaBucketBatchSampler

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
    'Xyl': 5
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
        header = ['Spec', 'isotope_shift','predict', 'label mass', 'predict mass','mass correct','mass difference', 'score', 'given pep mass', 'predict pep']
        writer.writerow(header)
        for i, finish_pool in enumerate(inference,start=0):
            for pool in finish_pool.values():
                # print('finish_pool', finish_pool)
                # (inf_seq, label_seq, psm_idx) = pool
                # inf_seq = inf_seq.cpu().tolist()[1:]
                inf_seq = pool.inference_seq
                psm_idx = pool.psm_idx
                mass_diff = pool.mass_diff
                given_pep_mass = pool.rank
                isotope_shift = pool.isotope_shift
                glycan_mass = pool.precursor_mass
                report_mass = pool.report_mass
                score = torch.mean(torch.stack(pool.score_list))
                inf_seq_r = [mono_comp_dict_reversed[i] for i in inf_seq]
                inf_seq_r = collections.Counter(inf_seq_r)
                inf_seq_str = ''.join(f'{key}:{count} ' for key, count in inf_seq_r.items())
                inf_seq = [str(i) for i in inf_seq]

                row = [psm_idx,isotope_shift,inf_seq_str, glycan_mass, report_mass]
                inf_seq.sort()
                print(psm_idx)
               
                if mass_diff <= 0.05:
                    row.append(1)
                    correct_comp.append(1)
                else:
                    row.append(0)
                    correct_comp.append(0)
                print('inf_seq', inf_seq, sum(correct_comp)/len(correct_comp), mass_diff)
                row.append(mass_diff)
                row.append(score.item())
                row.append(given_pep_mass)
                writer.writerow(row)

                # wandb.log({'accuracy_comp':sum(correct_comp)/len(correct_comp)})

    return correct_comp
# def main_wrapper(beam_size):
@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg:DictConfig):
    train_spec_header = pd.read_csv(cfg.test_spec_header_path)
    # print(tra)
    # import pdb;pdb.set_trace()
    train_spec_header[['mass','pep mass given','Glycan mass']] = train_spec_header[['mass','pep mass given','Glycan mass']].astype(float)
    train_spec_header[['Node Number','MSGP Datablock Pointer','MSGP Datablock Length']] = train_spec_header[['Node Number','MSGP Datablock Pointer','MSGP Datablock Length']].astype(int)
    train_spec_header = train_spec_header[train_spec_header['Glycan mass']>0]
    train_ds = GenovaDataset(cfg, aa_dict, spec_header=train_spec_header, dataset_dir_path=cfg.test_dataset_dir)
    collator = GenovaCollator(cfg)
    train_sampler = RnovaBucketBatchSampler(cfg, train_spec_header)
    train_dl = DataLoader(train_ds,batch_sampler=train_sampler,collate_fn=collator,num_workers=8,pin_memory=True)
    train_dl = DataPrefetcher(train_dl,local_rank)

    mass_list = [0]+list(detokenize_aa_dict.values())[:-1]

    model = Rnova(cfg, torch.tensor(mass_list,device=local_rank), detokenize_aa_dict).to(local_rank)
    new_state_dict = {}
    model_path='save/ethcd-all-20.pt9.pt'#ethcd-all-20.pt9.pt' #save/ethcd-glycan-only-rat-structured9.pt
    state_dict = torch.load(model_path, weights_only=True,
            map_location='cuda:0')  # rl-best-pos9.pt
    for key in state_dict.keys():
        new_key = key.replace('module.', '')  # Remove the 'module.' prefix
        new_state_dict[new_key] = state_dict[key]
    model.load_state_dict(new_state_dict)
    print('model loaded ', model_path)
    if dist.is_initialized(): model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    # optimizer = Lion(params=model.parameters(),lr=cfg.test.lr,weight_decay=cfg.test.weight_decay)
    knapsack_mask = dict()
    knapsack_mask['mass'] = np.array(list(detokenize_aa_dict.values()))[:-1]
    knapsack_mask['aa_composition'] =  np.array(list(tokenize_aa_dict.keys()))
    inference = Inference_label_comp_o(cfg, model, train_dl, aa_dict, tokenize_aa_dict, detokenize_aa_dict, knapsack_mask)
    #loss_fn = sigmoid_focal_loss#FocalLoss(alpha=0.25, ) torch.nn.MSELoss()
    tar_more_peak_psm = evaluate( inference, rank, cfg)
    # mask = train_spec_header['Spec Index'].isin(list(tar_more_peak_psm))
    # canbeconverted = train_spec_header[mask]
    # canbeconverted.to_csv(cfg.test_dataset_dir+'/test_branch8_path.csv', index=False)
    # main()

if __name__ == '__main__':
    # beam_size = int(sys.argv[1])
    # main_wrapper(beam_size)
    main()
