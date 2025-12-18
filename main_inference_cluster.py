import os
import re
import torch
import ast
from pathlib import Path
import pandas as pd
import csv
import numpy as np
import logging
from glob import glob
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from model.DeepGlycan import DeepGlycan
from data.dataset import DGDataset
from data.BasicClass import Composition,Residual_seq
from data.collator import DGCollator
from data.prefetcher import DataPrefetcher
from data.inference import OptimisedInference
from data.preprocess_dataset_optimized import GraphGeneratorOptimized
from data.preprocess_dataset import GraphGenerator
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

import hydra
import json
import gzip
from hydra.utils import to_absolute_path
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
detokenize_aa_dict = {i:aa.mass for i, aa in enumerate(mono_composition.values())}
detokenize_aa_dict[len(detokenize_aa_dict)] = 0
def read_mgf(
    file_path,
    mode,
):
    
    raw_mgf_blocks = []
    for file in glob(os.path.join(file_path, '*.mgf')):
        print('file', file)
        with open(file) as f:
            for line in f:
                if line.startswith('BEGIN IONS'):
                    product_ions_moverz = []
                    product_ions_intensity = []
                    mz = None
                    z = None
                    scan = None
                elif line.startswith('PEPMASS'):
                    mz = float(re.split(r'=|\r|\n|\s', line)[1])
                elif line.startswith('CHARGE'):
                    z = int(re.search(r'CHARGE=(\d+)\+', line).group(1))
                elif line.startswith('TITLE'):
                    scan_pattern = r'scan=(\d+)'
                    m_ = re.search(scan_pattern, line)
                    scan = m_.group(1) if m_ else None
                elif line and line[0].isdigit():
                    # Robust split on whitespace (some MGF writers use multiple spaces/tabs)
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        mz_i, I_i = parts[0], parts[1]
                        product_ions_moverz.append(float(mz_i))
                        product_ions_intensity.append(float(I_i))
                elif line.startswith('END IONS'):
                    # Basic sanity
                    if mz is None or z is None or scan is None:
                        continue
                    mz_arr = np.asarray(product_ions_moverz, dtype=float)
                    I_arr = np.asarray(product_ions_intensity, dtype=float)
                    rawfile = file.split('.')[0].split('/')[-1] + '.'
                    neutral_precursor_mass = mz * z - (z-1) * Composition('proton').mass
                    raw_mgf_blocks.append(
                        (
                            rawfile + scan,
                            mz_arr,
                            I_arr,
                            neutral_precursor_mass,
                            z,
                            mode,
                        )
                    )
    return raw_mgf_blocks

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
    rows = [['Spec', 'isotope_shift','predict', 'mass', 'predict mass','mass difference', 'score_list', 'Pep mass', 'predict pep']]
    for i, finish_pool in enumerate(inference,start=0):
        for pool in finish_pool.values():
            # if pool.mass_diff > 0.05:
            #     continue
            inf_seq = pool.inference_seq
            psm_idx = pool.psm_idx
            mass_diff = pool.mass_diff
            if mass_diff > cfg.mass_diff_threshold:
                continue
            given_pep_mass = pool.pep_mass
            isotope_shift = pool.isotope_shift
            precursor_mass = pool.precursor_mass
            report_mass = pool.report_mass
            pep = pool.pep
            score =[i.item() for i in pool.score_list]
            inf_seq_r = [mono_comp_dict_reversed[i] for i in inf_seq]
            # print(psm_idx,'final',''.join(inf_seq_r), sum(score), pool.mass_diff, isotope_shift)
            row = [psm_idx,isotope_shift,''.join(inf_seq_r), precursor_mass, report_mass, mass_diff,score,given_pep_mass,pep]        
            rows.append(row)

    with open(Path(to_absolute_path(Path(cfg.out_dir) / cfg.out_put_file)), mode='w', newline='') as file:
        csv.writer(file).writerows(rows)
    logging.info('number of spectrum inferenced on test set'+str(len(rows)-1))

    return correct_comp


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg:DictConfig):
    psms = []
    spectrum = read_mgf(cfg.out_dir, cfg.mode.lower())
    scan_ids, m_over_zs, intensities, precursor_masses, charges, modes = zip(*spectrum)
    cluster_psm_path = Path(to_absolute_path(cfg.out_dir)) / "clustered_peptide_search.csv"
    cluster_psm = pd.read_csv(cluster_psm_path)

    # df_path = "/home/q359zhan/olinked/data/ethcd/heart/glycan/heart-y4-precursor-calibrated-inference.csv"  # change if needed
    # df = pd.read_csv(df_path)
    # df = df[df['mass difference'] < 0.02]
    for idx, row in tqdm(cluster_psm.iterrows(), total=cluster_psm.shape[0]):
        target_peptides2mass = dict()
        top3_pep = ast.literal_eval(row['top3_pep'])
        scans = ast.literal_eval(row['scan'])
        # print(scans, top3_pep)
        for j, s in enumerate(top3_pep):
            target_peptides2mass[s] = round(Residual_seq(s).mass + Composition('H2O').mass + Composition('proton').mass,5)
        
        if len(target_peptides2mass) > 0:
            graph_gen = GraphGenerator(target_peptides2mass,cfg.inference.isotope)
            for j in scans:
                if j in scan_ids: #and j in df['Spec'].to_list():
                    # s = df[df['Spec']==j]['predict pep'].to_list()[0]
                    # target_peptides2mass[s]=round(Residual_seq(s).mass + Composition('H2O').mass + Composition('proton').mass,5)
                    idx = scan_ids.index(j)
                    x = 20 + 10 * torch.rand(1)  # [20.0, 30.0)
                    decoy_value = x.item()
                    psms += graph_gen(scan_ids[idx],m_over_zs[idx],intensities[idx],precursor_masses[idx],charges[idx],modes[idx])
                    psms += graph_gen(scan_ids[idx]+'decoy',m_over_zs[idx],intensities[idx],precursor_masses[idx]+decoy_value,charges[idx],modes[idx])
                
        mass_list = [0]+list(detokenize_aa_dict.values())[:-1]
        
    logging.info('preprocess done ' + str(len(psms)))
    # print(stop)
    train_ds = DGDataset(cfg, aa_dict, psms)
    collator = DGCollator(cfg)
    # train_sampler = DGBucketBatchSampler(cfg, train_spec_header)
    train_dl = DataLoader(train_ds,batch_size=128,collate_fn=collator,num_workers=24,pin_memory=True)
    train_dl = DataPrefetcher(train_dl,local_rank)
    model = DeepGlycan(cfg, torch.tensor(mass_list,device=local_rank), detokenize_aa_dict).to(local_rank)
    new_state_dict = {}
    model_path=cfg.model_path
    state_dict = torch.load(model_path, weights_only=True, map_location='cuda:0')  # rl-best-pos9.pt
    for key in state_dict.keys():
        new_key = key.replace('module.', '')  # Remove the 'module.' prefix
        new_state_dict[new_key] = state_dict[key]
    model.load_state_dict(new_state_dict)
    logging.info('model loaded')
    if dist.is_initialized(): model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    mass_candidates = [int(i) for i in cfg.inference.mass_cand.split(",")]

    inference_iter = OptimisedInference(
        cfg=cfg,
        model=model,
        inference_dl=train_dl,
        aa_dict=aa_dict,
        tokenize_aa_dict=tokenize_aa_dict,
        detokenize_aa_dict=detokenize_aa_dict,
        mass_candidates=mass_candidates,
    )
    evaluate( inference_iter, rank, cfg)
    logging.info('evaluation done')

    

if __name__ == '__main__':
    # beam_size = int(sys.argv[1])
    # main_wrapper(beam_size)
    main()
