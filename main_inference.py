import collections
import os
import re
import sys
import torch
import pickle
from torch import optim
import pandas as pd
from itertools import chain
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
from concurrent.futures import ProcessPoolExecutor
from ANN_search import cluster_spectra

import hydra
import json
import gzip
from pyteomics import mzml
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
detokenize_aa_dict = {i: round(aa.mass, 3) for i, aa in enumerate(mono_composition.values())}
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
    rows = [['Spec', 'isotope_shift','predict', 'mass', 'predict mass','mass difference', 'score_list', 'Pep mass', 'predict pep']]
    for i, finish_pool in enumerate(inference,start=0):
        for pool in finish_pool.values():
            # if pool.mass_diff > 0.05:
            #     continue
            inf_seq = pool.inference_seq
            psm_idx = pool.psm_idx
            mass_diff = pool.mass_diff
            given_pep_mass = pool.pep_mass
            isotope_shift = pool.isotope_shift
            precursor_mass = pool.precursor_mass
            report_mass = pool.report_mass
            pep = pool.pep
            score =[i.item() for i in pool.score_list]
            inf_seq_r = [mono_comp_dict_reversed[i] for i in inf_seq]
            # print(psm_idx,'final',''.join(inf_seq_r), sum(score), pool.mass_diff, isotope_shift)
            row = [psm_idx,isotope_shift,''.join(inf_seq_r), precursor_mass, report_mass, mass_diff,score,given_pep_mass,pep]
            # inf_seq_r = collections.Counter(inf_seq_r)
            # inf_seq_str = ''.join(f'{key}:{count} ' for key, count in inf_seq_r.items())                
            rows.append(row)

                # wandb.log({'accuracy_comp':sum(correct_comp)/len(correct_comp)})
    with open(cfg.test_spec_header_path + cfg.out_put_file, mode='w', newline='') as file:
        csv.writer(file).writerows(rows)
    logging.info('number of spectrum inferenced on test set'+str(len(rows)-1))

    return correct_comp
'''
def read_mgf(
    file_path,
    mode,
    *,
    # Common glycan oxonium ions (m/z, label). Feel free to add/remove.
    signature_ions=None,
    ppm_tol=20.0,
    min_count=1,
    min_rel_intensity=0.0,
):
    
    if signature_ions is None:
        signature_ions = [
            (126.055, "HexNAc-CH4O2+"),
            (138.055, "HexNAc-CH4O3+ / Hex-?"),
            (144.066, "HexNAc-CH3O3+"),
            (163.060, "Hex+"),
            (168.066, "HexNAc-H2O+"),
            (186.076, "HexNAc-?"),
            (204.087, "HexNAc+"),
            (274.092, "NeuAc-H2O+"),
            (292.103, "NeuAc+"),
            (366.140, "Hex+HexNAc+"),
        ]
    sig_mz = np.array([m for m, _ in signature_ions], dtype=float)

    raw_mgf_blocks = []
    for file in glob(os.path.join(file_path, 'HEART-Y4.mgf')):
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

                    if mz_arr.size == 0:
                        continue

                    # Relative intensity filter
                    base = I_arr.max() if I_arr.size and I_arr.max() > 0 else 1.0
                    rel = I_arr / base
                    keep = rel >= float(min_rel_intensity)
                    mz_keep = mz_arr[keep]
                    if mz_keep.size == 0:
                        # No peaks survive intensity threshold; discard spectrum
                        continue

                    # Oxonium matching (broadcast), count distinct signature ions matched
                    # diff_ppm shape: [num_kept_peaks, num_signature_ions]
                    diff_ppm = np.abs((mz_keep[:, None] - sig_mz[None, :]) / sig_mz[None, :]) * 1e6
                    matched_per_sig = (diff_ppm <= ppm_tol).any(axis=0)  # bool array over signature list
                    n_matches = int(matched_per_sig.sum())

                    if n_matches >= int(min_count):
                        rawfile = file.split('.')[0].split('/')[-1] + '.'
                        # Calibrate precursor using NeuAc oxonium ions (274.092, 292.103) within 0.02 Da
                        # neuac_targets = np.array([274.092, 292.103], dtype=float)
                        neuac_targets = np.array([Composition('proton').mass,Composition('C11H15O7N1').mass+Composition('proton').mass, Composition('C11H17O8N1').mass+Composition('proton').mass], dtype=float)
                        # print(neuac_targets)
                        tol_da = 0.02
                        deltas = []
                        intensities_for_matches = []
                        diff_da = np.abs(mz_arr[:, None] - neuac_targets[None, :])
                        match_mask = diff_da <= tol_da
                        if match_mask.any():
                            matched_indices = np.argwhere(match_mask)
                            for peak_idx, target_idx in matched_indices:
                                intensities_for_matches.append(I_arr[peak_idx])
                                deltas.append(neuac_targets[target_idx] - mz_arr[peak_idx])
                            # Use the highest-intensity matched peak to compute shift
                            # best = int(np.argmax(intensities_for_matches))
                            precursor_mz_calibrated = mz + sum(deltas)#deltas[best]
                        else:
                            precursor_mz_calibrated = mz

                        neutral_precursor_mass = precursor_mz_calibrated * z - z * Composition('proton').mass
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
                            # else: filtered out (no sufficient signature evidence)

    return raw_mgf_blocks
'''
def read_mgf(
    file_path,
    mode,
    *,
    # Common glycan oxonium ions (m/z, label). Feel free to add/remove.
    signature_ions=None,
    ppm_tol=20.0,
    min_count=1,
    min_rel_intensity=0.0,
):
    
    if signature_ions is None:
        signature_ions = [
            (126.055, "HexNAc-CH4O2+"),
            (138.055, "HexNAc-CH4O3+ / Hex-?"),
            (144.066, "HexNAc-CH3O3+"),
            (163.060, "Hex+"),
            (168.066, "HexNAc-H2O+"),
            (186.076, "HexNAc-?"),
            (204.087, "HexNAc+"),
            (274.092, "NeuAc-H2O+"),
            (292.103, "NeuAc+"),
            (366.140, "Hex+HexNAc+"),
        ]
    sig_mz = np.array([m for m, _ in signature_ions], dtype=float)

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

                    if mz_arr.size == 0:
                        continue

                    # Relative intensity filter
                    base = I_arr.max() if I_arr.size and I_arr.max() > 0 else 1.0
                    rel = I_arr / base
                    keep = rel >= float(min_rel_intensity)
                    mz_keep = mz_arr[keep]
                    if mz_keep.size == 0:
                        # No peaks survive intensity threshold; discard spectrum
                        continue

                    # Oxonium matching (broadcast), count distinct signature ions matched
                    # diff_ppm shape: [num_kept_peaks, num_signature_ions]
                    diff_ppm = np.abs((mz_keep[:, None] - sig_mz[None, :]) / sig_mz[None, :]) * 1e6
                    matched_per_sig = (diff_ppm <= ppm_tol).any(axis=0)  # bool array over signature list
                    n_matches = int(matched_per_sig.sum())

                    if n_matches >= int(min_count):
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
                    # else: filtered out (no sufficient signature evidence)

    return raw_mgf_blocks



def read_mzml(
    file_path,
    mode,
    *,
    signature_ions=None,
    ppm_tol=20.0,
    min_count=1,
    min_rel_intensity=0.0,
):
    if signature_ions is None:
        signature_ions = [
            (126.055, "HexNAc-CH4O2+"),
            (138.055, "HexNAc-CH4O3+ / Hex-?"),
            (144.066, "HexNAc-CH3O3+"),
            (163.060, "Hex+"),
            (168.066, "HexNAc-H2O+"),
            (186.076, "HexNAc-?"),
            (204.087, "HexNAc+"),
            (274.092, "NeuAc-H2O+"),
            (292.103, "NeuAc+"),
            (366.140, "Hex+HexNAc+"),
        ]
    sig_mz = np.array([m for m, _ in signature_ions], dtype=float)

    raw_mzml_blocks = []
    for file in glob(os.path.join(file_path, '*_calibrated.mzML')):
        print('file', file)
        with mzml.MzML(file) as reader:
            for spec in reader:
                if spec.get('ms level') != 2:
                    continue

                precursors = spec.get('precursorList', {}).get('precursor', [])
                if not precursors:
                    continue
                selected_ions = precursors[0].get('selectedIonList', {}).get('selectedIon', [])
                if not selected_ions:
                    continue
                selected_ion = selected_ions[0]
                mz_val = selected_ion.get('selected ion m/z')
                charge = selected_ion.get('charge state')
                if mz_val is None or charge is None:
                    continue
                try:
                    z = int(charge)
                except (TypeError, ValueError):
                    continue

                scan_id = spec.get('id')
                m_ = re.search(r'scan=(\d+)', scan_id or '')
                scan = m_.group(1) if m_ else scan_id
                if scan is None:
                    continue

                mz_arr = np.asarray(spec.get('m/z array', []), dtype=float)
                I_arr = np.asarray(spec.get('intensity array', []), dtype=float)
                if mz_arr.size == 0 or I_arr.size == 0:
                    continue

                base = I_arr.max() if I_arr.max() > 0 else 1.0
                rel = I_arr / base
                keep = rel >= float(min_rel_intensity)
                mz_keep = mz_arr[keep]
                if mz_keep.size == 0:
                    continue

                diff_ppm = np.abs((mz_keep[:, None] - sig_mz[None, :]) / sig_mz[None, :]) * 1e6
                matched_per_sig = (diff_ppm <= ppm_tol).any(axis=0)
                n_matches = int(matched_per_sig.sum())

                if n_matches >= int(min_count):
                    rawfile = os.path.splitext(os.path.basename(file))[0] + '.'
                    neutral_precursor_mass = mz_val * z - z * Composition('proton').mass
                    raw_mzml_blocks.append(
                        (
                            rawfile + str(scan),
                            mz_arr,
                            I_arr,
                            neutral_precursor_mass,
                            z,
                            mode,
                        )
                    )

    return raw_mzml_blocks


def subsequences_starting_and_ending_with_KR(s, enzyme):
    # Identify positions of 'K' and 'R'
    positions = [i for i, c in enumerate(s) if c in enzyme]# {'F', 'W', 'Y', 'E'}]
    
    # Collect subsequences that start and end with K or R
    subsequences = []
    for start in positions:
        for end in positions:
            if start < end:
                subsequences.append(s[start+1:end+1])
    
    return subsequences

def read_fasta(path, enzyme='KR', miss_cleavaged=3):
    pep2mass = dict()
    with open(path, 'r') as f:
        line = f.readline()
        # print(f.readlines())
        while len(line) != 0:
            if line.startswith('>'):
                line = f.readline()
            else:
                proteins =[]
                while (not line.startswith('>')) and len(line) != 0:
                    proteins.append(line.strip())
                    
                    line = f.readline()
                protein_seq = ''.join(proteins)
                if 'U' in protein_seq:
                    continue
                split_strings = subsequences_starting_and_ending_with_KR(protein_seq, enzyme)

                # max len < 40; miss_cleavage = 3
                for s in split_strings:
                    if len(s) < 40 and  sum(s.count(i) for i in enzyme) <= miss_cleavaged:
                        pattern = r'N[A-Za-z][ST]'
                        match = re.search(pattern, s)
                        # if match:
                        pep2mass[s] = round(Residual_seq(s).mass + Composition('H2O').mass + Composition('proton').mass,5)
    return pep2mass


def sample_in_range_numpy(a, b, *, seed=None, inclusive=True):
    """
    Return len(a) distinct samples from b within [min(a), max(a)] (or (min,max) if inclusive=False).
    Requires: numeric a, b. Raises if not enough values in range.
    """
    if not a:
        return []
    rng = np.random.default_rng(seed)
    lo, hi = (min(a), max(a))
    b_arr = np.asarray(list(b.values()))
    b_key_arr = np.asarray(list(b.keys()))
    if inclusive:
        mask = (b_arr >= lo) & (b_arr <= hi)
    else:
        mask = (b_arr >  lo) & (b_arr <  hi)
    exclude_mask = np.isin(b_arr, list(set(a)), invert=True)
    candidates = b_arr[mask & exclude_mask]
    # candidates = b_arr[mask]
    k = len(a) *100
    if candidates.size < k:
        raise ValueError(f"Need {k} items, found {candidates.size} in range [{lo}, {hi}].")
    # sample without replacement
    idx = rng.choice(candidates.size, size=k, replace=False)
    key_selected = b_key_arr[mask & exclude_mask][idx]
    value_selected = candidates[idx]
    return dict(zip(key_selected, value_selected))

def determine_list():
    psm = pd.read_csv('/home/q359zhan/olinked/data/alzheimer/lfq-normal/psmclean.tsv',delimiter='\t')
    df = pd.read_parquet("Hyper-Spec/output.csv.parquet")
    df["identifier_scan"] = df["identifier"].astype(str) + "." + df["scan"].astype(str)
    spec_set = []

    for i, row in psm.iterrows():
        # normalize and add unique values from this chunk
        val = row['Spectrum'].split('.')
        # print(val[:2])
        val = '.'.join(val[:2])
        spec_set.append(val)
    present_mask = df['identifier_scan'].isin(spec_set)  # vectorized
    any_present_by_cluster = present_mask.groupby(df['cluster']).any()
    clusters_absent = any_present_by_cluster[~any_present_by_cluster].index
    result = df[df['cluster'].isin(clusters_absent)]
    group = result.groupby('cluster').size()
    big_clusters = group[group > 50].keys()
    return df[df['cluster'].isin(big_clusters)]["identifier_scan"].tolist()

def determine_list_temp(cluster_number):
    df = pd.read_parquet("Hyper-Spec/output.csv.parquet")
    df["identifier_scan"] = df["identifier"].astype(str) + "." + df["scan"].astype(str)
    return df[df['cluster']==cluster_number]["identifier_scan"].tolist()


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg:DictConfig):
    # test why not found
    # not_found_psm = pd.read_csv('/home/q359zhan/olinked/data/test_igg/label_not_found.csv')
    spectrum = read_mgf(cfg.inference.directory, cfg.inference.fragmentation.lower())
    # with open('/home/q359zhan/olinked/data/ethcd/heart/mzml/spec_heart_calibrated_mzml.pkl', 'rb') as f:
    #     spectrum = pickle.load(f)
    a,b,c,d,e,f = zip(*spectrum)
    print(a[0])
    scan_ids, m_over_zs, intensities, precursor_masses, multi_chargeds, modes = [],[],[],[],[],[]
    # for s in ['MOUSE-MALE-O4-HEART_calibrated.6874', 'MOUSE-MALE-Y4-HEART_calibrated.7310', 'MOUSE-MALE-Y2-HEART_calibrated.6894', 'MOUSE-MALE-Y2-HEART_calibrated.7124', 'MOUSE-MALE-Y1-HEART_calibrated.7559', 'MOUSE-MALE-O3-HEART_calibrated.6850', 'MOUSE-MALE-Y3-HEART_calibrated.7080', 'MOUSE-MALE-Y3-HEART_calibrated.7304', 'MOUSE-MALE-O1-HEART_calibrated.7275', 'MOUSE-MALE-O1-HEART_calibrated.7541', 'MOUSE-MALE-O2-HEART_calibrated.6932', 'MOUSE-MALE-O2-HEART_calibrated.7156', 'MOUSE-MALE-Y4-HEART_calibrated.7081']:
    for s in ['HEART-Y4.7310', 'HEART-O4.6874', 'HEART-Y2.6894', 'HEART-Y2.7124', 'HEART-Y1.7559', 'HEART-O3.6850', 'HEART-Y3.7080', 'HEART-Y3.7304', 'HEART-O1.7275', 'HEART-O1.7541', 'HEART-O2.6932', 'HEART-O2.7156', 'HEART-Y4.7081']: 
        idx = a.index(s)
        scan_ids.append(s)
        m_over_zs.append(b[idx])
        intensities.append(c[idx])
        precursor_masses.append(d[idx])
        multi_chargeds.append(e[idx])
        modes.append(f[idx])
    # print(m_over_zs)
    # with open('spec_alz_disease.pkl', 'rb') as f:
    #     spectrum = pickle.load(f)
    # scan_ids,_,_,_,_,_ = zip(*spectrum)
    # undetermined_scan = determine_list_temp(24364)
    # print(undetermined_scan)
    # undetermined_spectrum = []
    # for s in undetermined_scan:
    #     undetermined_spectrum.append(spectrum[scan_ids.index(s)])
    # target_peptides2mass = read_fasta(cfg.inference.fasta_file)
    target_peptides2mass = {'IRTTTSGVPR':round(Residual_seq('IRTTTSGVPR').mass + Composition('H2O').mass + Composition('proton').mass,5)}
    mass_list = [0]+list(detokenize_aa_dict.values())[:-1]
    logging.info('Number of peptides'+ str(len(target_peptides2mass)))
    graph_gen = GraphGeneratorOptimized(local_sliding_window=50,
                                        data_acquisition_upper_limit=3500,
                                        pep_masses=target_peptides2mass,
                                        isotopes=cfg.inference.isotope, 
                                        min_cand = min(mass_list))
    psms = []
    n_cores = os.cpu_count()//2
    logging.info('Number of spectrum'+ str(len(scan_ids)))
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        psms += list(executor.map(graph_gen,
                             scan_ids,
                             m_over_zs,
                             intensities,
                             precursor_masses,
                             multi_chargeds,
                             modes))
    psms = list(chain.from_iterable(psms))
    logging.info('preprocess done ' + str(len(psms)))
    # print(stop)
    train_ds = DGDataset(cfg, aa_dict, psms)
    collator = DGCollator(cfg)
    # train_sampler = DGBucketBatchSampler(cfg, train_spec_header)
    train_dl = DataLoader(train_ds,batch_size=512,collate_fn=collator,num_workers=24,pin_memory=True)
    train_dl = DataPrefetcher(train_dl,local_rank)
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
