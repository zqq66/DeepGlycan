from pyteomics import mgf      # pip install pyteomics
from glob import glob
import os
import numpy as np
import pandas as pd
import re
from .BasicClass import Residual_seq
import pickle
import logging
from .preprocess_dataset import GraphGenerator
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from functools import partial

C_mass = 1.0034
PROTON_MASS = 1.00727646688


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

def N_from_fasta(fasta_file, enzyme, miss_cleavaged):
    motif_n_pep = {'precursors':[],'pep_seq':[], 'pep_len':[]}
    with open(fasta_file, 'r') as f:
        line = f.readline()
        # print(f.readlines())
        while len(line) != 0:
            if line.startswith('>'):
                name = line
                line = f.readline()
            else:
                proteins =[]
                while (not line.startswith('>')) and len(line) != 0:
                    proteins.append(line.strip())
                    
                    line = f.readline()
                protein_seq = ''.join(proteins)
                split_strings = subsequences_starting_and_ending_with_KR(protein_seq, enzyme)

                # max len < 40; miss_cleavage = 3
                for s in split_strings:
                    if len(s) < 40 and  sum(s.count(i) for i in enzyme) <= miss_cleavaged:
                        pattern = r'N[A-Za-z][ST]'
                        match = re.search(pattern, s)
                        if match:
                            motif_n_pep['precursors'].append(Residual_seq(s).mass)
                            motif_n_pep['pep_len'].append(len(s))
                            motif_n_pep['pep_seq'].append(s)
   
    print('number of motif in this sample', len(motif_n_pep['precursors']))          
    return motif_n_pep

def build_query_arrays(mz_lists, int_lists=None): 
    query_frags = np.concatenate(mz_lists)

    # build index vector: indices[i] == start of spectrum i
    lens = np.fromiter((len(a) for a in mz_lists), dtype=np.int64)
    query_indices = np.zeros(len(lens) + 1, dtype=np.int64)
    query_indices[1:] = np.cumsum(lens)

    # concatenate intensities if provided
    if int_lists is not None:
        query_ints = np.concatenate(int_lists)
        return query_frags, query_indices, query_ints

    return query_frags, query_indices

def read_mgf(file):
    queries = {'indices_ms2': [0], 'mass_list_ms2': [], 'int_list_ms2': [],'prec_mass_list2': [], 'charge':[]}
    scan2idx = dict()

    for i, spec in enumerate(mgf.read(file, convert_arrays=1)):  # convert_arrays→ NumPy
        p = spec['params']
        prec_mz  = p.get('pepmass')[0]
        charge = p.get('charge')[0]
        precursor = (prec_mz - PROTON_MASS) * abs(charge)
        title = p.get('title')
        m = re.search(r'\bscan=(\d+)\b', title, flags=re.IGNORECASE)
        scan2idx[i] = int(m.group(1)) if m else None
        queries['indices_ms2'].append(queries['indices_ms2'][-1]+spec['m/z array'].size)
        queries['mass_list_ms2'].append(spec['m/z array'])
        queries['int_list_ms2'].append(spec['intensity array'])
        queries['prec_mass_list2'].append(precursor)
        queries['charge'].append(charge)
    queries['mass_list_ms2'] = np.concatenate(queries['mass_list_ms2'])
    queries['int_list_ms2'] = np.concatenate(queries['int_list_ms2'])
    return queries, scan2idx

def analyze_file(query_idx,queries,scan2idx, db_data,isotope,fragmentation, graph_gen):
    all_psms = []
    query_indices = queries["indices_ms2"]
    query_frags = queries['mass_list_ms2']
    query_ints = queries['int_list_ms2']
    precursors = queries['prec_mass_list2']
    for db_idx in range(len(db_data['precursors'])):
        spec_index = scan2idx[query_idx]
        pep_mass_given = db_data['precursors'][db_idx]
        pep_seq = db_data['pep_seq'][db_idx]
        product_ions_moverz = query_frags[query_indices[query_idx]:query_indices[query_idx+1]]
        product_ions_intensity = query_ints[query_indices[query_idx]:query_indices[query_idx+1]]
        precursor_charge = queries['charge'][query_idx]
        precursor_ion_mass = precursors[query_idx]
        for iso in isotope.split(','):
            precursor_mass = precursor_ion_mass+int(iso)*C_mass
            node_mass, node_input = graph_gen(spec_index, product_ions_moverz, product_ions_intensity, precursor_mass, precursor_charge>2, pep_mass=pep_mass_given, mode=fragmentation) 
            if pep_mass_given < precursor_mass:
                psm_dic = {'scan':spec_index,
                        'node_mass': node_mass,
                        'node_input': node_input,
                        'charge': precursor_charge,
                        'precursor_mass':precursor_mass,
                        'isotope_shift': iso,
                        'pep_mass': pep_mass_given,
                        'pep': pep_seq,
                        }
                all_psms.append(psm_dic)
    return all_psms

def peptide_search(cfg):
    
    graph_gen = GraphGenerator()
    if cfg.inference.mode == 'N':
        db_data = N_from_fasta(cfg.inference.fasta_file, cfg.inference.enzyme, cfg.inference.miss_cleavaged)
        if Path('decoy.pkl').is_file():
            with open('decoy.pkl', 'rb') as f:
                (decoy_db_data, decoy_db2idx) = pickle.load(f)
        else:
            decoy_db = N_from_fasta('athaliana.fasta','KR', 2)
            with open('decoy.pkl', 'wb') as f:
                pickle.dump(decoy_db, f)
    all_mgf = glob(os.path.join(cfg.inference.directory, 'IgG-MCC-150ng-2.mgf'))
    results = []
    for file in all_mgf:
        logging.info('file {} processing.'.format(file))
        queries, scan2idx = read_mgf(file)
        n_cores = os.cpu_count() -4
        analyze = partial(analyze_file,queries=queries,scan2idx=scan2idx, db_data=db_data, isotope=cfg.inference.isotope,fragmentation=cfg.inference.fragmentation, graph_gen=graph_gen)
        with ProcessPoolExecutor(max_workers=n_cores) as exe:
            results += list(exe.map(analyze,  list(scan2idx.keys())))
        break
    return results

