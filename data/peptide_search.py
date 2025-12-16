from pyteomics import mgf      # pip install pyteomics
from glob import glob
import os
import numpy as np
import pandas as pd
import re
from alphapept import constants
from alphapept.fasta import get_frag_dict, parse,get_precmass
import logging
import alphapept
import pickle
from .preprocess_dataset import GraphGenerator
from pathlib import Path

PROTON_MASS = 1.00727646688
C_mass = 1.0034
aa_mass = constants.mass_dict
aa_mass['C'] = 160.03064895955

@alphapept.performance.performance_function
def compare_spectrum_parallel(query_idx,db_len, idxs_lower, idxs_higher, query_indices, query_frags, query_ints, db_indices, db_frags, best_hits,raw_hits, score ,frag_tol:float, ppm:bool):

    idx_low = idxs_lower[query_idx]
    idx_high = idxs_higher[query_idx]

    query_idx_start = query_indices[query_idx]
    query_idx_end = query_indices[query_idx + 1]
    query_frag = query_frags[query_idx_start:query_idx_end]
    query_int = query_ints[query_idx_start:query_idx_end]

    query_int_sum = 0
    for qi in query_int:
        query_int_sum += qi

    for db_idx in range(idx_low, idx_high):
        db_idx_start = db_indices[db_idx]
        db_idx_next = db_idx +1
        db_idx_end = db_indices[db_idx_next]

        db_frag = db_frags[db_idx_start:db_idx_end]

        q_max = len(query_frag)
        d_max = len(db_frag)
        # d_len = db_len[db_idx]
        hits,raw_matches = 0,0
        q, d = 0, 0  # q > query, d > database
        while q < q_max and d < d_max:
            mass1 = query_frag[q]
            mass2 = db_frag[d]
            delta_mass = mass1 - mass2

            if ppm:
                sum_mass = mass1 + mass2
                mass_difference = 2 * delta_mass / sum_mass * 1e6
            else:
                mass_difference = delta_mass

            if abs(mass_difference) <= frag_tol:
                raw_matches += 1
                hits += query_int[q]/query_int_sum
                d += 1
                q += 1  # Only one query for each db element
            elif delta_mass < 0:
                q += 1
            elif delta_mass > 0:
                d += 1
        hits /= d_max
        raw_matches /= d_max
        # raw_matches /= d_max

        len_ = best_hits.shape[1]
        for i in range(len_):
            if score[query_idx, i] < raw_matches:
                j = 1
                while len_-j >= (i+1):
                    k = len_-j
                    score[query_idx, k] = score[query_idx, k-1]
                    best_hits[query_idx, k] = best_hits[query_idx, k-1]
                    raw_hits[query_idx, k] = raw_hits[query_idx, k-1]
                    j+=1

                score[query_idx, i] = hits
                best_hits[query_idx, i] = db_idx
                raw_hits[query_idx, i] = raw_matches

                break

def get_psms(
    query_data: dict,
    db_data: dict,
    frag_tol: float = 0.02,
    prec_tol: float = 10,
    ppm: bool = False,
    min_frag_hits = 0,
    top_n: int = 50,
    **kwargs
):
    if alphapept.performance.COMPILATION_MODE == "cuda":
        import cupy
        cupy = cupy
    else:
        import numpy
        cupy = numpy

    db_masses = cupy.array(db_data['precursors'])
    db_frags = db_data['fragmasses']
    db_indices = db_data['indices']
    db_len = db_data['pep_len']

    query_indices = query_data["indices_ms2"]
    query_frags = query_data['mass_list_ms2']
    query_ints = query_data['int_list_ms2']

    query_masses = cupy.array(query_data['prec_mass_list2'])

    # idxs_lower, idxs_higher = get_idxs(
    #     db_masses,
    #     query_masses,
    #     prec_tol,
    #     ppm
    # )
    n_queries = len(query_masses)
    n_db = len(db_masses)
    idxs_lower = np.zeros(n_queries,  dtype=np.int64)
    idxs_higher = np.full (n_queries, n_db, dtype=np.int64)
    

    idxs_lower = cupy.array(idxs_lower)
    idxs_higher = cupy.array(idxs_higher)
    query_indices = cupy.array(query_indices)
    query_ints = cupy.array(query_ints)
    query_frags = cupy.array(query_frags)
    db_indices = cupy.array(db_indices)
    db_frags = cupy.array(db_frags)
    db_len = cupy.array(db_len)


    best_hits = cupy.zeros((n_queries, top_n), dtype=cupy.int_)-1
    score = cupy.zeros((n_queries, top_n), dtype=cupy.float32)
    raw_hits = cupy.zeros((n_queries, top_n), dtype=cupy.float32)

    logging.info(f'Performing search on {n_queries:,} query and {n_db:,} db entries with frag_tol = {frag_tol:.2f} and prec_tol = {prec_tol:.2f}.')

    compare_spectrum_parallel(cupy.arange(n_queries), db_len,idxs_lower, idxs_higher, query_indices, query_frags, query_ints, db_indices, db_frags, best_hits,raw_hits, score, frag_tol, ppm)

    query_idx, db_idx_ = cupy.where(raw_hits >= min_frag_hits)
    db_idx = best_hits[query_idx, db_idx_]
    score_ = score[query_idx, db_idx_]
    raw_hits = raw_hits[query_idx, db_idx_]

    if cupy.__name__ != 'numpy':
        query_idx = query_idx.get()
        db_idx = db_idx.get()
        score_ = score_.get()
        raw_hits = raw_hits.get()

    psms = np.array(
        list(zip(query_idx, db_idx, score_, raw_hits)), dtype=[("query_idx", int), ("db_idx", int), ("hits", float), ("raw_hits", float)]
    )

    logging.info('Found {:,} psms.'.format(len(psms)))

    return psms, 0

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
    db2idx = dict()
    motif_n_pep = {'precursors':[],'fragmasses':[],'indices':[], 'pep_len':[]}
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
                            p = parse(s)
                            frag_dict = get_frag_dict(p,aa_mass)
                            db_frag = [v for k, v in frag_dict.items() if 'b' in k]
                            motif_n_pep['precursors'].append(get_precmass(p, aa_mass)-aa_mass["H2O"])
                            motif_n_pep['fragmasses'] += db_frag
                            motif_n_pep['indices'].append(len(db_frag))
                            motif_n_pep['pep_len'].append(len(s))
                            db2idx[len(db2idx.keys())] = s
    query_indices = np.zeros(len(motif_n_pep['indices']) + 1, dtype=np.int64)
    query_indices[1:] = np.cumsum(motif_n_pep['indices'])
    motif_n_pep['indices'] = query_indices
    print('number of motif in this sample', len(db2idx))          
    return motif_n_pep, db2idx


def read_mgf(file):
    queries = {'indices_ms2': [], 'mass_list_ms2': [], 'int_list_ms2': [],'prec_mass_list2': [], 'charge':[]}
    scan2idx = dict()

    for i, spec in enumerate(mgf.read(file, convert_arrays=1)):  # convert_arrays→ NumPy
        p = spec['params']
        prec_mz  = p.get('pepmass')[0]
        charge = p.get('charge')[0]
        precursor = (prec_mz - PROTON_MASS) * abs(charge)
        title = p.get('title')
        m = re.search(r'\bscan=(\d+)\b', title, flags=re.IGNORECASE)
        scan2idx[i] = int(m.group(1)) if m else None
        queries['indices_ms2'].append(i)
        queries['mass_list_ms2'].append(spec['m/z array'])
        queries['int_list_ms2'].append(spec['intensity array'])
        queries['prec_mass_list2'].append(precursor)
        queries['charge'].append(charge)
    queries['mass_list_ms2'], queries['indices_ms2'], queries['int_list_ms2'] = build_query_arrays(queries['mass_list_ms2'], queries['int_list_ms2'])
    return queries, scan2idx

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

# def random_decoy(n_queries, )

def peptide_search(cfg):
    all_psms = []
    graph_gen = GraphGenerator()
    if cfg.inference.mode == 'N':
        db_data, db2idx = N_from_fasta(cfg.inference.fasta_file, cfg.inference.enzyme, cfg.inference.miss_cleavaged)
        if Path('decoy.pkl').is_file():
            with open('decoy.pkl', 'rb') as f:
                (decoy_db_data, decoy_db2idx) = pickle.load(f)
        else:
            decoy_db = N_from_fasta('athaliana.fasta','KR', 2)
            with open('decoy.pkl', 'wb') as f:
                pickle.dump(decoy_db, f)
    all_mgf = glob(os.path.join(cfg.inference.directory, '*.mgf'))

    for file in all_mgf:
        print('file',file)
        queries, scan2idx = read_mgf(file)
        query_indices = queries["indices_ms2"]
        query_frags = queries['mass_list_ms2']
        query_ints = queries['int_list_ms2']
        precursor_ion_mass = queries['prec_mass_list2']
        psms, _ = get_psms(queries, db_data)
        decoy_psms,_ = get_psms(queries, decoy_db[0], min_frag_hits=0)
        for psm in psms:
            query_idx, db_idx, score = psm
            spec_index = scan2idx[query_idx]
            pep_mass_given = db_data['precursors'][db_idx]
            pep_seq = db2idx[db_idx]
            product_ions_moverz = query_frags[query_indices[query_idx]:query_indices[query_idx+1]]
            product_ions_intensity = query_ints[query_indices[query_idx]:query_indices[query_idx+1]]
            precursor_charge = queries['charge']
            node_mass, node_input = graph_gen(spec_index, product_ions_moverz, product_ions_intensity, precursor_ion_mass, precursor_charge>2, pep_mass=pep_mass_given, mode=cfg.inference.fragmentation) 
            for iso in cfg.inference.isotope.split(','):
                psm_dic = {'scan':spec_index,
                        'node_mass': node_mass,
                        'node_input': node_input,
                        'charge': precursor_charge,
                        'precursor_mass': precursor_ion_mass+iso*C_mass,
                        'isotope_shift': iso,
                        'pep_mass': pep_mass_given,
                        'pep': pep_seq,
                        'pep_score': score}
            all_psms.append(psm_dic)
        for psm in decoy_psms:
            query_idx, db_idx, score = psm
            spec_index = scan2idx[query_idx]
            pep_mass_given = decoy_db_data['precursors'][db_idx]
            pep_seq = decoy_db2idx[db_idx]
            product_ions_moverz = query_frags[query_indices[query_idx]:query_indices[query_idx+1]]
            product_ions_intensity = query_ints[query_indices[query_idx]:query_indices[query_idx+1]]
            precursor_charge = queries['charge']
            node_mass, node_input = graph_gen(spec_index, product_ions_moverz, product_ions_intensity, precursor_ion_mass, precursor_charge>2, pep_mass=pep_mass_given, mode=cfg.inference.fragmentation) 
            psm_dic = {'scan':spec_index,
                       'node_mass': node_mass,
                       'node_input': node_input,
                       'charge': precursor_charge,
                       'precursor_mass': precursor_ion_mass,
                       'pep_mass': pep_mass_given,
                       'pep': 'decoy'+pep_seq,
                       'isotope_shift': iso,
                       'pep_score': score}
            all_psms.append(psm_dic)
    return all_psms

