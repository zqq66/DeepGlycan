import csv
import logging
import os
import re
from glob import glob
from pathlib import Path
from typing import Iterable, Optional

import hydra
import multiprocessing as mp
import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from data.BasicClass import Composition
from ion_indexed_open_search import (
    IonIndexedOpenSearch,
    PeptideEntry,
    _monoisotopic_mass,
    _theoretical_fragments,
)
from hydra import compose, initialize
from omegaconf import DictConfig

def load_cfg() -> DictConfig:
    # config_path is relative to the repo root where the configs/ dir lives
    with initialize(version_base=None, config_path="configs"):
        cfg = compose(config_name="config")
    return cfg

# usage
cfg = load_cfg()

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

def build_peptide_entries_from_fasta(fasta_path: Path, include_cz_ions: bool):
    """
    Build peptide entries from a plain text file where each line is a peptide sequence.
    The enzyme and miss_cleavage parameters are kept for compatibility but are not used.
    """
    peptides = []
    seen = set()
    with open(fasta_path, "r") as f:
        for line in f:
            sequence = line.strip()
            if sequence in seen:
                continue
            fragments, cleavages = _theoretical_fragments(sequence, include_cz_ions=include_cz_ions)
            peptide = PeptideEntry(sequence, _monoisotopic_mass(sequence), fragments, cleavages)
            peptides.append(peptide)
            seen.add(sequence)
    return peptides

logger = logging.getLogger(__name__)
mgf_dir = cfg.out_dir
glob_pattern = cfg.mgf_preprocess.glob_pattern
cluster_parquet = cfg.mgf_preprocess.cluster_parquet
fasta = cfg.mgf_preprocess.fasta
mode = cfg.mode
include_cz_ions = str(mode).lower() == "ethcd"

cluster_df = pd.read_parquet(Path(to_absolute_path(Path(mgf_dir) / cluster_parquet)))
cluster_df["scan"] = cluster_df["identifier"].str.replace('_with_scans_filtered', '', regex=False) + "." + cluster_df["scan"].astype(str)

spectrum = read_mgf(mgf_dir, mode)
scan_ids, m_over_zs, intensities, precursor_masses, multi_chargeds, modes = zip(*spectrum)
scan_id_to_idx = {scan: i for i, scan in enumerate(scan_ids)}

fasta_path = Path(to_absolute_path(fasta))
logger.info("Building peptide entries from FASTA: %s", fasta_path)
peptides = build_peptide_entries_from_fasta(fasta_path, include_cz_ions=include_cz_ions)
searcher = IonIndexedOpenSearch(peptides)
summary = (
    cluster_df
        .groupby('cluster')
        .agg(
            size=('cluster', 'size'),
            avg_precursor_mz=('precursor_mz', 'mean'),
            avg_precursor_charge=('precursor_charge', 'mean')
        )
)
# has_Y4 = (
#     cluster_df
#         .groupby('cluster')['scan']
#         .apply(lambda s: s.str.contains('Y4').any())
# )

# # Clusters to keep
# keep_clusters = has_Y4[has_Y4].index

# # Now filter your summary
# summary_filtered = summary.loc[keep_clusters]

top_clusters = summary.index
print(len(top_clusters))
# '''
def _process_cluster(cluster):
    """Process a single cluster and return the row to be written."""
    dic = dict()
    subset = cluster_df[cluster_df['cluster'] == cluster]
    for _, row in subset.iterrows():
        scan = row['scan']
        idx = scan_id_to_idx.get(scan)
        if idx is None:
            continue
        spectrum = list(zip(m_over_zs[idx], intensities[idx]))
        precursor_mass = precursor_masses[idx]
        match = searcher.find_top_k_peptides(spectrum, precursor_mass=precursor_mass, top_k=3)
        for m in match:
            dic[m.sequence] = dic.get(m.sequence, 0) + 1
    top3_keys = sorted(dic, key=dic.get, reverse=True)[:3]
    top3_values = [dic[k] for k in top3_keys]
    return [cluster, subset['scan'].tolist(), len(subset['scan'].tolist()), dic, top3_keys, top3_values]


def main_cluster():
    rows = [['cluster', 'scan', '#scan', 'pep_dic', 'top3_pep', 'top3_values']]
    output_path = Path(to_absolute_path(Path(mgf_dir) / "clustered_peptide_search_try.csv"))
    with mp.Pool(mp.cpu_count()-2) as pool:
        for row in tqdm(pool.imap(_process_cluster, top_clusters), desc="Processing clusters", total=len(top_clusters)):
            rows.append(row)
    with open(output_path, mode='w', newline='') as file:
        csv.writer(file).writerows(rows)

if __name__ == "__main__":
    main_cluster()
