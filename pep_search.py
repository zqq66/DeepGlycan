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

logger = logging.getLogger(__name__)
_SEARCHER = None
_CLUSTER_DF = None
_SCAN_ID_TO_IDX = None
_M_OVER_ZS = None
_INTENSITIES = None
_PRECURSOR_MASSES = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_mgf_settings(config_path: Path, mgf_dir_override: Optional[Path] = None, glob_override: Optional[str] = None):
    """
    Load mgf_preprocess config to retrieve directory and glob pattern.
    Preference order for directory: override -> out_directory -> in_directory.
    """
    cfg = OmegaConf.load(str(config_path))
    mgf_dir = mgf_dir_override or cfg.get("out_directory") or cfg.get("in_directory")
    if mgf_dir is None:
        raise ValueError("MGF directory not found in config and no override provided.")
    glob_pattern = glob_override or cfg.get("glob_pattern", "*.mgf")
    return Path(mgf_dir), glob_pattern


def read_mgf(file_path: Path, mode: str, glob_pattern: str = "*.mgf"):
    raw_mgf_blocks = []
    for file in glob(os.path.join(str(file_path), glob_pattern)):
        logger.info("Reading MGF file: %s", file)
        with open(file) as f:
            for line in f:
                if line.startswith("BEGIN IONS"):
                    product_ions_moverz = []
                    product_ions_intensity = []
                    mz = None
                    z = None
                    scan = None
                elif line.startswith("PEPMASS"):
                    mz = float(re.split(r"=|\r|\n|\s", line)[1])
                elif line.startswith("CHARGE"):
                    z = int(re.search(r"CHARGE=(\d+)\+", line).group(1))
                elif line.startswith("TITLE"):
                    scan_pattern = r"scan=(\d+)"
                    m_ = re.search(scan_pattern, line)
                    scan = m_.group(1) if m_ else None
                elif line and line[0].isdigit():
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        mz_i, I_i = parts[0], parts[1]
                        product_ions_moverz.append(float(mz_i))
                        product_ions_intensity.append(float(I_i))
                elif line.startswith("END IONS"):
                    if mz is None or z is None or scan is None:
                        continue
                    mz_arr = np.asarray(product_ions_moverz, dtype=float)
                    I_arr = np.asarray(product_ions_intensity, dtype=float)
                    if mz_arr.size == 0:
                        continue
                    rawfile = file.split(".")[0].split("/")[-1] + "."
                    neutral_precursor_mass = mz * z - (z - 1) * Composition("proton").mass
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


def build_peptide_entries_from_fasta(fasta_path: Path):
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
            fragments, cleavages = _theoretical_fragments(sequence, include_cz_ions=True)
            peptide = PeptideEntry(sequence, _monoisotopic_mass(sequence), fragments, cleavages)
            peptides.append(peptide)
            seen.add(sequence)
    return peptides


def build_peptide_entries_from_list(sequences):
    peptides = []
    seen = set()
    for sequence in sequences:
        sequence = sequence.replace("J", "N")
        if sequence in seen:
            continue
        fragments, cleavages = _theoretical_fragments(sequence, include_cz_ions=True)
        peptide = PeptideEntry(sequence, _monoisotopic_mass(sequence), fragments, cleavages)
        peptides.append(peptide)
        seen.add(sequence)
    return peptides


def _init_pool(searcher, cluster_df, scan_id_to_idx, m_over_zs, intensities, precursor_masses):
    global _SEARCHER, _CLUSTER_DF, _SCAN_ID_TO_IDX, _M_OVER_ZS, _INTENSITIES, _PRECURSOR_MASSES
    _SEARCHER = searcher
    _CLUSTER_DF = cluster_df
    _SCAN_ID_TO_IDX = scan_id_to_idx
    _M_OVER_ZS = m_over_zs
    _INTENSITIES = intensities
    _PRECURSOR_MASSES = precursor_masses


def _process_cluster(cluster):
    dic = dict()
    subset = _CLUSTER_DF[_CLUSTER_DF["cluster"] == cluster]
    for _, row in subset.iterrows():
        scan = row["scan"]
        idx = _SCAN_ID_TO_IDX.get(scan)
        if idx is None:
            continue
        spectrum_peaks = list(zip(_M_OVER_ZS[idx], _INTENSITIES[idx]))
        precursor_mass = _PRECURSOR_MASSES[idx]
        match = _SEARCHER.find_top_k_peptides(spectrum_peaks, precursor_mass=precursor_mass, top_k=3)
        for m in match:
            dic[m.sequence] = dic.get(m.sequence, 0) + 1
    top3_keys = sorted(dic, key=dic.get, reverse=True)[:3]
    top3_values = [dic[k] for k in top3_keys]
    return [cluster, subset["scan"].tolist(), len(subset["scan"].tolist()), dic, top3_keys, top3_values]


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------
@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    logging.basicConfig(
        level=getattr(logging, "INFO", logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    mgf_dir = cfg.out_dir
    glob_pattern = cfg.mgf_preprocess.glob_pattern
    mode = cfg.mode
    logger.info("Using MGF directory: %s (pattern: %s)", mgf_dir, glob_pattern)

    spectrum = read_mgf(mgf_dir, mode, glob_pattern=glob_pattern)
    print(len(spectrum))
    if not spectrum:
        logger.warning("No spectra found in %s matching %s", mgf_dir, glob_pattern)
        return
    scan_ids, m_over_zs, intensities, precursor_masses, multi_chargeds, modes = zip(*spectrum)
    scan_id_to_idx = {scan: i for i, scan in enumerate(scan_ids)}

    cluster_parquet = cfg.mgf_preprocess.cluster_parquet
    fasta = cfg.mgf_preprocess.fasta
    
    if fasta:
        fasta_path = Path(to_absolute_path(fasta))
        logger.info("Building peptide entries from FASTA: %s", fasta_path)
        peptides = build_peptide_entries_from_fasta(fasta_path)
    else:
        id_csv = cfg.mgf_preprocess.pep_csv
        id_df = pd.read_csv(to_absolute_path(id_csv))
        id_df["scan"] = id_df["RawName"].astype(str) + "." + id_df["Scan"].astype(str)
        peptide_dbsearch = id_df["Peptide"].str.replace("J", "N").unique().tolist()
        logger.info("Building peptide entries directly from ID CSV sequences")
        peptides = build_peptide_entries_from_list(peptide_dbsearch)
    logger.info("Peptide entries retained: %d", len(peptides))

    searcher = IonIndexedOpenSearch(peptides)

    # Ensure proper path joining even when mgf_dir is a string
    cluster_df = pd.read_parquet(Path(to_absolute_path(Path(mgf_dir) / cluster_parquet)))
    cluster_df["scan"] = cluster_df["identifier"].str.replace("_with_scans_filtered", "", regex=False) + "." + cluster_df["scan"].astype(str)

    summary = (
        cluster_df.groupby("cluster")
        .agg(
            size=("cluster", "size"),
            avg_precursor_mz=("precursor_mz", "mean"),
            avg_precursor_charge=("precursor_charge", "mean"),
        )
    )
    top_clusters = summary.index
    logger.info("Clusters to process: %d", len(top_clusters))

    rows = [["cluster", "scan", "#scan", "pep_dic", "top3_pep", "top3_values"]]
    with mp.Pool(
        mp.cpu_count() // 2,
        initializer=_init_pool,
        initargs=(searcher, cluster_df, scan_id_to_idx, m_over_zs, intensities, precursor_masses),
    ) as pool:
        for row in tqdm(pool.imap(_process_cluster, top_clusters), desc="Processing clusters", total=len(top_clusters)):
            rows.append(row)

    output_path = Path(to_absolute_path(Path(mgf_dir) / "clustered_peptide_search.csv"))
    with open(output_path, mode="w", newline="") as file:
        csv.writer(file).writerows(rows)
    logger.info("Wrote results to %s", output_path)


if __name__ == "__main__":
    main()
