"""
Standalone script combining the first two cells of helper/FDR.ipynb with the
supporting scoring utilities from helper/rescoring.py. It computes glycan
scores for identifications, applies an FDR filter, and writes the passing
entries to disk. Configured via Hydra (configs/config.yaml).
"""

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from pyteomics import mgf
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

# Local dependency for mass calculations
from data.BasicClass import Composition, Ion

PROTON_MASS = 1.00727646688

# Monosaccharide composition and masses borrowed from helper/rescoring.py
MONO_COMPOSITION = {
    "H": Composition("C6H12O6") - Composition("H2O"),
    "N": Composition("C8H15O6N") - Composition("H2O"),
    "A": Composition("C11H19O9N") - Composition("H2O"),
    "G": Composition("C11H19O10N") - Composition("H2O"),
    "F": (Composition("C6H12O5") - Composition("H2O")),
    "X": (Composition("C5H10O5") - Composition("H2O")),
}
ID_TO_MASS = {k: v.mass for k, v in MONO_COMPOSITION.items()}
DEFAULT_FDR_CFG = {
    "input_csv": "/home/q359zhan/olinked/data/pxd035846-test/filtered_mgf/glyco-engineered-deepglcyan-inference.csv",
    "mgf": "/home/q359zhan/olinked/data/ethcd/heart/glycan/HEART-Y4.mgf",
    "output": "/home/q359zhan/olinked/data/ethcd/heart/glycan/heart-y4-precursor-calibrated-inference-fdr.csv",
    "spec_prefix": "HEART-Y4.",
    "mass_diff_threshold": 0.02,
    "fdr_threshold": 0.01,
}


def record_filter(mass_list: np.ndarray, pep_mass: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Filter masses by a trivial non-negative mask (kept for parity with rescoring.py)."""
    del pep_mass  # retained for signature compatibility
    mask = mass_list >= 0
    return mass_list[mask], mask


def find_fragments(sequence: Iterable[str]) -> np.ndarray:
    """Enumerate all unique glycan fragment masses for the provided sequence."""
    import more_itertools

    glyan_ions = {0}
    for i in range(1, len(sequence) + 1):
        all_comb = set(more_itertools.distinct_combinations(sequence, i))
        all_comb_mass = set(sum(ID_TO_MASS[s] for s in ss) for ss in all_comb)
        glyan_ions = glyan_ions.union(all_comb_mass)
    return np.sort(np.array(list(glyan_ions)))


def graphnode_mass_generator(
    product_ions_moverz: np.ndarray,
    product_ions_intensity: np.ndarray,
    muti_charged: int,
    pep_mass: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate observed fragment masses across multiple charge states."""
    node_1y_mass_cterm, mask = record_filter(product_ions_moverz, pep_mass)
    node_1y_mass_cterm_int = product_ions_intensity[mask]

    node_2y_mass_cterm, mask = record_filter(Ion.mass2mz(product_ions_moverz, 2), pep_mass)
    node_2y_mass_cterm_int = product_ions_intensity[mask]

    node_3y_mass_cterm, mask = record_filter(Ion.mass2mz(product_ions_moverz, 3), pep_mass)
    node_3y_mass_cterm_int = product_ions_intensity[mask]

    if muti_charged:
        graphnode_mass = np.concatenate([node_1y_mass_cterm, node_2y_mass_cterm, node_3y_mass_cterm])
        graphnode_mass_int = np.concatenate(
            [node_1y_mass_cterm_int, node_2y_mass_cterm_int, node_3y_mass_cterm_int]
        )
    else:
        graphnode_mass = np.concatenate([node_1y_mass_cterm, node_2y_mass_cterm])
        graphnode_mass_int = np.concatenate([node_1y_mass_cterm_int, node_2y_mass_cterm_int])

    unique_values, inverse_indices = np.unique(graphnode_mass, return_inverse=True)
    unique_values = np.sort(unique_values)
    rel_inten = graphnode_mass_int
    sums = [np.sum(rel_inten[graphnode_mass == value]) for value in unique_values]
    return unique_values, np.array(sums)


def check_signature_ion(glycan: str) -> np.ndarray:
    signature_ion: List[float] = []
    if "G" in glycan:
        signature_ion.append(308.098)
        signature_ion.append(290.087)
        signature_ion.append(511.177)
    return np.array(signature_ion)


def read_mgf(path: str, glob_pattern: str = "*.mgf") -> Tuple[Dict[str, List[int]], Dict[int, int]]:
    """
    Read one or many MGF files and collect MS2 data and scan lookup.
    If `path` is a directory, all files matching `glob_pattern` are read.
    """
    queries = {"indices_ms2": [0], "mass_list_ms2": [], "int_list_ms2": [], "prec_mass_list2": [], "charge": []}
    scan2idx: Dict[int, int] = {}

    path_obj = Path(path)
    mgf_files: List[Path]
    if path_obj.is_dir():
        mgf_files = sorted(path_obj.glob(glob_pattern))
    else:
        mgf_files = [path_obj]

    if not mgf_files:
        raise FileNotFoundError(f"No MGF files found in {path} with pattern {glob_pattern}")

    current_idx = 0
    for mgf_file in mgf_files:
        for spec in mgf.read(str(mgf_file), convert_arrays=1):
            p = spec["params"]
            prec_mz = p.get("pepmass")[0]
            charge = p.get("charge")[0]
            precursor = (prec_mz - PROTON_MASS) * abs(charge)
            title = p.get("title")
            # print(title)
            raw_name = title.split('.')[0]
            m = re.search(r"\bscan=(\d+)\b", title, flags=re.IGNORECASE)
            scan2idx[raw_name +'.'+ str(int(m.group(1))) if m else None] = current_idx
            # print(raw_name +'.'+ str(int(m.group(1))) if m else None)
            queries["indices_ms2"].append(queries["indices_ms2"][-1] + spec["m/z array"].size)
            queries["mass_list_ms2"].append(spec["m/z array"])
            queries["int_list_ms2"].append(spec["intensity array"])
            queries["prec_mass_list2"].append(precursor)
            queries["charge"].append(charge)
            current_idx += 1
    queries["mass_list_ms2"] = np.concatenate(queries["mass_list_ms2"])
    queries["int_list_ms2"] = np.concatenate(queries["int_list_ms2"])
    return queries, scan2idx


def compute_glycan_scores(
    df: pd.DataFrame, queries: Dict[str, List[int]], scan2idx: Dict[int, int]) -> List[float]:
    """Compute glycan scores mirroring the original notebook logic."""
    query_indices = queries["indices_ms2"]
    query_frags = queries["mass_list_ms2"]
    query_ints = queries["int_list_ms2"]

    glycan_scores: List[float] = []
    for _, row in df.iterrows():
        spec_str = str(row["Spec"])
        spec_num = spec_str.replace("decoy", "")
        query_idx = scan2idx[spec_num]

        pep_mass = row["Pep mass"]
        theo_mass = pep_mass + find_fragments(row["predict"])
        charge = queries["charge"][query_idx]

        product_ions_moverz = query_frags[query_indices[query_idx] : query_indices[query_idx + 1]]
        product_ions_intensity = query_ints[query_indices[query_idx] : query_indices[query_idx + 1]]
        check_signature_ion(row["predict"])  # retained for parity; current scoring does not use it

        observe_mass, observe_mass_inten = graphnode_mass_generator(
            product_ions_moverz, product_ions_intensity, int(charge), pep_mass
        )
        low_index = observe_mass.searchsorted(theo_mass - 0.02)
        high_index = observe_mass.searchsorted(theo_mass + 0.02)

        glycan_score = 0.0
        matched = 0
        for idx, (low, high) in enumerate(zip(low_index, high_index)):
            if low < high:
                matched += 1
                for j in range(low, high):
                    glycan_score += np.log(observe_mass_inten[j]) * (
                        1 - (np.abs(theo_mass[idx] - observe_mass[j]) / 0.05) ** 4
                    )
        glycan_score *= matched / len(theo_mass)
        glycan_scores.append(glycan_score)
    return glycan_scores


def apply_fdr(df: pd.DataFrame, threshold: float) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[float]]:
    """
    Apply FDR estimation to the scored dataframe.

    Returns the dataframe with FDR annotations, the subset passing the threshold,
    and the score cutoff (None if no hits pass).
    """
    df = df.copy()
    df["base_spec"] = df["Spec"].str.replace("decoy", "", case=False)
    idx = df.groupby("base_spec")["glycan_score"].idxmax()
    df = df.loc[idx].reset_index(drop=True)
    df = df.drop(columns=["base_spec"])

    df["is_decoy"] = df["Spec"].str.contains("decoy", case=False)
    df = df.sort_values("glycan_score", ascending=False).reset_index(drop=True)
    df["cum_decoy"] = df["is_decoy"].cumsum()
    df["cum_target"] = (~df["is_decoy"]).cumsum()
    df["fdr"] = df["cum_decoy"] / df["cum_target"].clip(lower=1)

    passed = df[df["fdr"] <= threshold]
    if passed.empty:
        print(f"No hits pass {threshold*100:.2f}% FDR.")
        return df, df.iloc[0:0].copy(), None

    fdr_threshold = passed["glycan_score"].iloc[-1]
    print(f"{threshold*100:.2f}% FDR score threshold:", fdr_threshold)
    return df, df[df["glycan_score"] >= fdr_threshold].copy(), fdr_threshold


def resolve_fdr_cfg(cfg: Optional[DictConfig]) -> Dict[str, object]:
    """Merge Hydra-provided cfg.fdr with defaults."""
    fdr_cfg = cfg.get("fdr") if cfg else None

    def _get(key: str):
        if fdr_cfg and key in fdr_cfg and fdr_cfg[key] is not None:
            return fdr_cfg[key]
        return DEFAULT_FDR_CFG[key]

    return {k: _get(k) for k in DEFAULT_FDR_CFG}


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    fdr_cfg = resolve_fdr_cfg(cfg)

    input_csv = Path(to_absolute_path(Path(cfg.out_dir)/cfg.out_put_file))
    mgf_path = Path(to_absolute_path(cfg.out_dir))
    output_path = Path(to_absolute_path(Path(cfg.out_dir)/cfg.fdr.output))

    df = pd.read_csv(input_csv)
    df = df[df['mass difference'] < 0.02]
    print(len(df))
    df = df.dropna()

    queries, scan2idx = read_mgf(str(mgf_path))
    df["glycan_score"] = compute_glycan_scores(df, queries, scan2idx)

    _, passed_df, cutoff = apply_fdr(df, float(fdr_cfg["fdr_threshold"]))
    print(len(passed_df))

    passed_df.to_csv(output_path, index=False)
    print(f"Saved FDR-filtered results to {output_path}")
    if cutoff is None:
        print("No score cutoff determined; output will be empty.")


if __name__ == "__main__":
    main()
