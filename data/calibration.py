#!/usr/bin/env python3
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Collection

import numpy as np
import pandas as pd
from pyteomics import mzml

PROTON = 1.007276466812
C13_C12_DIFF = 1.0033548378  # Da


# ----------------------- Data containers ----------------------- #


@dataclass
class MS1Spectrum:
    scan_id: str
    mz: np.ndarray
    intensity: np.ndarray
    rt: Optional[float] = None

@dataclass
class MS2Spectrum:
    scan_id: str
    precursor_mz: float
    precursor_charge: int
    parent_ms1_id: str
    mz: np.ndarray
    intensity: np.ndarray
    rt: Optional[float] = None

# ----------------- Oxonium-based ppm calibration ----------------- #

OXONIUM_MZ = np.array([
    204.0867,  # HexNAc+
    186.0761,  # HexNAc - H2O
    163.0601,  # Hex+
    274.0921,  # Neu5Ac - H2O
    292.1027,  # Neu5Ac+
    138.055
])

def estimate_oxonium_mass_error(
    ms2_mz: np.ndarray,
    ms2_int: np.ndarray,
    ppm_tolerance: float = 20.0,
    intensity_quantile: float = 0.8,
) -> Optional[float]:
    """
    Estimate per-spectrum mass error (ppm) from oxonium ions in MS2.
    """
    if ms2_mz.size == 0:
        return None

    ppm_errors = []
    int_thresh = np.quantile(ms2_int, intensity_quantile) if ms2_int.size > 0 else 0.0

    for theo in OXONIUM_MZ:
        tol = theo * ppm_tolerance * 1e-6
        lo, hi = theo - tol, theo + tol

        idx = np.where((ms2_mz >= lo) & (ms2_mz <= hi))[0]
        if idx.size == 0:
            continue

        # most intense peak in the window
        best_idx = idx[np.argmax(ms2_int[idx])]
        if ms2_int[best_idx] < int_thresh:
            continue

        obs = ms2_mz[best_idx]
        ppm = (obs - theo) / theo * 1e6
        ppm_errors.append(ppm)
        print(obs, obs - theo)
    if len(ppm_errors) < 2:
        return None
    return float(np.median(ppm_errors))

def apply_ppm_shift(mz: float, ppm: float) -> float:
    return mz * (1.0 - ppm * 1e-6)

# ---------------------- mzML parsing ---------------------- #

def load_ms1_ms2_from_mzml(
    path: str,
    allowed_ms2_ids: Optional[Collection[str]] = None,
) -> Tuple[Dict[str, MS1Spectrum], List[MS2Spectrum]]:
    """
    Parse mzML and return MS1 dict + subset of MS2 scans.

    allowed_ms2_ids:
      - None  -> keep ALL MS2.
      - list/set of ids -> keep only MS2 whose 'id' is in that collection.
    """
    ms1_dict: Dict[str, MS1Spectrum] = {}
    ms2_list: List[MS2Spectrum] = []

    last_ms1_id: Optional[str] = None
    allowed_set = set(allowed_ms2_ids) if allowed_ms2_ids is not None else None

    with mzml.MzML(path) as reader:
        for spec in reader:
            ms_level = spec.get('ms level')
            spec_id = spec.get('id')
            mz = spec['m/z array']
            inten = spec['intensity array']

            # optional RT
            rt = None
            scan_list = spec.get('scanList')
            if scan_list and scan_list.get('scan'):
                rt = scan_list['scan'][0].get('scan start time')

            if ms_level == 1:
                ms1_dict[spec_id] = MS1Spectrum(
                    scan_id=spec_id, mz=mz, intensity=inten, rt=rt
                )
                last_ms1_id = spec_id

            elif ms_level == 2:
                if allowed_set is not None and spec_id not in allowed_set:
                    continue

                prec_list = spec.get('precursorList', {}).get('precursor', [])
                if not prec_list:
                    continue
                prec = prec_list[0]
                sel_list = prec.get('selectedIonList', {}).get('selectedIon', [])
                if not sel_list:
                    continue
                sel = sel_list[0]

                precursor_mz = float(sel.get('selected ion m/z'))
                charge = int(sel.get('charge state', 0) or 0)

                parent_ref = prec.get('spectrumRef', None)
                if parent_ref is None:
                    parent_ref = last_ms1_id

                if parent_ref is None:
                    continue

                ms2_list.append(
                    MS2Spectrum(
                        scan_id=spec_id,
                        precursor_mz=precursor_mz,
                        precursor_charge=charge,
                        parent_ms1_id=parent_ref,
                        mz=mz,
                        intensity=inten,
                        rt=rt,
                    )
                )

    return ms1_dict, ms2_list

# ----------------- Calibration (oxonium only) ----------------- #

def calibrate_precursors_oxonium_only(
    ms2_list: List[MS2Spectrum],
    file_label: Optional[str] = None,
) -> List[Dict[str, float]]:
    """
    For each MS2:
      - estimate ppm error from oxonium ions
      - apply that ppm to precursor m/z
      - convert both old and new m/z to neutral mass

    No monoisotopic / isotope-envelope correction here.
    """
    results = []

    for spec in ms2_list:
        z = spec.precursor_charge
        if z <= 0:
            continue

        reported_mz = spec.precursor_mz# + 1 * (C13_C12_DIFF / z)
        reported_mass = (reported_mz - PROTON) * z
        print(spec.scan_id)
        ox_ppm = estimate_oxonium_mass_error(
            spec.mz,
            spec.intensity,
            ppm_tolerance=20.0,
            intensity_quantile=0.8,
        )

        if ox_ppm is not None:
            corrected_mz = apply_ppm_shift(reported_mz, ox_ppm)
        else:
            corrected_mz = reported_mz  # no change if we couldn't estimate

        corrected_mass = (corrected_mz - PROTON) * z

        results.append({
            "file": file_label if file_label is not None else "",
            "ms2_scan": spec.scan_id,
            "z": z,

            "reported_precursor_mz": reported_mz,
            "corrected_precursor_mz": corrected_mz,

            "reported_precursor_mass": reported_mass,
            "corrected_precursor_mass": corrected_mass,

            "oxonium_ppm_shift": ox_ppm if ox_ppm is not None else np.nan,
        })

    return results

# -------- Directory-level driver with dict of scan IDs -------- #

def calibrate_mzml_directory_oxonium(
    mzml_dir: str,
    selected_scans: Dict[str, List[str]],
) -> pd.DataFrame:
    """
    mzml_dir:
        directory containing mzML files.

    selected_scans:
        dict mapping file name -> list of MS2 spectrum ids to calibrate.
        e.g. {"run1.mzML": ["controllerType=0 controllerNumber=1 scan=1234", ...]}
    """
    all_rows = []

    for fname, id_list in selected_scans.items():
        path = Path(mzml_dir) / fname
        if not path.exists():
            print(f"[WARN] mzML file not found: {path}")
            continue

        print(f"Processing {path} with {len(id_list)} selected MS2 scans")

        _, ms2_list = load_ms1_ms2_from_mzml(str(path), allowed_ms2_ids=id_list)
        print(f"  -> Loaded {len(ms2_list)} selected MS2")

        if not ms2_list:
            continue

        rows = calibrate_precursors_oxonium_only(
            ms2_list,
            file_label=fname,
        )
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    return df

# -------------------------- Example usage -------------------------- #

if __name__ == "__main__":
    # Example: fill this dict however you like (from CSV, cluster output, etc.)
    # Keys = mzML file names (basename), values = list of MS2 spectrum IDs in that file.
    selected_scans = {
        "MOUSE-MALE-Y4-HEART_calibrated.mzML": [
            "scan=7310",
            "scan=7081",
        ],
        "MOUSE-MALE-Y3-HEART_calibrated.mzML": [
            "scan=7080",
            "scan=7304",
        ],
        "MOUSE-MALE-Y2-HEART_calibrated.mzML": [
            "scan=6894",
            "scan=7124",
        ],
        "MOUSE-MALE-Y1-HEART_calibrated.mzML": [
            "scan=7559",
        ],
        "MOUSE-MALE-O1-HEART_calibrated.mzML": [
            "scan=7275",
            "scan=7541",
        ],
        "MOUSE-MALE-O2-HEART_calibrated.mzML": [
            "scan=6932",
            "scan=7156"
        ],
        "MOUSE-MALE-O3-HEART_calibrated.mzML": [
            "scan=6850",
        ],
        "MOUSE-MALE-O4-HEART_calibrated.mzML": [
            "scan=6874","scan=7255"
        ],
    }

    base_dir = "/home/q359zhan/olinked/data/ethcd/heart/mzml"

    df = calibrate_mzml_directory_oxonium(base_dir, selected_scans)
    out_path = "oxonium_only_precursor_calibrated.tsv"
    df.to_csv(out_path, sep="\t", index=False)
    print(f"Wrote {out_path} with {len(df)} calibrated precursors.")
