import re
from pathlib import Path

import hydra
import numpy as np
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

# only keep those meeting n-glycan ladder ions
# ────────────────────────────────────────────────────────────────────────────
SCAN_REGEX = re.compile(r'scan=(\d+)', re.IGNORECASE)

def process_mgf(
    in_path: Path,
    out_path: Path,
    sig_mz,
    ppm_tol: float,
    min_rel_intensity: float = 0.0,
) -> None:
    """
    Read `in_path` MGF and only keep spectra with n_matches >= 3 signature ions.
    """
    sig_mz = np.asarray(sig_mz, dtype=float)

    with in_path.open("r") as fin, open(out_path, "w") as fout:
        block_lines = []
        product_ions_moverz = []
        product_ions_intensity = []

        mz = None
        z = None
        scan = None

        def flush_block_if_kept():
            nonlocal block_lines, product_ions_moverz, product_ions_intensity
            nonlocal mz, z, scan

            # Basic sanity
            if mz is None or z is None or scan is None:
                return
            mz_arr = np.asarray(product_ions_moverz, dtype=float)
            I_arr = np.asarray(product_ions_intensity, dtype=float)

            if mz_arr.size == 0:
                return

            # Relative intensity filter
            base = I_arr.max() if I_arr.size and I_arr.max() > 0 else 1.0
            rel = I_arr / base
            keep = rel >= float(min_rel_intensity)
            mz_keep = mz_arr[keep]
            if mz_keep.size == 0:
                # No peaks survive intensity threshold; discard spectrum
                return

            # Oxonium matching (broadcast), count distinct signature ions matched
            diff_ppm = np.abs((mz_keep[:, None] - sig_mz[None, :]) / sig_mz[None, :]) * 1e6
            matched_per_sig = (diff_ppm <= ppm_tol).any(axis=0)  # bool array over signature list
            n_matches = int(matched_per_sig.sum())

            if n_matches < 3:
                # Discard spectrum
                return

            # If we are here → keep the spectrum.
            for line in block_lines:
                fout.write(line)

        for raw_line in fin:
            line = raw_line.rstrip("\n")

            # Start of a new block
            if line.startswith("BEGIN IONS"):
                # Flush previous block (if any)
                if block_lines:
                    flush_block_if_kept()

                # Reset for new block
                block_lines = [raw_line]
                product_ions_moverz = []
                product_ions_intensity = []
                mz = None
                z = None
                scan = None

            # End of block
            elif line.startswith("END IONS"):
                block_lines.append(raw_line)
                # Flush current block with filtering
                flush_block_if_kept()

                # Reset block buffer
                block_lines = []
                product_ions_moverz = []
                product_ions_intensity = []
                mz = None
                z = None
                scan = None

            else:
                # Inside a block: accumulate line and parse fields/peaks
                block_lines.append(raw_line)

                if line.startswith("PEPMASS"):
                    # PEPMASS=936.33 12345.0
                    parts = re.split(r"[=\s]+", line)
                    # parts[0] = 'PEPMASS', parts[1] = mz
                    if len(parts) >= 2:
                        try:
                            mz = float(parts[1])
                        except ValueError:
                            mz = None

                elif line.startswith("CHARGE"):
                    # CHARGE=4+
                    m = re.search(r"CHARGE=(\d+)\+", line)
                    if m:
                        try:
                            z = int(m.group(1))
                        except ValueError:
                            z = None

                elif line.startswith("TITLE="):
                    # TITLE=... scan=123"
                    m = SCAN_REGEX.search(line)
                    if m:
                        scan = m.group(1)

                else:
                    # Try to parse as peak line: "<mz> <intensity>"
                    # Skip metadata lines with '=' etc.
                    if line and "=" not in line and not line.startswith("#"):
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                mz_val = float(parts[0])
                                I_val = float(parts[1])
                            except ValueError:
                                pass
                            else:
                                product_ions_moverz.append(mz_val)
                                product_ions_intensity.append(I_val)

        # In case file doesn't end with newline after END IONS (rare),
        # but block_lines still has content without an END IONS.
        if block_lines:
            flush_block_if_kept()


def run(
    in_directory: Path,
    out_directory: Path,
    sig_mz,
    ppm_tol: float,
    min_rel_intensity: float,
    glob_pattern: str = "*.mgf",
) -> None:
    mgf_dir = Path(in_directory).expanduser().resolve()
    out_dir = Path(out_directory).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    mgf_files = list(mgf_dir.glob(glob_pattern))

    for mgf in mgf_files:
        out_file = out_dir / mgf.name
        process_mgf(
            mgf,
            out_file,
            sig_mz=sig_mz,
            ppm_tol=ppm_tol,
            min_rel_intensity=min_rel_intensity,
        )
        print(f"✅  Wrote {out_file}")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: resolves configured paths and runs preprocessing.
    """
    in_dir = Path(to_absolute_path(cfg.in_dir))
    out_dir = Path(to_absolute_path(cfg.out_dir))
    run(
        in_dir,
        out_dir,
        sig_mz=cfg.mgf_preprocess.sig_mz,
        ppm_tol=cfg.mgf_preprocess.ppm_tol,
        min_rel_intensity=cfg.mgf_preprocess.min_rel_intensity,
        glob_pattern=cfg.mgf_preprocess.glob_pattern,
    )


if __name__ == "__main__":
    main()
