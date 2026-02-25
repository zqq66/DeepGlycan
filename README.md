## Workflow Runner

Use the helper script to execute the full pipeline (mgf preprocessing → Hyper-Spec clustering → peptide search → inference → FDR) with a single command:

```bash
bash run_workflow.sh \
  in_dir=/path/to/mgf_directory \
  out_dir=/path/to/output_directory 
```

Notes:
- The script reads defaults from `configs/config.yaml`; CLI overrides take precedence.
- `out_dir` defaults to `in_dir` if not provided. `working_dir` defaults to the same MGF/output directory.
- You can pass any Hydra override in the same form (e.g., `mgf_preprocess.ppm_tol=15.0`).
- Run `bash run_workflow.sh -h` for a brief help message.

## Environment

Python dependencies are in `requirements.txt`.
If you prefer conda, a minimal `environment.yml` is also provided.

Hyper-Spec clustering
- For large-scale analysis we use Hyper-Spec to cluster spectra and help locate peptides. Install Hyper-Spec from https://github.com/wh-xu/Hyper-Spec (clone or install from source).
- `run_workflow.sh` assumes `Hyper-Spec` is available (e.g., cloned alongside this repo or adjust the script to point to your install path).

The model checkpoint and example files are available at the following Google Drive folder:
https://drive.google.com/drive/folders/1VU26ol_vrD6ZKxQ9wVzJUrOV52jFxg1V?usp=sharing

We also provide the cluster file and the preprocessed MGF file in the same folder to help reviewers run our code more easily.

If the peptide sequences are already known and only de novo glycan sequencing is required, please refer to:
https://github.com/zqq66/DeepGlycanEval
