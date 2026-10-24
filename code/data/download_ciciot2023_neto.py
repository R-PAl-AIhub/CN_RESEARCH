"""
Download the 46-feature CIC-IoT2023 release (schema used by the senior paper).

Original source : https://www.unb.ca/cic/datasets/iotdataset-2023.html (CIC, University of New Brunswick)
Dataset paper   : Neto et al., Sensors 23(13):5941, 2023, doi:10.3390/s23135941
Transport mirror: https://huggingface.co/datasets/lacg030175/CIC-IoT-2023-neto-full  (declared licence: CC-BY-4.0)
                  46,686,579 rows x 46 features; keeps the original `Label` column.
Only the `random` config (train + test parquet) is fetched; we ignore its split and re-split ourselves.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from pathlib import Path
import json
from huggingface_hub import HfApi, hf_hub_download

REPO = "lacg030175/CIC-IoT-2023-neto-full"
OUT = Path(__file__).resolve().parents[2] / "data" / "raw" / "ciciot2023_neto"
OUT.mkdir(parents=True, exist_ok=True)
info = HfApi().dataset_info(REPO)
files = ["random/train-00000-of-00001.parquet", "random/test-00000-of-00001.parquet", "README.md", "feature_importance.json"]
for f in files:
    p = hf_hub_download(REPO, f, repo_type="dataset", local_dir=str(OUT))
    print("ok", p, flush=True)
(OUT / "manifest.json").write_text(json.dumps({"repo": REPO, "commit_sha": info.sha, "files": files}, indent=1))
print("DONE commit", info.sha)
