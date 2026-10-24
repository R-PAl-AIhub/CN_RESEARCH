"""
One-command reproduction (CPU only, run inside ./.venv):      python code/run_all.py [--skip-download]

Steps (each skips work whose output already exists, so an interrupted run resumes):
  1  download real datasets (CIC-IoT2023 via HF mirror of the official release; N-BaIoT from UCI) and extract N-BaIoT attack rars
  2  build the working pools (seeded samples of the real data)
  3  Phase 1: run the UNCHANGED senior code on a real 500k sample + TensorFlow port check + gradient unit test
  4  Phase 4: tune (validation only) -> final runs (10 seeds, test once) -> sensitivity -> cost profile
  5  Phase 4.5: diagrams, tables, figures;  references.bib;  paper PDF
"""
import os
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8", "CUDA_VISIBLE_DEVICES": "-1"}


def run(cmd, cwd=ROOT, label=None, optional=False):
    print(f"\n=== {label or ' '.join(map(str, cmd))}", flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=cwd, env=ENV)
    if r.returncode and not optional:
        sys.exit(f"step failed: {label or cmd}")


def missing(*paths):
    return not all((ROOT / p).exists() for p in paths)


def main():
    skip_dl = "--skip-download" in sys.argv
    if not skip_dl:
        if missing("data/raw/ciciot2023_neto/random/train-00000-of-00001.parquet"):
            run([PY, "code/data/download_ciciot2023_neto.py"], label="download CIC-IoT2023 (46-feature release)")
        if missing("data/raw/nbaiot/nbaiot.zip"):
            url = "https://archive.ics.uci.edu/static/public/442/detection+of+iot+botnet+attacks+n+baiot.zip"
            (ROOT / "data/raw/nbaiot").mkdir(parents=True, exist_ok=True)
            run(["curl", "--http1.1", "-L", "-o", "data/raw/nbaiot/nbaiot.zip", url], label="download N-BaIoT (1.77 GB; server does not support resume)")
        if missing("data/raw/nbaiot/extracted/Danmini_Doorbell/mirai_attacks/ack.csv"):
            print("\n=== extract N-BaIoT attack CSVs from nested rar archives (needs Windows bsdtar: C:/Windows/System32/tar.exe)")
            import zipfile
            ext = ROOT / "data/raw/nbaiot/extracted"
            z = zipfile.ZipFile(ROOT / "data/raw/nbaiot/nbaiot.zip")
            for n in z.namelist():
                if n.endswith(".rar"):
                    z.extract(n, ext)
            for dev in ext.iterdir():
                if not dev.is_dir():
                    continue
                for botnet, files in (("gafgyt", ["combo.csv", "junk.csv", "tcp.csv", "udp.csv"]), ("mirai", ["ack.csv", "syn.csv", "udp.csv", "udpplain.csv"])):
                    rar = dev / f"{botnet}_attacks.rar"
                    if rar.exists():
                        (dev / f"{botnet}_attacks").mkdir(exist_ok=True)
                        subprocess.run(["C:/Windows/System32/tar.exe", "-xf", str(rar), "-C", str(dev / f"{botnet}_attacks"), *files], check=True)
    if missing("data/processed/ciciot_pool.parquet"):
        run([PY, "code/data/build_ciciot_pool.py"], label="build CIC-IoT2023 pool")
    if missing("data/processed/nbaiot_pool.parquet"):
        run([PY, "code/data/build_nbaiot_pool.py"], label="build N-BaIoT pool")

    # Phase 1
    if missing("code/senior_repro/outputs/results/experiment_results.json"):
        (ROOT / "data/repro/cic-iot-2023").mkdir(parents=True, exist_ok=True)
        run([PY, "-c", (
            "import pandas as pd;p=pd.read_parquet('data/processed/ciciot_pool.parquet');"
            "b=p[p.is_attack==0].sample(n=255000,random_state=42);d=p[p.is_attack==1].sample(n=245000,random_state=42);"
            "df=pd.concat([b,d]).sample(frac=1,random_state=42).reset_index(drop=True).rename(columns={'Label':'label'}).drop(columns=['is_attack']);"
            "df.to_csv('data/repro/cic-iot-2023/cic_iot_2023_sample500k.csv',index=False)")], label="write the 500k real sample for the senior code")
        run([PY, "main_experiment.py"], cwd=ROOT / "code/senior_repro", label="senior code (unchanged) on real data")
    run([PY, "code/phase1_tf_port_check.py"], label="TensorFlow port check")
    run([PY, "code/tests/test_gradients.py"], label="gradient unit test")

    # Phase 4
    run([PY, "code/run_experiments.py", "all", "--workers", "4", "--seeds", "10"], label="tune + final + sensitivity")
    if missing("results/cost_profile.json"):
        run([PY, "code/run_cost.py"], label="cost profile")

    # Phase 4.5 / 5
    run([PY, "code/plots/diagrams.py"], label="diagrams")
    run([PY, "code/plots/make_tables.py"], label="tables")
    run([PY, "code/plots/make_figures.py"], label="figures")
    run([PY, "code/data/verify_refs.py"], label="verify references", optional=True)
    run([PY, "code/data/make_bib.py"], label="references.bib", optional=True)
    if (ROOT / "tools/tectonic.exe").exists():
        run([ROOT / "tools/tectonic.exe", "main.tex"], cwd=ROOT / "paper", label="compile paper", optional=True)
    print("\nALL DONE")


if __name__ == "__main__":
    main()
