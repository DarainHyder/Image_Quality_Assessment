"""Build unified training CSVs from the raw IQA databases.

Expected layout under --root (archives from the pyiqa / IQA dataset mirrors on the HF Hub):

    koniq/512x384/*.jpg                    + meta_info_KonIQ10kDataset.csv, meta_info_KonIQ10k++Dataset.csv
    kadid10k/images/*.png                  + meta_info_KADID10kDataset.csv
    spaq/... (any depth)                   + meta_info_SPAQDataset.csv     -> resized into spaq_512/
    live_challenge/... (any depth)         + meta_info_LIVEChallengeDataset.csv

Writes koniq.csv / kadid.csv / spaq.csv / livec.csv next to the data, with columns
``path, mos, split, d_<defect>..., a_<attr>...`` (paths relative to --root).
"""
from __future__ import annotations

import argparse
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lumina.constants import ATTRS, DEFECTS, KADID_DEFECT  # noqa: E402
from lumina.data import to_canvas  # noqa: E402


def blank(n: int) -> dict:
    return {f"d_{k}": [np.nan] * n for k in DEFECTS} | {f"a_{k}": [np.nan] * n for k in ATTRS}


def find_dir(root: Path, sample: str) -> Path | None:
    hits = list(root.rglob(sample))
    return hits[0].parent if hits else None


def koniq(root: Path) -> None:
    meta = pd.read_csv(root / "meta_info_KonIQ10kDataset.csv")
    df = pd.DataFrame({"path": "koniq/512x384/" + meta["img_name"], "mos": meta["mos"], "split": meta["official_split"]})
    df = df.assign(**blank(len(df)))
    pp_csv = root / "meta_info_KonIQ10k++Dataset.csv"
    if pp_csv.exists():  # KonIQ++: fraction of raters flagging each defect, joined by filename
        pp = pd.read_csv(pp_csv).set_index("filename")
        for src, dst in (("blur", "blur"), ("artifacts", "artifacts"), ("contrast", "contrast"), ("colors", "color")):
            df[f"d_{dst}"] = meta["img_name"].map(pp[src]).to_numpy()
    df.to_csv(root / "koniq.csv", index=False)
    print("koniq", df["split"].value_counts().to_dict(), "defect-labelled", int(df["d_blur"].notna().sum()))


def kadid(root: Path) -> None:
    img_dir = find_dir(root / "kadid10k", "I01_01_01.png")
    if img_dir is None:
        print("kadid: images not found, skipped"); return
    meta = pd.read_csv(root / "meta_info_KADID10kDataset.csv")
    rel = img_dir.relative_to(root).as_posix()
    df = pd.DataFrame({"path": rel + "/" + meta["dist_name"], "mos": meta["dmos"] * 20.0})
    refs = sorted(meta["ref_name"].unique())
    random.Random(0).shuffle(refs)
    test_refs = set(refs[: len(refs) // 5])  # reference-disjoint 80/20 split
    df["split"] = np.where(meta["ref_name"].isin(test_refs), "test", "train")
    df = df.assign(**blank(len(df)))
    for c in DEFECTS:
        df[f"d_{c}"] = 0.0
    for i, name in enumerate(meta["dist_name"]):
        t, lv = int(name[4:6]), int(name[7:9])
        df.at[i, f"d_{KADID_DEFECT[t]}"] = lv / 5.0
    # reference (pristine) images: all defects absent
    ref_rows = pd.DataFrame({"path": [f"{rel}/{r}" for r in refs], "mos": 100.0,
                             "split": ["test" if r in test_refs else "train" for r in refs]})
    ref_rows = ref_rows.assign(**blank(len(ref_rows)))
    for c in DEFECTS:
        ref_rows[f"d_{c}"] = 0.0
    ref_rows = ref_rows[[(img_dir / r).exists() for r in refs]]
    ref_rows["mos"] = np.nan  # no MOS for references — defect supervision only
    df = pd.concat([df, ref_rows], ignore_index=True)
    df.to_csv(root / "kadid.csv", index=False)
    print("kadid", df["split"].value_counts().to_dict())


def _resize_one(job):
    src, dst = job
    if dst.exists():
        return
    img = Image.open(src).convert("RGB")
    img.thumbnail((2048, 2048), Image.BILINEAR)  # cheap pre-shrink of 4000px phone shots
    to_canvas(img).save(dst, "PNG", compress_level=1)


def spaq(root: Path, workers: int) -> None:
    meta = pd.read_csv(root / "meta_info_SPAQDataset.csv")
    src_dir = find_dir(root / "spaq", meta["Image name"].iloc[0])
    if src_dir is None:
        print("spaq: images not found, skipped"); return
    out = root / "spaq_512"
    out.mkdir(exist_ok=True)
    jobs = [(src_dir / n, out / (Path(n).stem + ".png")) for n in meta["Image name"]]
    with ProcessPoolExecutor(workers) as ex:
        list(ex.map(_resize_one, jobs, chunksize=32))
    df = pd.DataFrame({"path": ["spaq_512/" + Path(n).stem + ".png" for n in meta["Image name"]], "mos": meta["MOS"],
                       "split": meta["ratio802_seed123_split_01"].map({"train": "train", "val": "test"})})
    df = df.assign(**blank(len(df)))
    for a, col in zip(ATTRS, ("Brightness", "Colorfulness", "Contrast", "Noisiness", "Sharpness")):
        df[f"a_{a}"] = meta[col]
    df.to_csv(root / "spaq.csv", index=False)
    print("spaq", df["split"].value_counts().to_dict())


def livec(root: Path) -> None:
    meta = pd.read_csv(root / "meta_info_LIVEChallengeDataset.csv")
    # images are split across Images/ and Images/trainingImages/ — resolve each one, preferring the top level
    found = {}
    for f in sorted((root / "live_challenge").rglob("*"), key=lambda f: -len(f.parts)):
        found[f.name] = f.relative_to(root).as_posix()
    if meta["img_name"].iloc[0] not in found:
        print("livec: images not found, skipped"); return
    df = pd.DataFrame({"path": meta["img_name"].map(found), "mos": meta["mos"], "split": "test"})
    df.assign(**blank(len(df))).to_csv(root / "livec.csv", index=False)
    print("livec", len(df))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--only", nargs="*", default=["koniq", "kadid", "spaq", "livec"])
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    for name in a.only:
        if name == "spaq":
            spaq(a.root, a.workers)
        else:
            globals()[name](a.root)
