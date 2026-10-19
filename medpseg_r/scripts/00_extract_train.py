"""Extract the 239 training volumes (images + labels) of the AirMorph-R split from the
ATM'26 Task-1 zip. Only train IDs are extracted; val/test already exist elsewhere and the
test volumes must never enter the training folder."""
import json, os, sys, zipfile, hashlib
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SPLIT = os.path.join(REPO, "splits", "atm26_split.json")
DATA = os.environ.get("MEDPSEG_R_DATA", os.path.join(REPO, "medpseg_r", "data"))
ZIP = os.environ["ATM26_TASK1_ZIP"]   # ATM26-Task-1-Binary-Airway-Segmentation-TrainBatch.zip
OUT = os.path.join(DATA, "ATM26")
sp = json.load(open(SPLIT))
train = sorted(c.split("/")[1] for c in sp["train"])
held = set(c.split("/")[1] for c in sp["val"] + sp["test"])
assert len(train) == 239 and not (set(train) & held)
z = zipfile.ZipFile(ZIP)
names = z.namelist()
todo = []
for cid in train:
    img = [n for n in names if n.endswith(f"imagesTr/{cid}_0000.mha")]
    lab = [n for n in names if n.endswith(f"labelsTr/{cid}.mha")]
    assert len(img) == 1 and len(lab) == 1, (cid, img, lab)
    todo += [(img[0], f"{OUT}/imagesTr/{cid}_0000.mha"), (lab[0], f"{OUT}/labelsTr/{cid}.mha")]
os.makedirs(f"{OUT}/imagesTr", exist_ok=True); os.makedirs(f"{OUT}/labelsTr", exist_ok=True)
for i, (src, dst) in enumerate(todo):
    info = z.getinfo(src)
    if os.path.exists(dst) and os.path.getsize(dst) == info.file_size:
        continue
    with z.open(src) as fi, open(dst + ".part", "wb") as fo:
        while True:
            b = fi.read(1 << 24)
            if not b: break
            fo.write(b)
    assert os.path.getsize(dst + ".part") == info.file_size
    os.replace(dst + ".part", dst)
    print(f"{i+1}/{len(todo)} {os.path.basename(dst)}", flush=True)
json.dump({"train_ids": train, "source_zip": ZIP, "split": SPLIT}, open(f"{OUT}/MANIFEST.json", "w"), indent=1)
print("DONE", len(todo))
