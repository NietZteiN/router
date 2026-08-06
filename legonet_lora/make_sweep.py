"""Generate Phase-3 sweep cell configs + a single training manifest.

Cells (all on the v2 recipe = canary_repeat 5, 6 epochs, rank 16; shared corpus
dbpedia_n4000). The anchor cell (knn, n=32, k=3) is the already-trained v2 run and
is NOT regenerated — collect_sweep pulls it in.

  n-sweep (k=3):     knn n16 k3, knn n64 k3              [+ v2 n32]
  k-sweep (n=32):    knn n32 k1, knn n32 k5             [+ v2 k3]
  SISA baseline:     random n32 k1 (s=32), random n64 k1 (s=64)
  ablation:          knn n32 k1  vs  random n32 k1  = LegoNet_{k=1} vs FixSISA

Writes configs/sweep/*.json, sweep_manifest.txt ("config_path adapter_idx" per
line, one line per adapter to train), and sweep_cells.txt (one config path/cell).

    python make_sweep.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = json.load(open(os.path.join(HERE, "configs", "legonet_7b_v2.json")))

CELLS = [
    ("knn", 16, 3), ("knn", 64, 3),       # n-sweep @ k=3
    ("knn", 32, 1), ("knn", 32, 5),       # k-sweep @ n=32
    ("random", 32, 1), ("random", 64, 1),  # SISA shards (s=n)
]


def main():
    outdir = os.path.join(HERE, "configs", "sweep")
    os.makedirs(outdir, exist_ok=True)
    manifest, cells = [], []
    for mode, n, k in CELLS:
        cfg = json.loads(json.dumps(BASE))  # deep copy
        cfg["name"] = f"sweep_{mode}_n{n}_k{k}"
        cfg["n"] = n
        cfg["k"] = k
        cfg["assignment_mode"] = mode
        path = os.path.join(outdir, f"{cfg['name']}.json")
        with open(path, "w") as f:
            json.dump(cfg, f, indent=2)
        cells.append(path)
        for j in range(n):
            manifest.append(f"{path} {j}")

    with open(os.path.join(HERE, "sweep_manifest.txt"), "w") as f:
        f.write("\n".join(manifest) + "\n")
    with open(os.path.join(HERE, "sweep_cells.txt"), "w") as f:
        f.write("\n".join(cells) + "\n")
    print(f"{len(cells)} cells, {len(manifest)} adapters to train")
    for c in cells:
        print("  ", os.path.basename(c))


if __name__ == "__main__":
    main()
