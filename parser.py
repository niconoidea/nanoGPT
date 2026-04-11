"""
Parse nanoGPT training log files in logs/ and emit a JSON summary with:
  - final_val_loss
  - final_train_loss
  - num_parameters   (float, in millions)
  - total_flops      (int, raw value from log)
for every run found.
"""

import json
import re
from pathlib import Path

LOG_DIR = Path(__file__).parent / "logs"
OUTPUT_FILE = Path(__file__).parent / "results.json"

# Patterns
RE_DECAYED     = re.compile(r"num decayed parameter tensors:\s+\d+,\s+with\s+([\d,]+)\s+parameters")
RE_NON_DECAYED = re.compile(r"num non-decayed parameter tensors:\s+\d+,\s+with\s+([\d,]+)\s+parameters")
RE_STEP        = re.compile(r"step\s+\d+:\s+train loss\s+([\d.]+),\s+val loss\s+([\d.]+)")
RE_FLOPS       = re.compile(r"Total FLOPS:\s+(\d+)")
RE_SIZE_SPLIT  = re.compile(r"train_([^_]+)_split(\d+)\.log")


def parse_log(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")

    # --- model size / split from filename ---
    m = RE_SIZE_SPLIT.match(path.name)
    size  = m.group(1) if m else None
    split = int(m.group(2)) if m else None

    # --- number of parameters (decayed + non-decayed exact counts) ---
    dm  = RE_DECAYED.search(text)
    ndm = RE_NON_DECAYED.search(text)
    if dm and ndm:
        num_parameters = (
            int(dm.group(1).replace(",", "")) +
            int(ndm.group(1).replace(",", ""))
        )
    else:
        num_parameters = None

    # --- final train / val loss (last matching step line) ---
    step_matches = RE_STEP.findall(text)
    if step_matches:
        final_train_loss = float(step_matches[-1][0])
        final_val_loss   = float(step_matches[-1][1])
    else:
        final_train_loss = None
        final_val_loss   = None

    # --- total FLOPs ---
    fm = RE_FLOPS.search(text)
    total_flops = int(fm.group(1)) if fm else None

    return {
        "run":              path.stem,          # e.g. "train_XS_split10"
        "size":             size,               # e.g. "XS"
        "split_pct":        split,              # e.g. 10
        "num_parameters":   num_parameters,     # exact integer count
        "total_flops":      total_flops,
        "final_train_loss": final_train_loss,
        "final_val_loss":   final_val_loss,
    }


def main():
    log_files = sorted(LOG_DIR.glob("train_*.log"))
    if not log_files:
        print(f"No log files found in {LOG_DIR}")
        return

    results = [parse_log(f) for f in log_files]

    OUTPUT_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {len(results)} entries to {OUTPUT_FILE}")
    for r in results:
        status = "complete" if r["total_flops"] is not None else "incomplete"
        print(
            f"  {r['run']:<25}  params={r['num_parameters']}"
            f"  train={r['final_train_loss']}  val={r['final_val_loss']}"
            f"  flops={r['total_flops']}  [{status}]"
        )


if __name__ == "__main__":
    main()
