"""
Runs train.py for every combination of model size (XS, S, M, L) and
dataset split (10, 25, 50, 100 %) and saves stdout/stderr to a log file.

Log files are written to: logs/train_<SIZE>_split<N>.log

Usage:
    python loop.py
"""

import subprocess
import sys
from pathlib import Path

SIZES = ['XS', 'S', 'M', 'L']
SPLITS = [10, 25, 50, 100]

LOG_DIR = Path('logs')
LOG_DIR.mkdir(exist_ok=True)

total = len(SIZES) * len(SPLITS)
run = 0

for size in SIZES:
    for split in SPLITS:
        run += 1
        config_file = f'config/train_shakespeare_char_{size}.py'
        out_dir = f'out-shakespeare-{size}-split{split}'
        run_name = f'{size}-split{split}'
        train_file = f'train_{split}.bin'
        log_file = LOG_DIR / f'train_{size}_split{split}.log'

        cmd = [
            sys.executable, 'train.py',
            config_file,
            f'--train_file={train_file}',
            f'--out_dir={out_dir}',
            f'--wandb_run_name={run_name}',
        ]

        header = (
            f"{'='*70}\n"
            f"Run {run}/{total}  |  size={size}  split={split}%\n"
            f"Config : {config_file}\n"
            f"Out dir: {out_dir}\n"
            f"Log    : {log_file}\n"
            f"{'='*70}\n"
        )
        print(header, flush=True)

        with open(log_file, 'w', encoding='utf-8') as log:
            log.write(header)
            log.write(f"Command: {' '.join(cmd)}\n\n")
            log.flush()

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
            )
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            process.wait()

            footer = f"\nExit code: {process.returncode}\n"
            print(footer, flush=True)
            log.write(footer)

        if process.returncode != 0:
            print(f"WARNING: run {size}/split{split} finished with non-zero exit code {process.returncode}", flush=True)

print(f"\nAll {total} runs complete. Logs saved to '{LOG_DIR}/'.")

