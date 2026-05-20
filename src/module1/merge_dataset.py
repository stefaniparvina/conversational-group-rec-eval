"""
merge_dataset.py
================
Merges 4 per-model subfolders (each with 2000 group_simulation_X.json files)
into a single folder with unique, sequential filenames and group_id values.

Project layout this script assumes:
  conversational-group-rec-eval/
    src/module1/merge_dataset.py    ← this script
    data/raw/gemma/                 ← 2,000 files (group_ids 1-2000)
    data/raw/llama/                 ← 2,000 files (group_ids 2001-4000)
    data/raw/mistral/               ← 2,000 files (group_ids 4001-6000)
    data/raw/olmo/                  ← 2,000 files (group_ids 6001-8000)
    data/full_dataset/              ← output, 8,000 files, sequential IDs

Output layout:
  data/full_dataset/
    group_simulation_1.json     (from gemma,   group_id = 1)
    group_simulation_2.json     (from gemma,   group_id = 2)
    ...
    group_simulation_2001.json  (from llama,   group_id = 2001)
    ...
    group_simulation_8000.json  (from olmo,    group_id = 8000)

Usage:
    From anywhere inside the project:
        python src/module1/merge_dataset.py

    Paths are anchored to this script's own location, so it works
    regardless of which directory you run it from. To re-point at
    different folders, edit the SUBFOLDERS and OUTPUT_FOLDER constants
    below.

The original files are never modified. Everything is written to
data/full_dataset/, and the only field changed in each JSON is group_id.
"""

import json
import shutil
from pathlib import Path

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# Paths below are relative to this script's location (src/module1/).
# Two levels up (../../) lands in the project root, then into data/.
# Order matters: first folder gets IDs 1–2000, second 2001–4000, etc.

SUBFOLDERS = [
    "../../data/raw/gemma",
    "../../data/raw/llama",
    "../../data/raw/mistral",
    "../../data/raw/olmo",
]

OUTPUT_FOLDER = "../../data/full_dataset"

# How many files per subfolder (change if yours differ)
FILES_PER_SUBFOLDER = 2000
# ── END CONFIGURATION ─────────────────────────────────────────────────────────


def merge():
    script_dir = Path(__file__).parent
    out_dir = script_dir / OUTPUT_FOLDER
    out_dir.mkdir(exist_ok=True)

    total_written = 0
    total_skipped = 0

    for folder_index, subfolder_name in enumerate(SUBFOLDERS):
        subfolder = Path(subfolder_name)
        if not subfolder.is_absolute():
            subfolder = script_dir / subfolder_name

        if not subfolder.exists():
            print(f"  [SKIP] Folder not found: {subfolder}")
            continue

        id_offset = folder_index * FILES_PER_SUBFOLDER
        print(f"\nProcessing: {subfolder.name}  (IDs {id_offset + 1} – {id_offset + FILES_PER_SUBFOLDER})")

        json_files = sorted(subfolder.glob("group_simulation_*.json"),
                            key=lambda p: int(p.stem.split("_")[-1]))

        if not json_files:
            print(f"  [WARN] No group_simulation_*.json files found in {subfolder}")
            continue

        for src_path in json_files:
            # Derive the original number from the filename
            try:
                original_number = int(src_path.stem.split("_")[-1])
            except ValueError:
                print(f"  [SKIP] Cannot parse number from: {src_path.name}")
                total_skipped += 1
                continue

            new_id = id_offset + original_number
            new_filename = f"group_simulation_{new_id}.json"
            dst_path = out_dir / new_filename

            # Load, update group_id, save
            with open(src_path, encoding="utf-8") as f:
                data = json.load(f)

            data["group_id"] = new_id

            with open(dst_path, "w", encoding="utf-8") as f:
                json.dump(data, f)

            total_written += 1

        print(f"  Written {len(json_files)} files.")

    print(f"\nDone. {total_written} files written to '{OUTPUT_FOLDER}/'.")
    if total_skipped:
        print(f"       {total_skipped} files skipped (see warnings above).")
    print(f"\nTo run the evaluation framework on the merged dataset:")
    print(f"  python src/module1/evaluation_framework.py data/full_dataset/ data/results/")


if __name__ == "__main__":
    merge()
