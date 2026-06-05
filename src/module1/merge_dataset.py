"""Merge the four per-model raw subfolders (data/raw/{gemma,llama,mistral,olmo},
2,000 files each) into data/full_dataset/ with sequential group_ids 1-8000. Only
group_id is changed; the originals are left untouched."""

import json
import shutil
from pathlib import Path

# Paths are relative to this script (src/module1/). Order sets the ID ranges:
# first folder -> IDs 1-2000, second -> 2001-4000, and so on.
SUBFOLDERS = [
    "../../data/raw/gemma",
    "../../data/raw/llama",
    "../../data/raw/mistral",
    "../../data/raw/olmo",
]

OUTPUT_FOLDER = "../../data/full_dataset"

FILES_PER_SUBFOLDER = 2000 


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
