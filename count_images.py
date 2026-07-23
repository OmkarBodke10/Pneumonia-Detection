import os

DATA_DIR = "data/chest_xray"

for split in ["train", "val", "test"]:
    print(f"\n{split.upper()}")

    total = 0

    for cls in ["NORMAL", "PNEUMONIA"]:
        folder = os.path.join(DATA_DIR, split, cls)

        count = len([
            f for f in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, f))
        ])

        total += count
        print(f"{cls:<12}: {count}")

    print(f"Total Images: {total}")