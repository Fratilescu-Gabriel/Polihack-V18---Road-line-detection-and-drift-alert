
from pathlib import Path
import cv2
import random


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_IMG_DIR = PROJECT_ROOT / "data" / "raw" / "lane-detection" / "ets2personal" / "gameplay_overtake"

OUT_DIR = PROJECT_ROOT / "data" / "processed" / "gameplay_overtake"

N_SAMPLES = 100000
TARGET_SIZE = (640, 360)


def crop_and_resize(img):
    h, w, _ = img.shape
    top = int(0.25 * h)
    cropped = img[top:, :]
    resized = cv2.resize(cropped, TARGET_SIZE)
    return resized


def main():
    print(f"[INFO] Project root: {PROJECT_ROOT}")
    print(f"[INFO] RAW_IMG_DIR:  {RAW_IMG_DIR}")

    if not RAW_IMG_DIR.exists():
        print("[ERROR] RAW_IMG_DIR does not exist. Check your folder structure or adjust RAW_IMG_DIR.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png"}
    img_paths = []

    counter = 0
    for p in RAW_IMG_DIR.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            img_paths.append(p)
            if len(img_paths) >= N_SAMPLES:
                break


    print(f"[INFO] Found {len(img_paths)} image files to process (requested {N_SAMPLES}).")

    if not img_paths:
        print("[ERROR] No image files found. Check RAW_IMG_DIR or extensions.")
        return

    count = 0
    for i, img_path in enumerate(img_paths):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] Could not read {img_path}")
            continue

        processed = crop_and_resize(img)
        out_name = f"sample_{count:04d}.jpg"
        out_path = OUT_DIR / out_name
        cv2.imwrite(str(out_path), processed)
        count += 1

    print(f"[INFO] Done. Created {count} sample images in: {OUT_DIR}")


if __name__ == "__main__":
    main()
