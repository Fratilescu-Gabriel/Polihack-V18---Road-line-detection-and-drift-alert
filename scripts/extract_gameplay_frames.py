
from pathlib import Path
import cv2
import csv


PROJECT_ROOT = Path(__file__).resolve().parents[1]

VIDEO_DIR = PROJECT_ROOT / "data" / "raw" / "lane-detection" / "ets2personal" / "gameplay_overtake"
OUT_ROOT = PROJECT_ROOT /  "data" / "processed" / "gameplay_overtake"


SAMPLE_EVERY_N_FRAMES = 1


def extract_frames_from_video(video_path: Path):
    run_id = video_path.stem
    out_dir = OUT_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # fallback

    print(f"[INFO] Processing {video_path.name} (fps ~ {fps:.1f})")

    frame_idx = 0
    saved_idx = 0
    rows = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # sample frames
        if frame_idx % SAMPLE_EVERY_N_FRAMES != 0:
            frame_idx += 1
            continue

        # keep the original resolution; lane detection ROI will handle cropping
        out_name = f"{run_id}_f{frame_idx:06d}.jpg"
        out_path = out_dir / out_name
        cv2.imwrite(str(out_path), frame)

        time_sec = frame_idx / fps

        rows.append({
            "run_id": run_id,
            "frame_idx": frame_idx,
            "time_sec": time_sec,
            "image_path": str(out_path.relative_to(PROJECT_ROOT)),
        })

        saved_idx += 1
        frame_idx += 1

    cap.release()
    print(f"[INFO] {video_path.name}: saved {saved_idx} frames to {out_dir}")
    return rows


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    if not VIDEO_DIR.exists():
        print(f"[ERROR] VIDEO_DIR does not exist: {VIDEO_DIR}")
        return

    video_files = [p for p in VIDEO_DIR.iterdir() if p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}]
    if not video_files:
        print(f"[ERROR] No video files found in {VIDEO_DIR}")
        return

    all_rows = []
    for video_path in video_files:
        rows = extract_frames_from_video(video_path)
        all_rows.extend(rows)

    # write a small meta CSV (optional but handy later)
    if all_rows:
        meta_path = OUT_ROOT / "user_runs_meta.csv"
        fieldnames = ["run_id", "frame_idx", "time_sec", "image_path"]
        with meta_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in all_rows:
                writer.writerow(r)
        print(f"[INFO] Wrote metadata to {meta_path}")
    else:
        print("[WARN] No frames saved at all.")


if __name__ == "__main__":
    main()
