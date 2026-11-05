"""
Process Videos into Sign Language Dataset / แปลงไฟล์วิดีโอเป็นชุดข้อมูลภาษามือ (.npy)
Reads video files (.mp4, .mov, .avi, etc.), extracts 112 MediaPipe landmarks per frame,
and saves them to data/raw/<gesture>/<person>_<timestamp>.npy

Usage:
    # 1. Process all videos organized in videos/<gesture_name>/*.mp4
    python src/process_videos.py

    # 2. Process with visual preview window
    python src/process_videos.py --preview

    # 3. Process video from rear camera (do not flip)
    python src/process_videos.py --no-flip

    # 4. Process a single video file
    python src/process_videos.py --video path/to/clip.mp4 --word sawasdee
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from features import extract_frame_features
from landmarkers import Tracker, hands_by_side, draw_hand, draw_pose

SUPPORTED_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def process_single_video(video_path, word, person, tracker, flip=True, preview=False, min_frames=10):
    """
    Reads a single video file, extracts features per frame, and saves to data/raw/<word>/.
    Returns (success: bool, num_frames: int, message: str)
    """
    video_path = Path(video_path)
    if not video_path.exists():
        return False, 0, f"File not found: {video_path}"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False, 0, f"Cannot open video: {video_path}"

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames_features = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        if flip:
            frame = cv2.flip(frame, 1)

        hand_res, pose_res, _ = tracker.process(frame)
        hands = hands_by_side(hand_res)
        pose_lms = pose_res.pose_landmarks[0] if pose_res.pose_landmarks else None

        # Extract 112 features
        vec = extract_frame_features(pose_lms, hands)
        frames_features.append(vec)

        if preview:
            view = frame.copy()
            if pose_lms is not None:
                draw_pose(view, pose_lms)
            for side, lms in hands.items():
                if lms is not None:
                    draw_hand(view, lms, side)

            # Draw status on preview
            h, w = view.shape[:2]
            cv2.rectangle(view, (0, 0), (w, 50), (20, 20, 20), -1)
            cv2.putText(view, f"File: {video_path.name} | Frame {frame_idx}/{total_frames} ({fps:.1f} FPS)",
                        (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow("Video Extraction Preview (Press 'q' to skip preview)", view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                preview = False
                cv2.destroyAllWindows()

    cap.release()

    if len(frames_features) < min_frames:
        return False, len(frames_features), f"Too few frames ({len(frames_features)} < {min_frames})"

    # Save to data/raw/<word>/
    save_dir = config.DATA_RAW_DIR / word
    save_dir.mkdir(parents=True, exist_ok=True)

    ts = int(time.time() * 1000)
    out_file = save_dir / f"{person}_{ts}_{video_path.stem}.npy"
    np.save(str(out_file), np.array(frames_features, dtype=np.float32))

    return True, len(frames_features), str(out_file.name)


def main():
    parser = argparse.ArgumentParser(description="Process video files into sign language dataset (.npy)")
    parser.add_argument("--input-dir", type=str, default="videos",
                        help="Folder containing gesture subfolders with videos (default: videos)")
    parser.add_argument("--video", type=str, default=None,
                        help="Path to a single video file to process")
    parser.add_argument("--word", type=str, default=None,
                        help="Gesture name when processing a single video file")
    parser.add_argument("--person", type=str, default=config.DEFAULT_PERSON,
                        help="Signer/person ID (default: p1)")
    parser.add_argument("--flip", action="store_true", default=True,
                        help="Mirror flip video horizontally (default: True)")
    parser.add_argument("--no-flip", dest="flip", action="store_false",
                        help="Do not flip video (use if already mirrored or filmed with rear camera)")
    parser.add_argument("--preview", action="store_true", default=False,
                        help="Show visual preview during extraction")
    parser.add_argument("--min-frames", type=int, default=10,
                        help="Minimum number of frames required to save a sample (default: 10)")
    args = parser.parse_args()

    print("=" * 60)
    print(" Video Dataset Extractor / ระบบสกัดข้อมูลภาษามือจากวิดีโอ ")
    print("=" * 60)

    with Tracker() as tracker:
        if args.video:
            # Process single video
            video_path = Path(args.video)
            if not video_path.exists():
                print(f"[Error] Video file not found: {video_path}")
                sys.exit(1)
            word = args.word or video_path.parent.name
            if not word or word == ".":
                word = "gesture"

            print(f"Processing single video: {video_path.name} -> Class: [{word}]")
            ok, n_frames, msg = process_single_video(
                video_path, word, args.person, tracker, flip=args.flip,
                preview=args.preview, min_frames=args.min_frames
            )
            if ok:
                print(f"[Success] Saved {msg} ({n_frames} frames)")
            else:
                print(f"[Failed] {msg}")

        else:
            # Scan input directory for gesture subfolders
            input_dir = Path(args.input_dir)
            if not input_dir.exists():
                input_dir.mkdir(parents=True, exist_ok=True)
                print(f"[Created] Created '{input_dir}' directory.")
                print("\nPlease place your gesture videos inside:")
                print(f"  {input_dir}/<gesture_name>/sample1.mp4")
                print(f"  {input_dir}/<gesture_name>/sample2.mp4")
                print("Then run this script again.")
                sys.exit(0)

            # Find subfolders
            gesture_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
            if not gesture_dirs:
                print(f"[Warning] No gesture subfolders found inside '{input_dir}'.")
                print("\nExpected directory layout:")
                print(f"  {input_dir}/sawasdee/01.mp4")
                print(f"  {input_dir}/sawasdee/02.mp4")
                print(f"  {input_dir}/khopkhun/01.mp4")
                sys.exit(0)

            total_success = 0
            total_failed = 0
            stats_by_word = {}

            print(f"Found {len(gesture_dirs)} gesture folder(s): {[d.name for d in gesture_dirs]}")
            print(f"Flip horizontal: {args.flip} | Person ID: {args.person}\n")

            for g_dir in gesture_dirs:
                word = g_dir.name
                video_files = [f for f in g_dir.iterdir() if f.suffix.lower() in SUPPORTED_EXTS]
                print(f"--- Gesture: [{word}] ({len(video_files)} video files) ---")

                stats_by_word[word] = 0

                for v_file in video_files:
                    ok, n_frames, msg = process_single_video(
                        v_file, word, args.person, tracker, flip=args.flip,
                        preview=args.preview, min_frames=args.min_frames
                    )
                    if ok:
                        print(f"  [OK] {v_file.name} -> {msg} ({n_frames} frames)")
                        total_success += 1
                        stats_by_word[word] += 1
                    else:
                        print(f"  [SKIP] {v_file.name} -> {msg}")
                        total_failed += 1

            if args.preview:
                cv2.destroyAllWindows()

            print("\n" + "=" * 60)
            print(" Extraction Complete Summary ")
            print("=" * 60)
            print(f"Total Successfully Extracted: {total_success} samples")
            print(f"Total Failed/Skipped: {total_failed} samples")
            print("Samples per gesture class in data/raw/:")
            for word, cnt in stats_by_word.items():
                raw_cnt = len(list((config.DATA_RAW_DIR / word).glob("*.npy"))) if (config.DATA_RAW_DIR / word).exists() else 0
                print(f"  - {word}: {raw_cnt} total samples (+{cnt} newly extracted)")

            print("\nNext step: Train the model using:")
            print("  python src/train_sklearn.py")
            print("  python src/train_torch.py")


if __name__ == "__main__":
    main()
