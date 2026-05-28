import argparse
import os
from pathlib import Path

import cv2
import torch

from drowsiness_detection import CONFIG, DrowsinessDetector, run_webcam_inference


def find_working_camera(max_id: int = 4):
    backends = [
        ("DEFAULT", None),
        ("CAP_DSHOW", getattr(cv2, "CAP_DSHOW", None)),
        ("CAP_MSMF", getattr(cv2, "CAP_MSMF", None)),
        ("CAP_ANY", getattr(cv2, "CAP_ANY", None)),
    ]

    for cid in range(max_id):
        for name, backend in backends:
            if backend is None and name != "DEFAULT":
                continue
            try:
                cap = cv2.VideoCapture(cid) if backend is None else cv2.VideoCapture(cid, backend)
                ok = cap.isOpened()
                cap.release()
                print(f"probe: id={cid} backend={name} opened={ok}")
                if ok:
                    return cid
            except Exception as exc:
                print(f"probe: id={cid} backend={name} error={exc}")
    return None


def parse_args():
    parser = argparse.ArgumentParser(description="Run live drowsiness detection on webcam")
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device to run inference on (default: auto)",
    )
    return parser.parse_args()


def resolve_device(device_choice: str) -> torch.device:
    if device_choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_choice == "cuda":
        if not torch.cuda.is_available():
            print("[TestRun] CUDA requested but not available. Falling back to CPU.")
            return torch.device("cpu")
    return torch.device(device_choice)


def load_model(model_path: Path, device: torch.device):
    model = DrowsinessDetector().to(device)
    if model_path.is_file():
        state = torch.load(model_path, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        if not isinstance(state, dict):
            print(f"[TestRun] Unsupported checkpoint format: {type(state).__name__}")
            print("[TestRun] Running with uninitialized weights.")
            return model

        model_state_keys = set(model.state_dict().keys())
        checkpoint_keys = set(state.keys())
        shared_keys = model_state_keys.intersection(checkpoint_keys)

        print(
            f"[TestRun] Checkpoint keys: {len(checkpoint_keys)}, "
            f"model keys: {len(model_state_keys)}, "
            f"common keys: {len(shared_keys)}"
        )

        if len(shared_keys) == 0:
            print("[TestRun] Checkpoint appears incompatible with the current model architecture.")
            print("[TestRun] Please retrain using the current code or use a matching checkpoint file.")
            print("[TestRun] Running with uninitialized weights.")
            return model

        try:
            load_result = model.load_state_dict(state, strict=False)
            print(f"[TestRun] Loaded model weights from: {model_path}")
            if load_result.missing_keys or load_result.unexpected_keys:
                print(
                    f"[TestRun] Warning: missing {len(load_result.missing_keys)} keys, "
                    f"unexpected {len(load_result.unexpected_keys)} keys."
                )
                if load_result.missing_keys:
                    print("[TestRun] Some model weights were not found in the checkpoint.")
                if load_result.unexpected_keys:
                    print("[TestRun] The checkpoint contains weights for a different model architecture.")
        except Exception as exc:
            print(f"[TestRun] Failed to load weights from {model_path}: {exc}")
            print("[TestRun] Running with uninitialized weights.")
    else:
        print(f"[TestRun] No checkpoint found at {model_path}. Running with uninitialized weights.")
    return model


def main():
    args = parse_args()
    device = resolve_device(args.device)
    print(f"[TestRun] Using device: {device}")

    model = load_model(Path(CONFIG["output_dir"]) / "best_model.pt", device)

    camera_id = find_working_camera(4)
    if camera_id is None:
        print("[TestRun] No working camera found on ids 0-3. Please verify camera access and try again.")
        return

    print(f"[TestRun] Using camera id={camera_id}")
    print("[TestRun] Press 'q' in the webcam window to quit.")
    run_webcam_inference(model, device, camera_id=camera_id)


if __name__ == "__main__":
    main()
