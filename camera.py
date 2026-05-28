device = "cuda" if torch.cuda.is_available() else "cpu"
# from torch import device
import torch

from drowsiness_detection import DrowsinessDetector, run_webcam_inference

device = "cuda" if torch.cuda.is_available() else "cpu"

model = DrowsinessDetector().to(device)

print("\n" + "=" * 60)
print("  STEP 7 - Real-time Webcam Inference")
print("=" * 60)
run_webcam_inference(model, device)