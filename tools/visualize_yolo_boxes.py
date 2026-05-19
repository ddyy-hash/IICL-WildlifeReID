import argparse
import os
import sys

import cv2
import numpy as np
import torch


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from tools.train_joint import YOLODetectorWrapper  # noqa: E402


def visualize(image_path: str, output_path: str, model_path: str, conf: float = 0.5) -> None:
    """Run YOLO on one image and save the detected boxes."""
    if not os.path.exists(image_path):
        print(f"Input image does not exist: {image_path}")
        return

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Unable to read image: {image_path}")
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = img_rgb.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0)  # (1,3,H,W)

    detector = YOLODetectorWrapper(model_path=model_path, conf=conf)
    boxes_list = detector.detect_batch(img_tensor)
    boxes = boxes_list[0]

    if boxes is None or boxes.numel() == 0:
        print("YOLO returned no boxes; the wrapper may have fallen back to a full-image box.")
    else:
        boxes_np = boxes.cpu().numpy()
        for box in boxes_np:
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, img_bgr)
    print(f"Detection result saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize YOLO detections on the current image")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--output", default="yolo_vis.jpg", help="Output visualization path")
    parser.add_argument(
        "--model",
        default="./fea_data/yolov8m-seg.pt",
        help="YOLO model path; keep it consistent with training",
    )
    parser.add_argument("--conf", type=float, default=0.5, help="YOLO confidence threshold")

    args = parser.parse_args()
    visualize(args.image, args.output, args.model, args.conf)


if __name__ == "__main__":
    main()
