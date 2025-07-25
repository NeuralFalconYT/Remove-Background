import cv2
import numpy as np
from soft_foreground_segmenter import SoftForegroundSegmenter

# Load model
foreground_model = "foreground-segmentation-model-vitb16_384.onnx"
segmenter = SoftForegroundSegmenter(onnx_model=foreground_model)

def apply_green_screen_to_frame(frame, segmenter):
    """Apply green screen effect to a single video frame."""
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mask = segmenter.estimate_foreground_segmentation(image_rgb)

    if mask.max() <= 1.0:
        mask = (mask * 255).astype(np.uint8)
    else:
        mask = mask.astype(np.uint8)

    if mask.ndim == 3 and mask.shape[2] == 3:
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask

    _, binary_mask = cv2.threshold(mask_gray, 128, 255, cv2.THRESH_BINARY)

    # Create green background
    green_bg = np.full_like(frame, (0, 255, 0), dtype=np.uint8)
    mask_3ch = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)

    # Composite
    output_frame = np.where(mask_3ch == 255, frame, green_bg)
    return output_frame

# Start webcam
cap = cv2.VideoCapture(0)  # Use 0 for default webcam

if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

print("🎥 Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Can't receive frame (stream end?). Exiting ...")
        break

    output = apply_green_screen_to_frame(frame, segmenter)

    cv2.imshow("Real-Time Green Screen", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
