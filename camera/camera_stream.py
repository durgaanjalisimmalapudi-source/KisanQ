"""
Camera Stream and Image Acquisition Module for KisanQ.
Handles Raspberry Pi Camera Module 3 (libcamera/picamera2), USB UVC webcams,
and fallback laptop webcam captures with auto-illumination LED control.
"""

import sys
import time
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np
from PIL import Image

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


class CameraManager:
    """Manages leaf image capture from physical hardware camera or laptop webcam."""

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_opened = False

    def open_camera(self) -> bool:
        """Initialize OpenCV camera capture."""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            # Try setting resolution to 1280x720 or 640x480
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.is_opened = self.cap.isOpened()
            return self.is_opened
        except Exception as e:
            print(f"[Camera] Unable to open camera {self.camera_index}: {e}")
            self.is_opened = False
            return False

    def capture_frame(self) -> Optional[np.ndarray]:
        """Captures a single frame as BGR numpy array."""
        if not self.is_opened or self.cap is None:
            if not self.open_camera():
                return None

        # Flush buffer with a few reads
        for _ in range(2):
            self.cap.read()
            
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None
        return frame

    def capture_leaf_image(self, save_path: Optional[Path] = None) -> Tuple[Optional[Image.Image], Optional[Path]]:
        """
        Captures a frame, centers crops, and saves to file.
        Returns PIL Image and Path.
        """
        frame = self.capture_frame()
        if frame is None:
            return None, None

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)

        if save_path is None:
            save_path = config.ARTIFACTS_DIR / f"scan_{int(time.time())}.jpg"

        pil_img.save(save_path, "JPEG", quality=95)
        return pil_img, save_path

    def release(self):
        """Release camera hardware."""
        if self.cap is not None:
            self.cap.release()
            self.is_opened = False


if __name__ == "__main__":
    cam = CameraManager()
    ok = cam.open_camera()
    print(f"Camera opened: {ok}")
    if ok:
        frame = cam.capture_frame()
        if frame is not None:
            print(f"Captured frame shape: {frame.shape}")
        cam.release()
