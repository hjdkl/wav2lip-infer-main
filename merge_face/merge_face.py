from typing import Any
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull
import gc


class MergeFace:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.face_mesh.close()
        self.face_mesh = None
        self.mp_face_mesh = None
        self.mp_drawing = None
        self.drawing_spec = None
        gc.collect()
        return

    def get_face_small(self, img: np.ndarray) -> Any:
        # 检测初始的人脸
        results = self.face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            print("No face detected")
            return None, None
        cors = []
        face_landmark = results.multi_face_landmarks[0]
        for landmark in face_landmark.landmark:
            x, y = landmark.x, landmark.y
            shape = img.shape
            relative_x = int(x * shape[1])
            relative_y = int(y * shape[0])
            cors.append((relative_x, relative_y))
        # cv2转PIL，不使用原有的对点置空的方法，而是使用spicy的ConvexHull，速度快非常多，并且不需要遍历
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img_pil = img_pil.convert("RGBA")
        width, height = img_pil.size
        img_np = np.array(img_pil)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGRA)
        # 全黑mask
        mask = np.zeros((height, width, 3), dtype=np.uint8)
        # 在mask上hull內的区域填充白色
        hull = ConvexHull(cors)
        hull_np = np.array([cors[i] for i in hull.vertices])
        cv2.fillPoly(mask, [hull_np], (255, 255, 255))
        # 缩小mask到0.9倍
        mask = cv2.resize(mask, (int(width * 0.9), int(height * 0.9)))
        final_mask = np.zeros((height, width, 3), dtype=np.uint8)
        top = int(height * 0.05)
        left = int(width * 0.05)
        bottom = top + mask.shape[0]
        right = left + mask.shape[1]
        final_mask[top:bottom, left:right] = mask
        half_height = int(height * 0.6)
        final_mask[:half_height] = 0
        final_mask = cv2.cvtColor(final_mask, cv2.COLOR_BGR2GRAY)
        return img_np, final_mask
