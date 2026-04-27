"""
[业务逻辑层 - 视觉识别厨师]
负责处理手势追踪相关的逻辑。
1. 使用新版 MediaPipe Tasks API (HandLandmarker) 进行检测。
2. 支持 VIDEO 模式以获得更平滑的追踪和运动预测。
3. 支持单手/双手模式切换。
"""
import cv2
import mediapipe as mp
import math
import time
from collections import deque
from app.config import (
    HAND_CONFIDENCE, TRACK_CONFIDENCE, PRESENCE_CONFIDENCE, TASK_PATH
)

class FingerSmoother:
    """手指坐标平滑处理 - EWMA + 自适应速度调整"""
    def __init__(self, method='ewma', alpha=0.5, buffer_size=5, adaptive=True):
        self.method, self.alpha, self.buffer_size, self.adaptive = method, alpha, buffer_size, adaptive
        self.smoothed_pos, self.prev_raw_pos = None, None
        self.position_buffer = deque(maxlen=buffer_size)
        self.kalman_x, self.kalman_y = None, None
        if method == 'kalman': self._init_kalman()
    
    def _init_kalman(self):
        self.kalman_x = {'x': 0, 'v': 0, 'P': [[1, 0], [0, 1]], 'Q': 0.001, 'R': 0.1}
        self.kalman_y = {'x': 0, 'v': 0, 'P': [[1, 0], [0, 1]], 'Q': 0.001, 'R': 0.1}
    
    def _kalman_update(self, k, m):
        k['x'] += k['v']
        k['P'][0][0] += k['P'][1][1] + k['Q']
        k['P'][1][1] += k['Q']
        S = k['P'][0][0] + k['R']
        Kp, Kv = k['P'][0][0] / S, k['P'][1][1] / S
        e = m - k['x']
        k['x'] += Kp * e; k['v'] += Kv * e
        k['P'][0][0] = (1 - Kp) * k['P'][0][0]
        k['P'][1][1] = (1 - Kv) * k['P'][1][1]
        return k['x']

    def smooth(self, x, y):
        if self.prev_raw_pos is None:
            self.prev_raw_pos = (x, y)
            speed = 0
        else:
            px, py = self.prev_raw_pos
            speed = math.hypot(x - px, y - py)
            self.prev_raw_pos = (x, y)

        if self.method == 'ewma':
            if self.smoothed_pos is None:
                self.smoothed_pos = (x, y)
                return x, y
            a = self._get_alpha(speed) if self.adaptive and speed > 0 else self.alpha
            sx = a * x + (1 - a) * self.smoothed_pos[0]
            sy = a * y + (1 - a) * self.smoothed_pos[1]
            self.smoothed_pos = (sx, sy)
            return int(sx), int(sy)
        return x, y

    def _get_alpha(self, speed):
        if speed < 3: return 0.25
        if speed > 20: return 1.0
        return 0.25 + (speed - 3) / 17 * 0.75

    def reset(self):
        self.smoothed_pos, self.prev_raw_pos = None, None
        self.position_buffer.clear()

class HandTracker:
    def __init__(self, num_hands=1):
        self.num_hands = num_hands
        self._init_landmarker(num_hands)
        
    def _init_landmarker(self, num_hands):
        from mediapipe.tasks.python import vision
        from mediapipe.tasks import python
        
        base_options = python.BaseOptions(model_asset_path=TASK_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=HAND_CONFIDENCE,
            min_hand_presence_confidence=PRESENCE_CONFIDENCE,
            min_tracking_confidence=TRACK_CONFIDENCE,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.start_ns = time.perf_counter_ns()

    def _now_ms(self):
        return int((time.perf_counter_ns() - self.start_ns) // 1_000_000)

    def process(self, frame_rgb):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.landmarker.detect_for_video(mp_image, self._now_ms())
        
        if not result.hand_landmarks:
            return [], []
        
        # 返回所有检测到的手部关键点和左右手标签
        labels = [h[0].category_name for h in result.handedness]
        return result.hand_landmarks, labels

    def reinit(self, num_hands):
        """重新初始化（切换单/双手模式）"""
        self.close()
        self.num_hands = num_hands
        self._init_landmarker(num_hands)

    def close(self):
        if hasattr(self, 'landmarker'):
            self.landmarker.close()
