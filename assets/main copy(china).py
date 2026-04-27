import os
import warnings

# 抑制 TensorFlow/oneDNN 的冗余日志
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import mediapipe as mp
import numpy as np
import random
import math
from collections import deque

# PIL 中文渲染支持
from PIL import Image, ImageDraw, ImageFont

# 消除 protobuf 弃用警告
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf.symbol_database')

try:
    import pygame
    pygame.mixer.init()
    HAS_SOUND = True
except ImportError:
    HAS_SOUND = False
    print("提示: 安装pygame可启用音效 (pip install pygame)")


# ============================================================
#  中文字体系统（批量渲染，每帧只做一次 BGR↔RGB 转换）
# ============================================================
_font_cache: dict = {}

def get_cn_font(size: int) -> ImageFont.ImageFont:
    """获取（并缓存）中文字体"""
    if size in _font_cache:
        return _font_cache[size]
    candidates = [
        # Windows
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/msyhbd.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        # Linux (常见发行版)
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        # macOS
        "/System/Library/Fonts/PingFang.ttc",
        "/Library/Fonts/Arial Unicode MS.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, size)
                _font_cache[size] = font
                return font
            except Exception:
                continue
    # 备用默认字体（无法显示中文，但不会崩溃）
    font = ImageFont.load_default()
    _font_cache[size] = font
    return font


# 每帧文字队列（避免多次 PIL 转换）
_text_queue: list = []

def add_cn_text(text: str, pos: tuple, font_size: int = 32,
                color=(255, 255, 255), bg_color=None, padding: int = 8):
    """将一条中文文字加入本帧渲染队列"""
    _text_queue.append((text, pos, font_size, color, bg_color, padding))


def flush_cn_texts(frame: np.ndarray):
    """将队列中所有文字批量渲染到 frame（仅一次 BGR↔RGB 转换）"""
    global _text_queue
    if not _text_queue:
        return
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    for text, (x, y), font_size, color, bg_color, padding in _text_queue:
        font = get_cn_font(font_size)
        if bg_color is not None:
            bbox = draw.textbbox((x, y), text, font=font)
            draw.rectangle(
                [bbox[0] - padding, bbox[1] - padding,
                 bbox[2] + padding, bbox[3] + padding],
                fill=(bg_color[2], bg_color[1], bg_color[0])  # BGR→RGB
            )
        draw.text((x, y), text, font=font,
                  fill=(color[2], color[1], color[0]))        # BGR→RGB
    _text_queue = []
    np.copyto(frame, cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR))


# ============================================================
#  坐标平滑处理类
# ============================================================
class FingerSmoother:
    """手指坐标平滑处理 - EWMA + 自适应速度调整"""

    def __init__(self, method='ewma', alpha=0.5, buffer_size=5, adaptive=True):
        self.method = method
        self.alpha = alpha
        self.buffer_size = buffer_size
        self.adaptive = adaptive
        self.smoothed_pos = None
        self.prev_raw_pos = None
        self.position_buffer = deque(maxlen=buffer_size)
        self.kalman_x = None
        self.kalman_y = None
        if method == 'kalman':
            self._init_kalman()

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
        k['x'] += Kp * e
        k['v'] += Kv * e
        k['P'][0][0] = (1 - Kp) * k['P'][0][0]
        k['P'][1][1] = (1 - Kv) * k['P'][1][1]
        return k['x']

    def _speed(self, x, y):
        if self.prev_raw_pos is None:
            self.prev_raw_pos = (x, y)
            return 0
        px, py = self.prev_raw_pos
        s = math.hypot(x - px, y - py)
        self.prev_raw_pos = (x, y)
        return s

    def _adaptive_alpha(self, speed):
        if speed < 3:
            return 0.25
        if speed > 20:
            return 1.0
        return 0.25 + (speed - 3) / 17 * 0.75

    def smooth(self, x, y):
        speed = self._speed(x, y) if self.adaptive else 0
        if self.method == 'ewma':
            if self.smoothed_pos is None:
                self.smoothed_pos = (x, y)
                return x, y
            a = self._adaptive_alpha(speed) if self.adaptive and speed > 0 else self.alpha
            sx = a * x + (1 - a) * self.smoothed_pos[0]
            sy = a * y + (1 - a) * self.smoothed_pos[1]
            self.smoothed_pos = (sx, sy)
            return int(sx), int(sy)
        elif self.method == 'moving_avg':
            self.position_buffer.append((x, y))
            pts = list(self.position_buffer)[-2:] if (self.adaptive and speed > 20) else list(self.position_buffer)
            return int(sum(p[0] for p in pts) / len(pts)), int(sum(p[1] for p in pts) / len(pts))
        elif self.method == 'kalman':
            if self.kalman_x is None:
                self._init_kalman()
                self.kalman_x['x'], self.kalman_y['x'] = x, y
                return x, y
            return int(self._kalman_update(self.kalman_x, x)), int(self._kalman_update(self.kalman_y, y))
        return x, y

    def reset(self):
        self.smoothed_pos = None
        self.prev_raw_pos = None
        self.position_buffer.clear()
        if self.method == 'kalman':
            self._init_kalman()


# ============================================================
#  MediaPipe 手部检测（支持双手）
# ============================================================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,          # 双手模式需要检测2只手
    model_complexity=0,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.7
)

# ============================================================
#  窗口尺寸
# ============================================================
WINDOW_WIDTH  = 1280
WINDOW_HEIGHT = 720

# ============================================================
#  素材加载
# ============================================================
FRUIT_TYPES = [
    'banana', 'boluo', 'iceBanana', 'Mango',
    'mugua', 'peach', 'pear', 'pineapple', 'strawberry', 'b1'
]
MULTI_FRUIT_TYPES = ['watermelon', 'dragonfruit']


def load_fruit_images():
    fruit_images = {}
    script_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.join(script_dir, 'sucai')
    print(f"\n正在加载水果素材... ({assets_dir})")
    for name in FRUIT_TYPES:
        whole_path = os.path.join(assets_dir, f'{name}.png')
        if name == 'b1':
            left_path  = os.path.join(assets_dir, 'bl.png')
            right_path = os.path.join(assets_dir, 'br.png')
        else:
            left_path  = os.path.join(assets_dir, f'{name}l.png')
            right_path = os.path.join(assets_dir, f'{name}r.png')
        if os.path.exists(whole_path):
            wi = cv2.imread(whole_path, cv2.IMREAD_UNCHANGED)
            li = cv2.imread(left_path,  cv2.IMREAD_UNCHANGED)
            ri = cv2.imread(right_path, cv2.IMREAD_UNCHANGED)
            fruit_images[name] = {'whole': wi, 'left': li, 'right': ri}
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name}")
    return fruit_images


def load_multi_fruit_images():
    images = {}
    script_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.join(script_dir, 'sucai')
    print("\n正在加载多部分水果素材...")
    sc = 0.5
    # 西瓜
    wp = os.path.join(assets_dir, 'watermelon.png')
    if os.path.exists(wp):
        wi = cv2.imread(wp, cv2.IMREAD_UNCHANGED)
        if wi is not None:
            wi = cv2.resize(wi, None, fx=sc, fy=sc)
            pieces = []
            for i in range(1, 9):
                pp = os.path.join(assets_dir, f'watermelon{i}.png')
                if os.path.exists(pp):
                    pi = cv2.imread(pp, cv2.IMREAD_UNCHANGED)
                    if pi is not None:
                        pieces.append(cv2.resize(pi, None, fx=sc, fy=sc))
            if len(pieces) == 8:
                images['watermelon'] = {'whole': wi, 'pieces': pieces, 'piece_count': 8}
                print(f"  ✓ 西瓜 (8片)")
    # 火龙果
    dp = os.path.join(assets_dir, 'all.png')
    if os.path.exists(dp):
        wi = cv2.imread(dp, cv2.IMREAD_UNCHANGED)
        if wi is not None:
            wi = cv2.resize(wi, None, fx=sc, fy=sc)
            pieces = []
            for i in range(1, 9):
                pp = os.path.join(assets_dir, f'00{i}.png')
                if os.path.exists(pp):
                    pi = cv2.imread(pp, cv2.IMREAD_UNCHANGED)
                    if pi is not None:
                        pieces.append(cv2.resize(pi, None, fx=sc, fy=sc))
            if len(pieces) == 8:
                images['dragonfruit'] = {'whole': wi, 'pieces': pieces, 'piece_count': 8}
                print(f"  ✓ 火龙果 (8片)")
    return images


def load_bomb_images():
    imgs = {}
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bomb_dir = os.path.join(script_dir, 'zhadan')
    print(f"\n正在加载炸弹素材...")
    for key, fname, sc in [
        ('bomb1',      'boom1.png', 1.0),
        ('bomb2',      'boom2.png', 1.0),
        ('explosion1', 'zha01.png', 2.0),
        ('explosion2', 'zha02.png', 2.0),
    ]:
        p = os.path.join(bomb_dir, fname)
        if os.path.exists(p):
            img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            if img is not None:
                imgs[key] = cv2.resize(img, None, fx=sc, fy=sc) if sc != 1.0 else img
                print(f"  ✓ {key}")
    return imgs


def load_blade_images():
    imgs = {}
    script_dir = os.path.dirname(os.path.abspath(__file__))
    blade_dir = os.path.join(script_dir, 'daoguang')
    print(f"\n正在加载刀光素材...")
    for key, fname in [('dao1', 'dao1.png'), ('dao2', 'dao2.png')]:
        p = os.path.join(blade_dir, fname)
        if os.path.exists(p):
            img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            if img is not None:
                imgs[key] = img
                print(f"  ✓ {key}")
    return imgs


def load_combo_images():
    imgs = {}
    script_dir = os.path.dirname(os.path.abspath(__file__))
    combo_dir = os.path.join(script_dir, 'texiao')
    print(f"\n正在加载连击特效...")
    for i in range(1, 4):
        p = os.path.join(combo_dir, f'combo{i}.png')
        if os.path.exists(p):
            img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            if img is not None:
                imgs[f'combo{i}'] = img
                print(f"  ✓ combo{i}")
    return imgs


def load_juice_images():
    imgs = {}
    script_dir = os.path.dirname(os.path.abspath(__file__))
    texiao_dir = os.path.join(script_dir, 'texiao')
    print(f"\n正在加载汁水特效...")
    colors = ['orange', 'green', 'pink', 'red']
    for i, c in enumerate(colors, 1):
        p = os.path.join(texiao_dir, f'guozhi{i}.png')
        if os.path.exists(p):
            img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            if img is not None:
                imgs[c] = img
                print(f"  ✓ {c}汁水")
    return imgs


def load_sound_effects():
    sfx = {}
    if not HAS_SOUND:
        return sfx
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sound_dir = os.path.join(script_dir, 'yinxiao')
    print(f"\n正在加载音效...")
    try:
        sp = os.path.join(sound_dir, 'qieshuiguoyinxiao.mp3')
        if os.path.exists(sp):
            sfx['slice'] = pygame.mixer.Sound(sp)
            print("  ✓ 切水果音效")
        ep = os.path.join(sound_dir, 'baozhayinxiao.mp3')
        if os.path.exists(ep):
            sfx['explosion'] = pygame.mixer.Sound(ep)
            print("  ✓ 爆炸音效")
    except Exception as e:
        print(f"  ✗ 音效加载失败: {e}")
    return sfx


# 全局素材
FRUIT_IMAGES       = load_fruit_images()
MULTI_FRUIT_IMAGES = load_multi_fruit_images()
BOMB_IMAGES        = load_bomb_images()
BLADE_IMAGES       = load_blade_images()
COMBO_IMAGES       = load_combo_images()
JUICE_IMAGES       = load_juice_images()
SOUND_EFFECTS      = load_sound_effects()

FRUIT_JUICE_MAP = {
    'banana': 'orange', 'boluo': 'orange', 'iceBanana': 'green',
    'Mango': 'orange', 'mugua': 'orange', 'peach': 'pink',
    'pear': 'green', 'pineapple': 'orange', 'strawberry': 'red',
    'watermelon': 'red', 'dragonfruit': 'pink', 'b1': 'orange',
}

def get_juice_color(fruit_type):
    c = FRUIT_JUICE_MAP.get(fruit_type)
    if c and c in JUICE_IMAGES:
        return c
    return random.choice(list(JUICE_IMAGES.keys())) if JUICE_IMAGES else None


# ============================================================
#  图片叠加函数（带透明通道 + 旋转）
# ============================================================
def overlay_image(bg, overlay, x, y, rotation=0, alpha=1.0):
    if overlay is None or overlay.size == 0:
        return
    try:
        ov = overlay.copy()
        if rotation != 0:
            h, w = ov.shape[:2]
            cx, cy = w // 2, h // 2
            M = cv2.getRotationMatrix2D((cx, cy), rotation, 1.0)
            cos, sin = abs(M[0, 0]), abs(M[0, 1])
            nw = int(h * sin + w * cos)
            nh = int(h * cos + w * sin)
            M[0, 2] += nw / 2 - cx
            M[1, 2] += nh / 2 - cy
            ov = cv2.warpAffine(ov, M, (nw, nh),
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(0, 0, 0, 0))
        oh, ow = ov.shape[:2]
        x1, y1 = int(x - ow // 2), int(y - oh // 2)
        x2, y2 = x1 + ow, y1 + oh
        if x1 >= bg.shape[1] or y1 >= bg.shape[0] or x2 <= 0 or y2 <= 0:
            return
        ox1, oy1 = max(0, -x1), max(0, -y1)
        ox2 = ow - max(0, x2 - bg.shape[1])
        oy2 = oh - max(0, y2 - bg.shape[0])
        bx1 = max(0, x1); by1 = max(0, y1)
        bx2 = min(bg.shape[1], x2); by2 = min(bg.shape[0], y2)
        if ox2 <= ox1 or oy2 <= oy1:
            return
        roi = ov[oy1:oy2, ox1:ox2]
        if roi.size == 0:
            return
        if len(roi.shape) == 3 and roi.shape[2] == 4:
            a = roi[:, :, 3:] / 255.0 * alpha
            c = roi[:, :, :3]
        else:
            c = roi[:, :, :3] if len(roi.shape) == 3 else roi
            a = np.ones((roi.shape[0], roi.shape[1], 1)) * alpha
        bgreg = bg[by1:by2, bx1:bx2]
        if bgreg.shape[:2] != c.shape[:2]:
            return
        bg[by1:by2, bx1:bx2] = (a * c + (1 - a) * bgreg).astype(np.uint8)
    except Exception:
        pass


# ============================================================
#  游戏实体类
# ============================================================
class Fruit:
    def __init__(self):
        self.x = random.randint(100, WINDOW_WIDTH - 100)
        self.y = WINDOW_HEIGHT + 50
        self.fruit_type = random.choice(FRUIT_TYPES)
        self.images = FRUIT_IMAGES.get(self.fruit_type)
        if self.images and self.images['whole'] is not None:
            h, w = self.images['whole'].shape[:2]
            self.radius = max(w, h) // 2
        else:
            self.radius = 50
        self.vx = random.uniform(-1, 1)
        self.vy = random.uniform(-18, -14)
        self.gravity = 0.3
        self.rot = random.uniform(0, 360)
        self.rot_spd = random.uniform(-5, 5)
        self.is_cut = False
        self.cut_pieces = []
        self.entered = False

    def update(self):
        if not self.is_cut:
            self.vy += self.gravity
            self.x += self.vx
            self.y += self.vy
            self.rot += self.rot_spd
            if not self.entered and 0 <= self.y <= WINDOW_HEIGHT - 50:
                self.entered = True
        else:
            for p in self.cut_pieces:
                p['vy'] += self.gravity * 2.0
                p['x'] += p['vx']; p['y'] += p['vy']
                p['rot'] += p['rot_spd']
                p['alpha'] -= 3

    def draw(self, frame):
        if not self.is_cut:
            if self.images and self.images['whole'] is not None:
                overlay_image(frame, self.images['whole'], int(self.x), int(self.y), self.rot)
            else:
                cv2.circle(frame, (int(self.x), int(self.y)), 40, (0, 255, 255), -1)
        else:
            for p in self.cut_pieces:
                if p['alpha'] > 0:
                    overlay_image(frame, p['image'], int(p['x']), int(p['y']),
                                  p['rot'], p['alpha'] / 255.0)

    def cut(self, _=0):
        self.is_cut = True
        if not self.images:
            return
        for img, dx in [(self.images['left'], -1), (self.images['right'], 1)]:
            if img is not None:
                self.cut_pieces.append({
                    'x': self.x, 'y': self.y, 'image': img,
                    'vx': random.uniform(5, 8) * dx,
                    'vy': random.uniform(2, 4),
                    'rot': self.rot,
                    'rot_spd': random.uniform(4, 8) * dx,
                    'alpha': 255
                })

    def out(self):
        if not self.is_cut:
            return self.y > WINDOW_HEIGHT + 150
        return all(p['alpha'] <= 0 or p['y'] > WINDOW_HEIGHT + 150 for p in self.cut_pieces)

    def check_collision(self, trail):
        if self.is_cut or len(trail) < 2:
            return False
        for pt in list(trail)[-15:]:
            if pt and math.hypot(pt[0] - self.x, pt[1] - self.y) < self.radius * 0.8:
                return True
        return False


class MultiFruit:
    def __init__(self):
        self.x = random.randint(100, WINDOW_WIDTH - 100)
        self.y = WINDOW_HEIGHT + 50
        self.fruit_type = random.choice(MULTI_FRUIT_TYPES)
        self.images = MULTI_FRUIT_IMAGES.get(self.fruit_type)
        if self.images and self.images['whole'] is not None:
            h, w = self.images['whole'].shape[:2]
            self.radius = max(w, h) // 2
        else:
            self.radius = 50
        self.vx = random.uniform(-1, 1)
        self.vy = random.uniform(-18, -14)
        self.gravity = 0.3
        self.rot = random.uniform(0, 360)
        self.rot_spd = random.uniform(-5, 5)
        self.is_cut = False
        self.cut_pieces = []
        self.entered = False

    def update(self):
        if not self.is_cut:
            self.vy += self.gravity
            self.x += self.vx
            self.y += self.vy
            self.rot += self.rot_spd
            if not self.entered and 0 <= self.y <= WINDOW_HEIGHT - 50:
                self.entered = True
        else:
            for p in self.cut_pieces:
                p['vy'] += self.gravity * 2.0
                p['x'] += p['vx']; p['y'] += p['vy']
                p['rot'] += p['rot_spd']
                p['alpha'] -= 3

    def draw(self, frame):
        if not self.is_cut:
            if self.images and self.images['whole'] is not None:
                overlay_image(frame, self.images['whole'], int(self.x), int(self.y), self.rot)
            else:
                cv2.circle(frame, (int(self.x), int(self.y)), 40, (0, 200, 200), -1)
        else:
            for p in self.cut_pieces:
                if p['alpha'] > 0:
                    overlay_image(frame, p['image'], int(p['x']), int(p['y']),
                                  p['rot'], p['alpha'] / 255.0)

    def cut(self, angle=0):
        self.is_cut = True
        if not self.images or not self.images.get('pieces'):
            return
        cnt = self.images['piece_count']
        step = 360 / cnt
        for i, img in enumerate(self.images['pieces']):
            a = math.radians(i * step)
            spd = random.uniform(6, 10)
            self.cut_pieces.append({
                'x': self.x, 'y': self.y, 'image': img,
                'vx': math.cos(a) * spd,
                'vy': math.sin(a) * spd - 2,
                'rot': self.rot + random.uniform(-30, 30),
                'rot_spd': random.uniform(-10, 10),
                'alpha': 255
            })

    def out(self):
        if not self.is_cut:
            return self.y > WINDOW_HEIGHT + 150
        return all(p['alpha'] <= 0 or p['y'] > WINDOW_HEIGHT + 150 for p in self.cut_pieces)

    def check_collision(self, trail):
        if self.is_cut or len(trail) < 2:
            return False
        for pt in list(trail)[-15:]:
            if pt and math.hypot(pt[0] - self.x, pt[1] - self.y) < self.radius * 0.8:
                return True
        return False


class Bomb:
    def __init__(self, bomb_type='normal'):
        self.x = random.randint(100, WINDOW_WIDTH - 100)
        self.y = WINDOW_HEIGHT + 50
        self.bomb_type = bomb_type
        self.image = BOMB_IMAGES.get('bomb2' if bomb_type == 'deadly' else 'bomb1')
        if self.image is not None:
            h, w = self.image.shape[:2]
            self.radius = max(w, h) // 2
        else:
            self.radius = 50
        self.vx = random.uniform(-1, 1)
        self.vy = random.uniform(-18, -14)
        self.gravity = 0.3
        self.rot = random.uniform(0, 360)
        self.rot_spd = random.uniform(-5, 5)
        self.is_exploded = False
        self.entered = False
        self.exp_frame = 0
        self.exp_max = 20

    def update(self):
        if not self.is_exploded:
            self.vy += self.gravity
            self.x += self.vx
            self.y += self.vy
            self.rot += self.rot_spd
            if not self.entered and 0 <= self.y <= WINDOW_HEIGHT - 50:
                self.entered = True
        else:
            self.exp_frame += 1

    def draw(self, frame):
        if not self.is_exploded:
            if self.image is not None:
                overlay_image(frame, self.image, int(self.x), int(self.y), self.rot)
            else:
                color = (0, 0, 255) if self.bomb_type == 'deadly' else (0, 0, 0)
                cv2.circle(frame, (int(self.x), int(self.y)), 40, color, -1)
        elif self.exp_frame < self.exp_max:
            key = 'explosion1' if self.exp_frame % 4 < 2 else 'explosion2'
            img = BOMB_IMAGES.get(key)
            if img is not None:
                a = 1.0 - self.exp_frame / self.exp_max
                overlay_image(frame, img, int(self.x), int(self.y), 0, a)

    def explode(self):
        self.is_exploded = True
        self.exp_frame = 0

    def out(self):
        if not self.is_exploded:
            return self.y > WINDOW_HEIGHT + 150
        return self.exp_frame >= self.exp_max

    def check_collision(self, trail):
        if self.is_exploded or len(trail) < 2:
            return False
        for pt in list(trail)[-15:]:
            if pt and math.hypot(pt[0] - self.x, pt[1] - self.y) < self.radius * 0.8:
                return True
        return False


class ComboEffect:
    def __init__(self, combo):
        self.x = WINDOW_WIDTH // 2
        self.y = WINDOW_HEIGHT // 2 - 100
        self.alpha = 0
        self.scale = 0.5
        self.frame = 0
        self.dur = 60
        self.key = ('combo3' if combo >= 20 else 'combo2' if combo >= 15 else 'combo1' if combo >= 10 else None)

    def update(self):
        self.frame += 1
        if self.frame < 15:
            self.alpha = int(255 * self.frame / 15)
            self.scale = 0.5 + self.frame / 15 * 0.5
        elif self.frame < 45:
            self.alpha = 255; self.scale = 1.0
        else:
            p = (self.frame - 45) / 15
            self.alpha = int(255 * (1 - p))
            self.scale = 1.0 + p * 0.2

    def done(self): return self.frame >= self.dur

    def draw(self, frame):
        if not self.key or self.key not in COMBO_IMAGES:
            return
        img = COMBO_IMAGES[self.key]
        h, w = img.shape[:2]
        sc = cv2.resize(img, (int(w * self.scale), int(h * self.scale)))
        overlay_image(frame, sc, int(self.x), int(self.y), 0, self.alpha / 255.0)


class SlashEffect:
    def __init__(self, x, y, angle):
        self.x = x; self.y = y; self.angle = angle
        self.alpha = 255; self.scale = 1.2
        self.frame = 0; self.dur = 20

    def update(self):
        self.frame += 1
        self.alpha = int(255 * (1 - self.frame / self.dur))
        self.scale = 1.2 + self.frame / self.dur * 0.3

    def done(self): return self.frame >= self.dur

    def draw(self, frame, blade_img):
        if blade_img is None: return
        h, w = blade_img.shape[:2]
        sc = cv2.resize(blade_img, (int(w * self.scale), int(h * self.scale)))
        overlay_image(frame, sc, int(self.x), int(self.y), self.angle, self.alpha / 255.0)


class JuiceEffect:
    def __init__(self, x, y, juice_img):
        self.x = x; self.y = y; self.img = juice_img
        self.alpha = 255; self.scale = 1.2
        self.frame = 0; self.dur = 20

    def update(self):
        self.frame += 1
        if self.frame < self.dur * 0.6:
            self.alpha = 255
        else:
            p = (self.frame - self.dur * 0.6) / (self.dur * 0.4)
            self.alpha = int(255 * (1 - p))
        self.scale = 1.2 + self.frame / self.dur * 0.8

    def done(self): return self.frame >= self.dur

    def draw(self, frame):
        if self.img is None: return
        h, w = self.img.shape[:2]
        sc = cv2.resize(self.img, (int(w * self.scale), int(h * self.scale)))
        overlay_image(frame, sc, int(self.x), int(self.y), 0, min(255, max(0, self.alpha)) / 255.0)


# ============================================================
#  游戏管理类
# ============================================================
class Game:
    def __init__(self, selected_blade='dao1'):
        self.fruits        = []
        self.multi_fruits  = []
        self.bombs         = []
        self.slash_fx      = []
        self.combo_fx      = []
        self.juice_fx      = []
        self.selected_blade = selected_blade
        self.score         = 0
        self.missed        = 0
        self.bombs_hit     = 0
        self.combo         = 0
        self.max_combo     = 0
        self.last_milestone = 0
        self.spawn_timer   = 0
        self.spawn_interval = 25
        self.game_over     = False
        self.game_over_reason = ""
        self.max_missed    = 10
        self.max_bombs_hit = 3
        self.max_on_screen = 10
        self.bomb_chance   = 0.20
        self.deadly_chance = 0.30
        self.multi_chance  = 0.15
        self.last_type     = None
        self.consec_bombs  = 0
        self.max_consec_bombs = 2
        self.spawn_single()

    # ---- 生成 ----
    def spawn_single(self):
        if self.consec_bombs >= self.max_consec_bombs:
            self._spawn_fruit(); self.last_type = 'fruit'; self.consec_bombs = 0; return
        bomb = random.random() < self.bomb_chance
        if self.last_type == 'bomb' and bomb and random.random() < 0.5:
            bomb = False
        if bomb:
            t = 'deadly' if random.random() < self.deadly_chance else 'normal'
            self.bombs.append(Bomb(t))
            self.last_type = 'bomb'; self.consec_bombs += 1
        else:
            self._spawn_fruit(); self.last_type = 'fruit'; self.consec_bombs = 0

    def _spawn_fruit(self):
        if random.random() < self.multi_chance and MULTI_FRUIT_IMAGES:
            self.multi_fruits.append(MultiFruit())
        else:
            self.fruits.append(Fruit())

    # ---- 更新 ----
    def update(self):
        if self.game_over: return
        active = (sum(1 for f in self.fruits       if not f.is_cut) +
                  sum(1 for f in self.multi_fruits  if not f.is_cut) +
                  sum(1 for b in self.bombs         if not b.is_exploded))
        if active < self.max_on_screen:
            self.spawn_timer += 1
            if self.spawn_timer >= self.spawn_interval:
                self.spawn_single()
                self.spawn_timer = 0
                self.spawn_interval = max(40, self.spawn_interval - 1)

        for lst in [self.fruits, self.multi_fruits]:
            for obj in lst[:]:
                obj.update()
                if obj.out():
                    if not obj.is_cut and obj.entered:
                        self.missed += 1
                        self.combo = 0; self.last_milestone = 0
                        if self.missed >= self.max_missed:
                            self.game_over = True
                            self.game_over_reason = "漏掉太多水果！"
                    lst.remove(obj)

        for b in self.bombs[:]:
            b.update()
            if b.out(): self.bombs.remove(b)

        for lst in [self.slash_fx, self.combo_fx, self.juice_fx]:
            for e in lst[:]:
                e.update()
                if e.done(): lst.remove(e)

    # ---- 碰撞检测（支持多条轨迹 - 双手模式）----
    def check_collisions(self, trail_list: list):
        """trail_list: 轨迹队列列表，单手=[trail]，双手=[trail1, trail2]"""
        fruit_hit = False
        for trail in trail_list:
            if len(trail) < 2:
                continue
            pts = list(trail)[-5:]
            if len(pts) >= 2:
                cut_angle = math.degrees(math.atan2(pts[-1][1]-pts[0][1], pts[-1][0]-pts[0][0]))
            else:
                cut_angle = 0

            for f in self.fruits:
                if not f.is_cut and f.check_collision(trail):
                    f.cut(); self.score += 10; fruit_hit = True
                    self.slash_fx.append(SlashEffect(f.x, f.y, cut_angle))
                    self._add_juice(f.x, f.y, f.fruit_type)
                    self._play('slice')

            for f in self.multi_fruits:
                if not f.is_cut and f.check_collision(trail):
                    f.cut(cut_angle); self.score += 20; fruit_hit = True
                    self.slash_fx.append(SlashEffect(f.x, f.y, cut_angle))
                    self._add_juice(f.x, f.y, f.fruit_type)
                    self._play('slice')

            for b in self.bombs:
                if not b.is_exploded and b.check_collision(trail):
                    b.explode()
                    self.slash_fx.append(SlashEffect(b.x, b.y, cut_angle))
                    self._play('explosion')
                    self.combo = 0; self.last_milestone = 0
                    if b.bomb_type == 'deadly':
                        self.game_over = True
                        self.game_over_reason = "切到致命炸弹！游戏结束！"
                    else:
                        self.score = max(0, self.score - 20)
                        self.bombs_hit += 1
                        if self.bombs_hit >= self.max_bombs_hit:
                            self.game_over = True
                            self.game_over_reason = "切到太多炸弹！"

        if fruit_hit:
            self.combo += 1
            if self.combo > self.max_combo: self.max_combo = self.combo
            self._check_milestone()

    def _add_juice(self, x, y, ftype):
        if not JUICE_IMAGES: return
        c = get_juice_color(ftype)
        if c: self.juice_fx.append(JuiceEffect(x, y, JUICE_IMAGES[c]))

    def _play(self, key):
        if HAS_SOUND and key in SOUND_EFFECTS:
            SOUND_EFFECTS[key].play()

    def _check_milestone(self):
        for m in [10, 15, 20]:
            if self.combo >= m and self.last_milestone < m:
                self.last_milestone = m
                self.combo_fx.append(ComboEffect(m))
                break

    # ---- 绘制 ----
    def draw(self, frame):
        for obj in self.fruits:       obj.draw(frame)
        for obj in self.multi_fruits: obj.draw(frame)
        for obj in self.bombs:        obj.draw(frame)

        blade_img = BLADE_IMAGES.get(self.selected_blade)
        for e in self.slash_fx: e.draw(frame, blade_img)
        for e in self.combo_fx: e.draw(frame)
        for e in self.juice_fx: e.draw(frame)

        # HUD - 使用中文批量渲染
        add_cn_text(f'得分: {self.score}',              (20, 45),  font_size=38, color=(255,255,255), bg_color=(0,180,0))
        add_cn_text(f'漏掉: {self.missed}/{self.max_missed}',   (20, 110), font_size=32, color=(255,255,255), bg_color=(0,0,200))
        add_cn_text(f'炸弹: {self.bombs_hit}/{self.max_bombs_hit}', (20, 170), font_size=32, color=(255,255,255), bg_color=(0,100,255))
        if self.combo > 0:
            combo_bg = (0, 165, 255) if self.combo >= 10 else (80, 80, 80)
            add_cn_text(f'连击: ×{self.combo}',          (20, 230), font_size=36, color=(255,255,255), bg_color=combo_bg)

        if self.game_over:
            # 半透明遮罩
            ov = frame.copy()
            cv2.rectangle(ov, (0, 0), (WINDOW_WIDTH, WINDOW_HEIGHT), (0, 0, 0), -1)
            cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
            cx = WINDOW_WIDTH // 2
            cy = WINDOW_HEIGHT // 2
            add_cn_text('游戏结束！', (cx - 150, cy - 120), font_size=72, color=(255, 60, 60))
            if self.game_over_reason:
                add_cn_text(self.game_over_reason, (cx - 200, cy - 30), font_size=40, color=(255, 200, 0))
            add_cn_text(f'最终得分: {self.score}', (cx - 180, cy + 40),  font_size=48, color=(0, 255, 100))
            if self.max_combo > 0:
                add_cn_text(f'最大连击: ×{self.max_combo}', (cx - 160, cy + 110), font_size=38, color=(0, 215, 255))
            add_cn_text('按 R 重新开始   按 Q 退出', (cx - 220, cy + 170), font_size=32, color=(200, 200, 200))

        flush_cn_texts(frame)


# ============================================================
#  选择界面辅助：悬停进度条
# ============================================================
def _draw_hover_bar(frame, area, progress):
    x, y, w, h = area['x'], area['y'], area['w'], area['h']
    bx, by = x + 10, y + h + 15
    bw, bh = w - 20, 25
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (60, 60, 60), -1)
    cv2.rectangle(frame, (bx, by), (bx + int(bw * progress), by + bh), (0, 220, 0), -1)
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (200, 200, 200), 2)


# ============================================================
#  开局选择1：游戏模式（单手 / 双手）
# ============================================================
def mode_selection_screen(cap) -> str:
    """
    返回 'single' 或 'dual'
    左半屏 = 单手，右半屏 = 双手；悬停食指3秒确认
    """
    areas = {
        'single': {'x': 150, 'y': 250, 'w': 380, 'h': 300},
        'dual':   {'x': 750, 'y': 250, 'w': 380, 'h': 300},
    }
    hover_timer   = {'single': 0, 'dual': 0}
    threshold     = 90   # 3秒 × 30fps
    current_hover = None
    smoother      = FingerSmoother(method='ewma', alpha=0.4, adaptive=True)

    labels = {'single': '单手模式', 'dual': '双手模式'}
    desc   = {'single': '一只手切水果', 'dual': '双手同时切水果'}

    while True:
        ret, frame = cap.read()
        if not ret:
            return 'single'
        frame = cv2.flip(frame, 1)
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (WINDOW_WIDTH, WINDOW_HEIGHT), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)

        # 标题
        add_cn_text('选择游戏模式', (WINDOW_WIDTH//2 - 180, 60),  font_size=60, color=(255, 220, 0))
        add_cn_text('悬停食指 3 秒确认选择', (WINDOW_WIDTH//2 - 200, 145), font_size=34, color=(200, 200, 200))

        for key, area in areas.items():
            x, y, w, h = area['x'], area['y'], area['w'], area['h']
            hovering  = (current_hover == key)
            progress  = hover_timer[key] / threshold if hovering else 0
            bcolor    = (0, 255, 0) if hovering else (180, 180, 180)
            thick     = 5 if hovering else 2

            # 选择框
            cv2.rectangle(frame, (x, y), (x + w, y + h), bcolor, thick)
            # 高亮填充
            if hovering:
                hl = frame.copy()
                cv2.rectangle(hl, (x, y), (x + w, y + h), (0, 80, 0), -1)
                cv2.addWeighted(hl, 0.25, frame, 0.75, 0, frame)

            add_cn_text(labels[key], (x + 60, y + 80),  font_size=48, color=(255, 255, 100))
            add_cn_text(desc[key],   (x + 30, y + 160), font_size=28, color=(200, 200, 200))

            if hovering and progress > 0:
                _draw_hover_bar(frame, area, progress)

        # 手部检测
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        current_hover = None
        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0].landmark[8]
            rx, ry = int(lm.x * WINDOW_WIDTH), int(lm.y * WINDOW_HEIGHT)
            sx, sy = smoother.smooth(rx, ry)
            cv2.circle(frame, (sx, sy), 22, (0, 255, 255), 3)
            cv2.circle(frame, (sx, sy), 12, (0, 255, 0), cv2.FILLED)
            for key, area in areas.items():
                ax, ay, aw, ah = area['x'], area['y'], area['w'], area['h']
                if ax <= sx <= ax + aw and ay <= sy <= ay + ah:
                    current_hover = key
                    hover_timer[key] += 1
                    if hover_timer[key] >= threshold:
                        flush_cn_texts(frame)
                        cv2.destroyWindow('Swift-Fruit-Slice')
                        return key
                else:
                    hover_timer[key] = max(0, hover_timer[key] - 2)
        else:
            smoother.reset()
            for k in hover_timer: hover_timer[k] = max(0, hover_timer[k] - 2)

        flush_cn_texts(frame)
        cv2.imshow('Swift-Fruit-Slice', frame)
        key_press = cv2.waitKey(1) & 0xFF
        if key_press == ord('q'):
            cv2.destroyWindow('Swift-Fruit-Slice')
            return 'single'
        elif key_press == ord('1'):
            cv2.destroyWindow('Swift-Fruit-Slice'); return 'single'
        elif key_press == ord('2'):
            cv2.destroyWindow('Swift-Fruit-Slice'); return 'dual'


# ============================================================
#  开局选择2：刀光样式
# ============================================================
def blade_selection_screen(cap) -> str:
    """返回 'dao1' 或 'dao2'"""
    areas = {
        'dao1': {'x': 200, 'y': 280, 'w': 300, 'h': 300},
        'dao2': {'x': 780, 'y': 280, 'w': 300, 'h': 300},
    }
    hover_timer   = {'dao1': 0, 'dao2': 0}
    threshold     = 90
    current_hover = None
    smoother      = FingerSmoother(method='ewma', alpha=0.4, adaptive=True)
    labels        = {'dao1': '刀光 一', 'dao2': '刀光 二'}

    while True:
        ret, frame = cap.read()
        if not ret:
            cv2.destroyWindow('Swift-Fruit-Slice')
            return 'dao1'
        frame = cv2.flip(frame, 1)
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (WINDOW_WIDTH, WINDOW_HEIGHT), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)

        add_cn_text('选择刀光样式', (WINDOW_WIDTH//2 - 160, 60), font_size=60, color=(0, 220, 255))
        add_cn_text('悬停食指 3 秒确认选择', (WINDOW_WIDTH//2 - 200, 148), font_size=34, color=(200, 200, 200))

        for key, area in areas.items():
            x, y, w, h = area['x'], area['y'], area['w'], area['h']
            hovering = (current_hover == key)
            progress = hover_timer[key] / threshold if hovering else 0
            bcolor   = (0, 255, 0) if hovering else (200, 200, 200)
            thick    = 5 if hovering else 2
            cv2.rectangle(frame, (x, y), (x + w, y + h), bcolor, thick)
            if hovering:
                hl = frame.copy()
                cv2.rectangle(hl, (x, y), (x + w, y + h), (0, 80, 0), -1)
                cv2.addWeighted(hl, 0.25, frame, 0.75, 0, frame)

            # 刀光预览
            blade_img = BLADE_IMAGES.get(key)
            if blade_img is not None:
                bh, bw = blade_img.shape[:2]
                sc = min((w - 20) / bw, (h - 20) / bh, 0.9)
                sized = cv2.resize(blade_img, (int(bw * sc), int(bh * sc)))
                overlay_image(frame, sized, x + w // 2, y + h // 2, 0, 1.0)

            add_cn_text(labels[key], (x + 60, y - 55), font_size=36, color=(255, 255, 255))
            if hovering and progress > 0:
                _draw_hover_bar(frame, area, progress)

        # 手部检测
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        current_hover = None
        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0].landmark[8]
            rx, ry = int(lm.x * WINDOW_WIDTH), int(lm.y * WINDOW_HEIGHT)
            sx, sy = smoother.smooth(rx, ry)
            cv2.circle(frame, (sx, sy), 22, (0, 255, 255), 3)
            cv2.circle(frame, (sx, sy), 12, (0, 255, 0), cv2.FILLED)
            for key, area in areas.items():
                ax, ay, aw, ah = area['x'], area['y'], area['w'], area['h']
                if ax <= sx <= ax + aw and ay <= sy <= ay + ah:
                    current_hover = key
                    hover_timer[key] += 1
                    if hover_timer[key] >= threshold:
                        flush_cn_texts(frame)
                        cv2.destroyWindow('Swift-Fruit-Slice')
                        return key
                else:
                    hover_timer[key] = max(0, hover_timer[key] - 2)
        else:
            smoother.reset()
            for k in hover_timer: hover_timer[k] = max(0, hover_timer[k] - 2)

        flush_cn_texts(frame)
        cv2.imshow('Swift-Fruit-Slice', frame)
        kp = cv2.waitKey(1) & 0xFF
        if kp == ord('q'):
            cv2.destroyWindow('Swift-Fruit-Slice'); return 'dao1'
        elif kp == ord('1'):
            cv2.destroyWindow('Swift-Fruit-Slice'); return 'dao1'
        elif kp == ord('2'):
            cv2.destroyWindow('Swift-Fruit-Slice'); return 'dao2'


# ============================================================
#  主函数
# ============================================================
def main():
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WINDOW_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)

    # ---- 开局选择流程 ----
    game_mode     = mode_selection_screen(cap)   # 'single' 或 'dual'
    selected_blade = blade_selection_screen(cap)

    dual = (game_mode == 'dual')
    n_hands = 2 if dual else 1

    print(f"\n✓ 游戏模式: {'双手模式' if dual else '单手模式'}")
    print(f"✓ 刀光: {selected_blade}")

    # ---- 为每只手创建独立的平滑器和轨迹队列 ----
    # slot 0 = Left（镜像后屏幕左侧，即用户右手）
    # slot 1 = Right（镜像后屏幕右侧，即用户左手）
    # 固定 slot 绑定，防止 MediaPipe 每帧返回顺序不一致导致的轨迹横跳
    smoothers    = [FingerSmoother(method='ewma', alpha=0.4, adaptive=True) for _ in range(n_hands)]
    trail_list   = [deque(maxlen=15) for _ in range(n_hands)]
    finger_pos   = [None] * n_hands
    finger_speed = [0.0] * n_hands
    # 记录上一帧每个 slot 的位置，用于位置距离兜底校验
    last_pos     = [None] * n_hands

    # handedness 标签 → slot 映射（图像已翻转，所以 MediaPipe "Left" = 屏幕左侧）
    HAND_SLOT = {'Left': 0, 'Right': 1}

    game = Game(selected_blade)
    debug_mode = False

    # 手指颜色（双手时用不同颜色区分）
    HAND_COLORS = [(0, 255, 0), (0, 100, 255)]

    print("=" * 50)
    print("🍉 体感切水果 - 启动成功！")
    print(f"   素材: {len(FRUIT_IMAGES)} 种普通水果, {len(MULTI_FRUIT_IMAGES)} 种特殊水果")
    print("   按 R 重新开始 | Q 退出 | D 调试模式")
    print("=" * 50)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        # ---- 手部检测 ----
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results   = hands.process(rgb_frame)

        # 重置本帧指尖位置
        finger_pos   = [None] * n_hands
        finger_speed = [0.0] * n_hands
        active_slots = set()

        if results.multi_hand_landmarks and results.multi_handedness:
            # ============================================================
            # 双重保障策略：
            # 1. 优先用 handedness 标签固定 slot（Left→0, Right→1）
            # 2. 若同一 slot 重复出现（两只手被检测成同一侧），
            #    则用"与上帧位置最近"的方式重新分配
            # ============================================================
            # 先收集所有检测到的手的原始位置
            raw_hands = []
            for hand_lm, handedness_info in zip(
                    results.multi_hand_landmarks[:n_hands],
                    results.multi_handedness[:n_hands]):
                label = handedness_info.classification[0].label  # 'Left' or 'Right'
                tip   = hand_lm.landmark[8]
                rx    = int(tip.x * WINDOW_WIDTH)
                ry    = int(tip.y * WINDOW_HEIGHT)
                raw_hands.append({'label': label, 'rx': rx, 'ry': ry})

            if n_hands == 1:
                # 单手模式：直接用 slot 0
                h = raw_hands[0]
                slot = 0
                sx, sy = smoothers[slot].smooth(h['rx'], h['ry'])
                finger_pos[slot]   = (sx, sy)
                finger_speed[slot] = math.hypot(h['rx'] - (last_pos[slot][0] if last_pos[slot] else h['rx']),
                                                h['ry'] - (last_pos[slot][1] if last_pos[slot] else h['ry']))
                trail_list[slot].append((sx, sy))
                last_pos[slot] = (h['rx'], h['ry'])
                active_slots.add(slot)

            else:
                # 双手模式：按 handedness 标签分配 slot
                # 步骤1：尝试按标签直接分配
                label_assigned = {}  # slot → hand_dict
                conflict = []        # 标签冲突的手（两只都检测成同侧）
                for h in raw_hands:
                    preferred_slot = HAND_SLOT.get(h['label'], 0)
                    if preferred_slot not in label_assigned:
                        label_assigned[preferred_slot] = h
                    else:
                        conflict.append(h)  # 该 slot 已被占用

                # 步骤2：冲突的手用位置距离兜底分配到空闲 slot
                for h in conflict:
                    free_slots = [s for s in range(n_hands) if s not in label_assigned]
                    if not free_slots:
                        break
                    if len(free_slots) == 1:
                        label_assigned[free_slots[0]] = h
                    else:
                        # 找与上帧位置最近的空闲 slot
                        best_slot = free_slots[0]
                        best_dist = float('inf')
                        for s in free_slots:
                            if last_pos[s]:
                                d = math.hypot(h['rx'] - last_pos[s][0], h['ry'] - last_pos[s][1])
                                if d < best_dist:
                                    best_dist = d
                                    best_slot = s
                        label_assigned[best_slot] = h

                # 步骤3：额外校验——若两个 slot 的分配位置与上帧相比发生了明显"互换"
                #         （即本帧 slot0 的位置比 slot1 更靠近上帧 slot1），则对调
                if (len(label_assigned) == 2 and
                        last_pos[0] is not None and last_pos[1] is not None):
                    h0 = label_assigned.get(0)
                    h1 = label_assigned.get(1)
                    if h0 and h1:
                        # 正常配对距离
                        d_normal = (math.hypot(h0['rx'] - last_pos[0][0], h0['ry'] - last_pos[0][1]) +
                                    math.hypot(h1['rx'] - last_pos[1][0], h1['ry'] - last_pos[1][1]))
                        # 对调配对距离
                        d_swap   = (math.hypot(h0['rx'] - last_pos[1][0], h0['ry'] - last_pos[1][1]) +
                                    math.hypot(h1['rx'] - last_pos[0][0], h1['ry'] - last_pos[0][1]))
                        # 只有在对调距离明显更小（差超过 80px）时才对调，避免误触发
                        if d_swap < d_normal - 80:
                            label_assigned[0], label_assigned[1] = h1, h0

                # 步骤4：写入结果
                for slot, h in label_assigned.items():
                    sx, sy = smoothers[slot].smooth(h['rx'], h['ry'])
                    finger_pos[slot]   = (sx, sy)
                    finger_speed[slot] = math.hypot(
                        h['rx'] - (last_pos[slot][0] if last_pos[slot] else h['rx']),
                        h['ry'] - (last_pos[slot][1] if last_pos[slot] else h['ry']))
                    trail_list[slot].append((sx, sy))
                    last_pos[slot] = (h['rx'], h['ry'])
                    active_slots.add(slot)

        # 未检测到的 slot：重置平滑器，last_pos 清空
        for i in range(n_hands):
            if i not in active_slots:
                smoothers[i].reset()
                last_pos[i] = None

        # ---- 轨迹渐消 ----
        for i in range(n_hands):
            if finger_pos[i] is None and len(trail_list[i]) > 10:
                for _ in range(min(5, len(trail_list[i]))):
                    if trail_list[i]: trail_list[i].popleft()

        # ---- 绘制轨迹线条 ----
        trail_colors = [(100, 100, 255), (100, 255, 100)]
        for i, trail in enumerate(trail_list):
            tc = trail_colors[i % 2]
            pts = list(trail)
            for j in range(1, len(pts)):
                if pts[j] and pts[j-1]:
                    a = (j / len(pts)) * 0.35
                    thickness = max(2, int(2 + a * 6))
                    cv2.line(frame, pts[j-1], pts[j], tc, thickness)

        # ---- 绘制手指标记 ----
        for i, pos in enumerate(finger_pos):
            if pos:
                spd = finger_speed[i]
                if spd > 20:   dot_c = (0, 0, 255)
                elif spd > 10: dot_c = (0, 165, 255)
                else:          dot_c = HAND_COLORS[i]
                cv2.circle(frame, pos, 25, dot_c, 3)
                cv2.circle(frame, pos, 15, (0, 255, 255), cv2.FILLED)
                if debug_mode:
                    cv2.putText(frame, f'S:{spd:.0f}', (pos[0]+28, pos[1]-15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

        # ---- 双手模式：显示"双手已就绪"提示 ----
        if dual:
            hands_active = sum(1 for p in finger_pos if p is not None)
            if hands_active == 2:
                add_cn_text('双手已就绪 ✓', (WINDOW_WIDTH - 260, 20),
                            font_size=26, color=(0, 255, 100))
            elif hands_active == 1:
                add_cn_text('等待第二只手...', (WINDOW_WIDTH - 280, 20),
                            font_size=26, color=(0, 180, 255))

        # ---- 游戏逻辑 ----
        if not game.game_over:
            game.update()
            game.check_collisions(trail_list)

        game.draw(frame)

        # 调试信息
        if debug_mode:
            cv2.putText(frame, '[DEBUG]', (WINDOW_WIDTH - 160, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow('Swift-Fruit-Slice', frame)

        # ---- 键盘控制 ----
        kp = cv2.waitKey(1) & 0xFF
        if kp == ord('q'):
            break
        elif kp == ord('r'):
            game = Game(selected_blade)
            for t in trail_list: t.clear()
            for s in smoothers:  s.reset()
        elif kp == ord('d'):
            debug_mode = not debug_mode
            print(f"✓ 调试模式: {'开' if debug_mode else '关'}")
        elif kp == ord('1'):
            smoothers = [FingerSmoother(method='ewma', alpha=0.4, adaptive=True) for _ in range(n_hands)]
            print("✓ EWMA 自适应平滑")
        elif kp == ord('2'):
            smoothers = [FingerSmoother(method='moving_avg', buffer_size=5, adaptive=True) for _ in range(n_hands)]
            print("✓ 移动平均平滑")
        elif kp == ord('3'):
            smoothers = [FingerSmoother(method='kalman', adaptive=False) for _ in range(n_hands)]
            print("✓ 卡尔曼滤波")

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()