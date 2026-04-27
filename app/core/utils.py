"""
[核心基础层 - 工具箱]
存放通用的、不涉及具体业务逻辑的小工具。
1. 图片叠加与旋转 (overlay_image)
2. 中文文字渲染 (PIL 驱动，支持批量渲染以提高性能)
3. 颜色计算与映射
"""
import cv2
import numpy as np
import random
import os
from PIL import Image, ImageDraw, ImageFont

# ============================================================
#  中文字体系统
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
        # Linux
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        # macOS
        "/System/Library/Fonts/PingFang.ttc",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, size)
                _font_cache[size] = font
                return font
            except Exception:
                continue
    return ImageFont.load_default()

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
                fill=(bg_color[2], bg_color[1], bg_color[0])
            )
        draw.text((x, y), text, font=font,
                  fill=(color[2], color[1], color[0]))
    _text_queue = []
    np.copyto(frame, cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR))

# ============================================================
#  图片处理
# ============================================================
def overlay_image(background, overlay, x, y, rotation=0, alpha=1.0):
    """将带透明通道的图片叠加到背景上"""
    if overlay is None or overlay.size == 0:
        return
    
    try:
        overlay_copy = overlay.copy()
        
        # 旋转图片
        if rotation != 0:
            h, w = overlay_copy.shape[:2]
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, rotation, 1.0)
            
            cos = np.abs(matrix[0, 0])
            sin = np.abs(matrix[0, 1])
            new_w = int((h * sin) + (w * cos))
            new_h = int((h * cos) + (w * sin))
            
            matrix[0, 2] += (new_w / 2) - center[0]
            matrix[1, 2] += (new_h / 2) - center[1]
            
            overlay_copy = cv2.warpAffine(overlay_copy, matrix, (new_w, new_h), 
                                         borderMode=cv2.BORDER_CONSTANT, 
                                         borderValue=(0, 0, 0, 0))
        
        h, w = overlay_copy.shape[:2]
        x1, y1 = int(x - w // 2), int(y - h // 2)
        x2, y2 = x1 + w, y1 + h
        
        if x1 >= background.shape[1] or y1 >= background.shape[0] or x2 <= 0 or y2 <= 0:
            return
        
        overlay_x1, overlay_y1 = max(0, -x1), max(0, -y1)
        overlay_x2, overlay_y2 = w - max(0, x2 - background.shape[1]), h - max(0, y2 - background.shape[0])
        
        bg_x1, bg_y1 = max(0, x1), max(0, y1)
        bg_x2, bg_y2 = min(background.shape[1], x2), min(background.shape[0], y2)
        
        if overlay_x2 <= overlay_x1 or overlay_y2 <= overlay_y1:
            return
        
        overlay_img = overlay_copy[overlay_y1:overlay_y2, overlay_x1:overlay_x2]
        if overlay_img.size == 0: return
        
        if len(overlay_img.shape) == 3 and overlay_img.shape[2] == 4:
            overlay_colors = overlay_img[:, :, :3]
            overlay_alpha = overlay_img[:, :, 3:] / 255.0 * alpha
        else:
            overlay_colors = overlay_img[:, :, :3] if len(overlay_img.shape) == 3 else overlay_img
            overlay_alpha = np.ones((overlay_img.shape[0], overlay_img.shape[1], 1)) * alpha
        
        bg_region = background[bg_y1:bg_y2, bg_x1:bg_x2]
        if bg_region.shape[:2] != overlay_colors.shape[:2]: return
        
        background[bg_y1:bg_y2, bg_x1:bg_x2] = (
            overlay_alpha * overlay_colors + (1 - overlay_alpha) * bg_region
        ).astype(np.uint8)
    except Exception:
        pass

def get_juice_color_for_fruit(fruit_type, juice_images):
    """根据水果类型返回对应的果汁颜色"""
    if not juice_images:
        return None
    
    fruit_juice_map = {
        'banana': 'orange', 'boluo': 'orange', 'iceBanana': 'green',
        'Mango': 'orange', 'mugua': 'orange', 'peach': 'pink',
        'pear': 'green', 'pineapple': 'orange', 'strawberry': 'red',
        'watermelon': 'red', 'dragonfruit': 'pink', 'b1': 'orange',
    }
    juice_color = fruit_juice_map.get(fruit_type)
    if juice_color and juice_color in juice_images:
        return juice_color
    return random.choice(list(juice_images.keys()))
