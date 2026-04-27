"""
[界面层 - 柜台服务员]
负责与用户的直接交互界面。
1. 模式选择界面 (mode_selection_screen): 单手/双手/双人PK。
2. 刀光选择界面 (blade_selection_screen): 选择不同的刀光样式。
"""
import cv2
import numpy as np
from app.config import WINDOW_WIDTH, WINDOW_HEIGHT
from app.core.utils import overlay_image, add_cn_text, flush_cn_texts

def _draw_hover_bar(frame, area, progress):
    """绘制悬停进度条"""
    x, y, w, h = area['x'], area['y'], area['w'], area['h']
    bx, by = x + 10, y + h + 15
    bw, bh = w - 20, 25
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (60, 60, 60), -1)
    cv2.rectangle(frame, (bx, by), (bx + int(bw * progress), by + bh), (0, 220, 0), -1)
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (200, 200, 200), 2)

def mode_selection_screen(cap, tracker):
    """选择游戏模式界面"""
    BOX_W, BOX_H = 320, 280
    GAP = (WINDOW_WIDTH - 3 * BOX_W) // 4
    areas = {
        'single': {'x': GAP, 'y': 240, 'w': BOX_W, 'h': BOX_H},
        'dual': {'x': GAP*2 + BOX_W, 'y': 240, 'w': BOX_W, 'h': BOX_H},
        'pk': {'x': GAP*3 + BOX_W*2, 'y': 240, 'w': BOX_W, 'h': BOX_H},
    }
    hover_timer = {k: 0 for k in areas}
    threshold = 90
    current_hover = None
    
    labels = {'single': '单手模式', 'dual': '双手模式', 'pk': '双人PK'}
    descs = {'single': '一只手切水果', 'dual': '双手同时切水果', 'pk': '两人对战 30秒'}
    colors = {'single': (255, 220, 80), 'dual': (80, 220, 255), 'pk': (255, 100, 100)}

    while True:
        ret, frame = cap.read()
        if not ret: return 'single'
        frame = cv2.flip(frame, 1)
        
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (WINDOW_WIDTH, WINDOW_HEIGHT), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)

        add_cn_text('选择游戏模式', (WINDOW_WIDTH//2 - 180, 55), font_size=60, color=(255, 220, 0))
        add_cn_text('悬停食指 3 秒确认', (WINDOW_WIDTH//2 - 160, 140), font_size=34, color=(200, 200, 200))

        for key, area in areas.items():
            x, y, w, h = area['x'], area['y'], area['w'], area['h']
            hovering = (current_hover == key)
            progress = hover_timer[key] / threshold if hovering else 0
            bcolor = (0, 255, 0) if hovering else (160, 160, 160)
            cv2.rectangle(frame, (x, y), (x + w, y + h), bcolor, 5 if hovering else 2)
            
            add_cn_text(labels[key], (x + 30, y + 70), font_size=44, color=colors[key])
            add_cn_text(descs[key], (x + 20, y + 155), font_size=26, color=(200, 200, 200))
            if hovering and progress > 0: _draw_hover_bar(frame, area, progress)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        lm_list, _ = tracker.process(rgb)
        current_hover = None
        if lm_list:
            tip = lm_list[0][8]
            sx, sy = int(tip.x * WINDOW_WIDTH), int(tip.y * WINDOW_HEIGHT)
            cv2.circle(frame, (sx, sy), 22, (0, 255, 255), 3)
            cv2.circle(frame, (sx, sy), 12, (0, 255, 0), -1)
            for key, area in areas.items():
                if area['x'] <= sx <= area['x'] + area['w'] and area['y'] <= sy <= area['y'] + area['h']:
                    current_hover = key; hover_timer[key] += 1
                    if hover_timer[key] >= threshold: return key
                else: hover_timer[key] = max(0, hover_timer[key] - 2)
        else:
            for k in hover_timer: hover_timer[k] = max(0, hover_timer[k] - 2)

        flush_cn_texts(frame)
        cv2.imshow('Fruit Ninja Refactored', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): return 'single'

def blade_selection_screen(cap, tracker, asset_manager):
    """选择刀光样式界面"""
    areas = {
        'dao1': {'x': 200, 'y': 280, 'w': 300, 'h': 300},
        'dao2': {'x': 780, 'y': 280, 'w': 300, 'h': 300},
    }
    hover_timer = {'dao1': 0, 'dao2': 0}
    threshold = 90
    current_hover = None
    labels = {'dao1': '刀光 一', 'dao2': '刀光 二'}

    while True:
        ret, frame = cap.read()
        if not ret: return 'dao1'
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
            bcolor = (0, 255, 0) if hovering else (200, 200, 200)
            cv2.rectangle(frame, (x, y), (x + w, y + h), bcolor, 5 if hovering else 2)
            
            img = asset_manager.blade_images.get(key)
            if img is not None:
                bh, bw = img.shape[:2]
                sc = min((w - 20) / bw, (h - 20) / bh, 0.9)
                sized = cv2.resize(img, (int(bw * sc), int(bh * sc)))
                overlay_image(frame, sized, x + w // 2, y + h // 2)
            
            add_cn_text(labels[key], (x + 60, y - 55), font_size=36, color=(255, 255, 255))
            if hovering and progress > 0: _draw_hover_bar(frame, area, progress)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        lm_list, _ = tracker.process(rgb)
        current_hover = None
        if lm_list:
            tip = lm_list[0][8]
            sx, sy = int(tip.x * WINDOW_WIDTH), int(tip.y * WINDOW_HEIGHT)
            cv2.circle(frame, (sx, sy), 22, (0, 255, 255), 3)
            cv2.circle(frame, (sx, sy), 12, (0, 255, 0), -1)
            for key, area in areas.items():
                if area['x'] <= sx <= area['x'] + area['w'] and area['y'] <= sy <= area['y'] + area['h']:
                    current_hover = key; hover_timer[key] += 1
                    if hover_timer[key] >= threshold: return key
                else: hover_timer[key] = max(0, hover_timer[key] - 2)
        else:
            for k in hover_timer: hover_timer[k] = max(0, hover_timer[k] - 2)

        flush_cn_texts(frame)
        cv2.imshow('Fruit Ninja Refactored', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): return 'dao1'
