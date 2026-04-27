"""
[主应用入口 - 导演角色]
负责组装和启动整个游戏。它不处理具体的业务逻辑，而是协调各个服务(Services)和模型(Models)的工作。

调用流程:
1. 初始化摄像头 (cv2.VideoCapture)
2. 加载资源 (AssetManager)
3. 启动手势追踪 (HandTracker)
4. 选择模式和刀光 (mode_selection_screen, blade_selection_screen)
5. 根据模式进入不同的游戏循环 (Single/Dual/PK)
"""
import os
import sys

# 将项目根目录添加到 sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

import cv2
import math
from collections import deque
from app.config import WINDOW_WIDTH, WINDOW_HEIGHT, FPS, TRAIL_LENGTH
from app.services.asset_manager import AssetManager
from app.services.hand_tracker import HandTracker, FingerSmoother
from app.services.game_engine import GameEngine, PKGameEngine
from app.interfaces.gui import mode_selection_screen, blade_selection_screen
from app.core.logger import logger
from app.core.utils import add_cn_text

def run_game():
    """启动游戏的入口函数"""
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

    # 1. 初始化资源
    assets = AssetManager().load_all()
    # 2. 初始手势追踪器 (先按单手创建)
    tracker = HandTracker(num_hands=1)
    
    # 3. 初始化摄像头
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    
    # 4. 选择模式和刀光
    game_mode = mode_selection_screen(cap, tracker)
    selected_blade = blade_selection_screen(cap, tracker, assets)
    
    # 根据模式调整手势追踪手数
    dual = (game_mode == 'dual')
    pk_mode = (game_mode == 'pk')
    num_hands = 2 if (dual or pk_mode) else 1
    tracker.reinit(num_hands)
    
    logger.info(f"✓ 游戏模式: {game_mode}, 刀光: {selected_blade}, 检测手数: {num_hands}")
    
    # 5. 进入双人 PK 模式循环
    if pk_mode:
        run_pk_mode(cap, tracker, assets, selected_blade)
    else:
        # 进入单人/双手模式循环
        run_standard_mode(cap, tracker, assets, selected_blade, num_hands, dual)

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    tracker.close()

def run_pk_mode(cap, tracker, assets, blade):
    """双人 PK 模式主循环"""
    game = PKGameEngine(assets, blade)
    smoothers = [FingerSmoother(method='ewma', alpha=0.4, adaptive=True) for _ in range(2)]
    trails = [deque(maxlen=TRAIL_LENGTH) for _ in range(2)]
    hand_colors = [(0, 255, 0), (0, 100, 255)] # 玩家1绿色，玩家2蓝色
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        lm_list, label_list = tracker.process(rgb)
        active_finger_pos = [None, None]
        
        # 处理手势并分配给两个玩家
        if lm_list:
            for lm, label in zip(lm_list[:2], label_list[:2]):
                tip = lm[8]
                rx, ry = int(tip.x * WINDOW_WIDTH), int(tip.y * WINDOW_HEIGHT)
                # 简单的左右手分配：左半屏玩家1，右半屏玩家2
                slot = 0 if rx < WINDOW_WIDTH // 2 else 1
                sx, sy = smoothers[slot].smooth(rx, ry)
                active_finger_pos[slot] = (sx, sy)
                trails[slot].append((sx, sy))
        
        # 轨迹渐消
        for i in range(2):
            if active_finger_pos[i] is None and len(trails[i]) > 5:
                trails[i].popleft()

        # 绘制轨迹
        trail_colors_pk = [(120, 80, 255), (255, 80, 80)]
        for i, trail in enumerate(trails):
            pts = list(trail)
            for j in range(1, len(pts)):
                if pts[j] and pts[j-1]:
                    cv2.line(frame, pts[j-1], pts[j], trail_colors_pk[i], int(2 + (j/len(pts))*6))

        # 绘制指尖
        for i, pos in enumerate(active_finger_pos):
            if pos:
                cv2.circle(frame, pos, 25, hand_colors[i], 3)
                cv2.circle(frame, pos, 15, (0, 255, 255), -1)

        if not game.game_over:
            game.update()
            game.check_collisions(trails[0], trails[1])
        
        game.draw(frame)
        cv2.imshow('Fruit Ninja Refactored', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('r'):
            game = PKGameEngine(assets, blade)
            for t in trails: t.clear()
            for s in smoothers: s.reset()

def run_standard_mode(cap, tracker, assets, blade, num_hands, dual):
    """标准模式 (单手/双手) 主循环"""
    game = GameEngine(assets, blade)
    smoothers = [FingerSmoother(method='ewma', alpha=0.4, adaptive=True) for _ in range(num_hands)]
    trails = [deque(maxlen=TRAIL_LENGTH) for _ in range(num_hands)]
    hand_colors = [(0, 255, 0), (0, 100, 255)]
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        lm_list, label_list = tracker.process(rgb)
        active_finger_pos = [None] * num_hands
        
        if lm_list:
            for i, (lm, label) in enumerate(zip(lm_list[:num_hands], label_list[:num_hands])):
                tip = lm[8]
                rx, ry = int(tip.x * WINDOW_WIDTH), int(tip.y * WINDOW_HEIGHT)
                sx, sy = smoothers[i].smooth(rx, ry)
                active_finger_pos[i] = (sx, sy)
                trails[i].append((sx, sy))
        
        # 轨迹渐消
        for i in range(num_hands):
            if active_finger_pos[i] is None and len(trails[i]) > 5:
                trails[i].popleft()

        # 绘制轨迹
        for i, trail in enumerate(trails):
            pts = list(trail)
            for j in range(1, len(pts)):
                if pts[j] and pts[j-1]:
                    cv2.line(frame, pts[j-1], pts[j], (100, 100, 255) if i==0 else (100, 255, 100), int(2 + (j/len(pts))*6))

        # 绘制指尖
        for i, pos in enumerate(active_finger_pos):
            if pos:
                cv2.circle(frame, pos, 25, hand_colors[i % 2], 3)
                cv2.circle(frame, pos, 15, (0, 255, 255), -1)

        # 双手模式提示
        if dual:
            active_count = sum(1 for p in active_finger_pos if p is not None)
            if active_count == 2: add_cn_text('双手已就绪 ✓', (WINDOW_WIDTH - 260, 20), font_size=26, color=(0, 255, 100))
            elif active_count == 1: add_cn_text('等待第二只手...', (WINDOW_WIDTH - 280, 20), font_size=26, color=(0, 180, 255))

        if not game.game_over:
            game.update()
            game.check_collisions(trails)
        
        game.draw(frame)
        cv2.imshow('Fruit Ninja Refactored', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('r'):
            game.reset()
            for t in trails: t.clear()
            for s in smoothers: s.reset()

if __name__ == "__main__":
    run_game()
