"""
[业务逻辑层 - 核心厨师/裁判员]
这是游戏的核心逻辑所在。它负责：
1. 游戏状态维护（得分、血量、游戏结束）
2. 物体生成逻辑（水果、炸弹的随机生成频率）
3. 碰撞检测（判断手指轨迹是否切中了水果）
4. 调用模型(Models)进行更新和绘制
5. PK 模式的分屏逻辑处理
"""
import random
import math
import cv2
import time
from collections import deque
from app.config import (
    WINDOW_WIDTH, WINDOW_HEIGHT, FRUIT_TYPES, MULTI_FRUIT_TYPES,
    MAX_MISSED, MAX_BOMBS_HIT, MAX_OBJECTS_ON_SCREEN, BOMB_SPAWN_CHANCE,
    DEADLY_BOMB_CHANCE, MULTI_FRUIT_CHANCE
)
from app.models.game_models import Fruit, MultiFruit, Bomb, ComboEffect, SlashEffect, JuiceEffect
from app.core.utils import get_juice_color_for_fruit, add_cn_text, flush_cn_texts, overlay_image
from app.core.logger import logger

class GameEngine:
    """标准单人/双手模式游戏引擎"""
    def __init__(self, asset_manager, selected_blade='dao1'):
        self.asset_manager = asset_manager
        self.selected_blade = selected_blade
        self.reset()
        
    def reset(self):
        self.fruits, self.multi_fruits, self.bombs = [], [], []
        self.slash_effects, self.combo_effects, self.juice_effects = [], [], []
        self.score, self.missed, self.bombs_hit = 0, 0, 0
        self.combo_count, self.max_combo, self.last_combo_milestone = 0, 0, 0
        self.spawn_timer, self.spawn_interval = 0, 25
        self.game_over, self.game_over_reason = False, ""
        self.last_spawn_type, self.consecutive_bombs = None, 0
        self.spawn_single_object()
        
    def spawn_single_object(self):
        if self.consecutive_bombs >= 2:
            self._spawn_fruit(); self.last_spawn_type, self.consecutive_bombs = 'fruit', 0
            return
        
        should_spawn_bomb = random.random() < BOMB_SPAWN_CHANCE
        if self.last_spawn_type == 'bomb' and should_spawn_bomb:
            if random.random() < 0.5: should_spawn_bomb = False
            
        if should_spawn_bomb:
            bomb_type = 'deadly' if random.random() < DEADLY_BOMB_CHANCE else 'normal'
            self.bombs.append(Bomb(bomb_type, 
                                 self.asset_manager.bomb_images.get('bomb2' if bomb_type == 'deadly' else 'bomb1'), 
                                 self.asset_manager.bomb_images))
            self.last_spawn_type, self.consecutive_bombs = 'bomb', self.consecutive_bombs + 1
        else:
            self._spawn_fruit(); self.last_spawn_type, self.consecutive_bombs = 'fruit', 0
    
    def _spawn_fruit(self):
        if random.random() < MULTI_FRUIT_CHANCE and self.asset_manager.multi_fruit_images:
            name = random.choice(list(self.asset_manager.multi_fruit_images.keys()))
            self.multi_fruits.append(MultiFruit(name, self.asset_manager.multi_fruit_images[name]))
        else:
            name = random.choice(FRUIT_TYPES)
            self.fruits.append(Fruit(name, self.asset_manager.fruit_images.get(name)))
    
    def update(self):
        if self.game_over: return
        active_objects = sum(1 for f in self.fruits if not f.is_cut) + \
                         sum(1 for f in self.multi_fruits if not f.is_cut) + \
                         sum(1 for b in self.bombs if not b.is_exploded)
        
        if active_objects < MAX_OBJECTS_ON_SCREEN:
            self.spawn_timer += 1
            if self.spawn_timer >= self.spawn_interval:
                self.spawn_single_object(); self.spawn_timer = 0
                self.spawn_interval = max(40, self.spawn_interval - 1)
        
        for fruit in self.fruits[:]:
            fruit.update()
            if fruit.is_out_of_screen():
                if not fruit.is_cut and fruit.has_entered_screen:
                    self.missed += 1; self.combo_count = 0; self.last_combo_milestone = 0
                    if self.missed >= MAX_MISSED: self.game_over, self.game_over_reason = True, "漏掉太多水果！"
                self.fruits.remove(fruit)
        
        for fruit in self.multi_fruits[:]:
            fruit.update()
            if fruit.is_out_of_screen():
                if not fruit.is_cut and fruit.has_entered_screen:
                    self.missed += 1; self.combo_count = 0; self.last_combo_milestone = 0
                    if self.missed >= MAX_MISSED: self.game_over, self.game_over_reason = True, "漏掉太多水果！"
                self.multi_fruits.remove(fruit)
        
        for bomb in self.bombs[:]:
            bomb.update()
            if bomb.is_out_of_screen(): self.bombs.remove(bomb)
        
        for effect in self.slash_effects + self.combo_effects + self.juice_effects:
            effect.update()
        self.slash_effects = [e for e in self.slash_effects if not e.is_finished()]
        self.combo_effects = [e for e in self.combo_effects if not e.is_finished()]
        self.juice_effects = [e for e in self.juice_effects if not e.is_finished()]
                
    def check_collisions(self, trail_list: list):
        """trail_list: 轨迹队列列表，单手=[trail]，双手=[trail1, trail2]"""
        hit = False
        for trail in trail_list:
            if len(trail) < 2: continue
            pts = list(trail)[-5:]
            cut_angle = math.degrees(math.atan2(pts[-1][1]-pts[0][1], pts[-1][0]-pts[0][0])) if len(pts)>=2 else 0

            for fruit in self.fruits + self.multi_fruits:
                if not fruit.is_cut and fruit.check_collision(trail):
                    fruit.cut(cut_angle)
                    self.score += 20 if isinstance(fruit, MultiFruit) else 10
                    hit = True
                    self.slash_effects.append(SlashEffect(fruit.x, fruit.y, cut_angle))
                    color = get_juice_color_for_fruit(fruit.fruit_type, self.asset_manager.juice_images)
                    if color: self.juice_effects.append(JuiceEffect(fruit.x, fruit.y, self.asset_manager.juice_images[color]))
                    if self.asset_manager.has_sound and 'slice' in self.asset_manager.sound_effects: self.asset_manager.sound_effects['slice'].play()
            
            for bomb in self.bombs:
                if not bomb.is_exploded and bomb.check_collision(trail):
                    bomb.explode(); self.slash_effects.append(SlashEffect(bomb.x, bomb.y, cut_angle))
                    if self.asset_manager.has_sound and 'explosion' in self.asset_manager.sound_effects: self.asset_manager.sound_effects['explosion'].play()
                    self.combo_count, self.last_combo_milestone = 0, 0
                    if bomb.bomb_type == 'deadly': self.game_over, self.game_over_reason = True, "切到致命炸弹！"
                    else:
                        self.score, self.bombs_hit = max(0, self.score - 20), self.bombs_hit + 1
                        if self.bombs_hit >= MAX_BOMBS_HIT: self.game_over, self.game_over_reason = True, "切到太多炸弹！"
        
        if hit:
            self.combo_count += 1; self.max_combo = max(self.max_combo, self.combo_count)
            for m in [10, 15, 20]:
                if self.combo_count >= m and self.last_combo_milestone < m:
                    self.last_combo_milestone = m; self.combo_effects.append(ComboEffect(m, self.asset_manager.combo_images)); break
                
    def draw(self, frame):
        for obj in self.fruits + self.multi_fruits + self.bombs + self.juice_effects: obj.draw(frame)
        blade_img = self.asset_manager.blade_images.get(self.selected_blade)
        for e in self.slash_effects: e.draw(frame, blade_img)
        for e in self.combo_effects: e.draw(frame)
        self._draw_ui(frame)
        flush_cn_texts(frame)
            
    def _draw_ui(self, frame):
        add_cn_text(f'得分: {self.score}', (20, 45), font_size=38, color=(255,255,255), bg_color=(0,180,0))
        add_cn_text(f'漏掉: {self.missed}/{MAX_MISSED}', (20, 110), font_size=32, color=(255,255,255), bg_color=(0,0,200))
        add_cn_text(f'炸弹: {self.bombs_hit}/{MAX_BOMBS_HIT}', (20, 170), font_size=32, color=(255,255,255), bg_color=(0,100,255))
        if self.combo_count > 0:
            combo_bg = (0, 165, 255) if self.combo_count >= 10 else (80, 80, 80)
            add_cn_text(f'连击: ×{self.combo_count}', (20, 230), font_size=36, color=(255,255,255), bg_color=combo_bg)

        if self.game_over:
            ov = frame.copy()
            cv2.rectangle(ov, (0, 0), (WINDOW_WIDTH, WINDOW_HEIGHT), (0, 0, 0), -1)
            cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
            cx, cy = WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2
            add_cn_text('游戏结束！', (cx - 150, cy - 120), font_size=72, color=(255, 60, 60))
            if self.game_over_reason: add_cn_text(self.game_over_reason, (cx - 200, cy - 30), font_size=40, color=(255, 200, 0))
            add_cn_text(f'最终得分: {self.score}', (cx - 180, cy + 40), font_size=48, color=(0, 255, 100))
            add_cn_text('按 R 重新开始   按 Q 退出', (cx - 220, cy + 170), font_size=32, color=(200, 200, 200))

class PKHalf:
    """管理一个玩家的半屏游戏状态"""
    def __init__(self, player_id, x_min, x_max, asset_manager, blade):
        self.player_id, self.x_min, self.x_max = player_id, x_min, x_max
        self.asset_manager, self.blade = asset_manager, blade
        self.score, self.fruits, self.multi_fruits, self.bombs = 0, [], [], []
        self.slash_fx, self.juice_fx = [], []
        self.spawn_timer, self.spawn_interval = 0, 30
        self.last_type, self.consec_bombs = None, 0
        self._spawn()

    def _spawn(self):
        if self.consec_bombs >= 2:
            self._spawn_fruit(); self.last_type = 'fruit'; self.consec_bombs = 0; return
        is_bomb = random.random() < 0.18
        if self.last_type == 'bomb' and is_bomb and random.random() < 0.5: is_bomb = False
        if is_bomb:
            t = 'deadly' if random.random() < 0.25 else 'normal'
            b = Bomb(t, self.asset_manager.bomb_images.get('bomb2' if t == 'deadly' else 'bomb1'), self.asset_manager.bomb_images)
            b.x = random.randint(self.x_min + 60, self.x_max - 60)
            self.bombs.append(b); self.last_type = 'bomb'; self.consec_bombs += 1
        else:
            self._spawn_fruit(); self.last_type = 'fruit'; self.consec_bombs = 0

    def _spawn_fruit(self):
        if random.random() < 0.15 and self.asset_manager.multi_fruit_images:
            name = random.choice(list(self.asset_manager.multi_fruit_images.keys()))
            f = MultiFruit(name, self.asset_manager.multi_fruit_images[name])
        else:
            name = random.choice(FRUIT_TYPES)
            f = Fruit(name, self.asset_manager.fruit_images.get(name))
        f.x = random.randint(self.x_min + 60, self.x_max - 60)
        if isinstance(f, MultiFruit): self.multi_fruits.append(f)
        else: self.fruits.append(f)

    def update(self):
        active = sum(1 for f in self.fruits if not f.is_cut) + \
                 sum(1 for f in self.multi_fruits if not f.is_cut) + \
                 sum(1 for b in self.bombs if not b.is_exploded)
        if active < 6:
            self.spawn_timer += 1
            if self.spawn_timer >= self.spawn_interval:
                self._spawn(); self.spawn_timer = 0
                self.spawn_interval = max(35, self.spawn_interval - 1)
        for lst in [self.fruits, self.multi_fruits]:
            for obj in lst[:]:
                obj.update()
                if obj.out() if hasattr(obj, 'out') else obj.is_out_of_screen(): lst.remove(obj)
        for b in self.bombs[:]:
            b.update()
            if b.is_out_of_screen(): self.bombs.remove(b)
        for lst in [self.slash_fx, self.juice_fx]:
            for e in lst[:]:
                e.update()
                if e.is_finished(): lst.remove(e)

    def check_collisions(self, trail):
        if len(trail) < 2: return
        pts = list(trail)[-5:]
        cut_angle = math.degrees(math.atan2(pts[-1][1]-pts[0][1], pts[-1][0]-pts[0][0])) if len(pts)>=2 else 0
        for f in self.fruits + self.multi_fruits:
            if not f.is_cut and f.check_collision(trail):
                if any(self.x_min <= p[0] <= self.x_max for p in list(trail)[-5:] if p):
                    f.cut(cut_angle); self.score += 20 if isinstance(f, MultiFruit) else 10
                    self.slash_fx.append(SlashEffect(f.x, f.y, cut_angle))
                    c = get_juice_color_for_fruit(f.fruit_type, self.asset_manager.juice_images)
                    if c: self.juice_fx.append(JuiceEffect(f.x, f.y, self.asset_manager.juice_images[c]))
                    if self.asset_manager.has_sound and 'slice' in self.asset_manager.sound_effects: self.asset_manager.sound_effects['slice'].play()
        for b in self.bombs:
            if not b.is_exploded and b.check_collision(trail):
                if any(self.x_min <= p[0] <= self.x_max for p in list(trail)[-5:] if p):
                    b.explode(); self.slash_fx.append(SlashEffect(b.x, b.y, cut_angle))
                    if self.asset_manager.has_sound and 'explosion' in self.asset_manager.sound_effects: self.asset_manager.sound_effects['explosion'].play()
                    if b.bomb_type == 'normal': self.score = max(0, self.score - 20)

    def draw(self, frame):
        for obj in self.fruits + self.multi_fruits + self.bombs + self.juice_fx: obj.draw(frame)
        blade_img = self.asset_manager.blade_images.get(self.blade)
        for e in self.slash_fx: e.draw(frame, blade_img)

class PKGameEngine:
    """双人PK对战游戏管理器（30秒倒计时）"""
    TOTAL_TIME = 30
    def __init__(self, asset_manager, selected_blade='dao1'):
        self.asset_manager = asset_manager
        half = WINDOW_WIDTH // 2
        self.p1 = PKHalf(1, 0, half, asset_manager, selected_blade)
        self.p2 = PKHalf(2, half, WINDOW_WIDTH, asset_manager, selected_blade)
        self.start_time, self.game_over, self.winner = time.time(), False, None

    def time_left(self) -> float: return max(0.0, self.TOTAL_TIME - (time.time() - self.start_time))

    def update(self):
        if self.game_over: return
        self.p1.update(); self.p2.update()
        if self.time_left() <= 0:
            self.game_over = True
            if self.p1.score > self.p2.score: self.winner = 'p1'
            elif self.p2.score > self.p1.score: self.winner = 'p2'
            else: self.winner = 'draw'

    def check_collisions(self, trail_p1, trail_p2):
        if not self.game_over:
            self.p1.check_collisions(trail_p1)
            self.p2.check_collisions(trail_p2)

    def draw(self, frame):
        self.p1.draw(frame); self.p2.draw(frame)
        half, tl = WINDOW_WIDTH // 2, self.time_left()
        cv2.line(frame, (half, 0), (half, WINDOW_HEIGHT), (255, 255, 255), 2)
        timer_color = (0, 60, 255) if tl <= 10 else (0, 220, 255)
        add_cn_text(f'{int(math.ceil(tl))}秒', (half - 55, 12), font_size=52, color=timer_color, bg_color=(30, 30, 30), padding=10)
        add_cn_text('玩家一', (18, 12), font_size=30, color=(255,255,255), bg_color=(200,80,0))
        add_cn_text(f'{self.p1.score} 分', (18, 58), font_size=42, color=(255,255,100), bg_color=(40,40,40))
        add_cn_text('玩家二', (half+18, 12), font_size=30, color=(255,255,255), bg_color=(0,120,220))
        add_cn_text(f'{self.p2.score} 分', (half+18, 58), font_size=42, color=(100,255,255), bg_color=(40,40,40))
        if not self.game_over:
            if self.p1.score > self.p2.score: add_cn_text('领先 ▶', (half - 220, 70), font_size=26, color=(255, 200, 0))
            elif self.p2.score > self.p1.score: add_cn_text('◀ 领先', (half + 20, 70), font_size=26, color=(255, 200, 0))
        if self.game_over:
            ov = frame.copy()
            cv2.rectangle(ov, (0, 0), (WINDOW_WIDTH, WINDOW_HEIGHT), (0, 0, 0), -1)
            cv2.addWeighted(ov, 0.60, frame, 0.40, 0, frame)
            cy = WINDOW_HEIGHT // 2
            if self.winner == 'p1': add_cn_text('🏆  玩家一获胜！', (half - 300, cy - 130), font_size=68, color=(255, 200, 0))
            elif self.winner == 'p2': add_cn_text('🏆  玩家二获胜！', (half - 300, cy - 130), font_size=68, color=(100, 200, 255))
            else: add_cn_text('势均力敌，平局！', (half - 290, cy - 130), font_size=68, color=(200, 200, 200))
            add_cn_text(f'玩家一  {self.p1.score} 分', (half - 340, cy - 30), font_size=46, color=(255, 180, 80))
            add_cn_text('VS', (half - 36, cy - 22), font_size=40, color=(200, 200, 200))
            add_cn_text(f'{self.p2.score} 分  玩家二', (half + 60, cy - 30), font_size=46, color=(80, 200, 255))
            add_cn_text('按 R 再来一局    按 Q 退出', (half - 290, cy + 60), font_size=34, color=(180, 180, 180))
        flush_cn_texts(frame)
