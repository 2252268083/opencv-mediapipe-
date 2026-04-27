"""
[数据模型层 - 模具/演员原型]
定义了游戏中的各种对象及其基本行为。
每个水果、炸弹或特效都有自己的坐标、速度、形状和绘制方法。
它不处理游戏全局规则，只关注“我这个物体”该如何运动和显示。
"""
import random
import math
import cv2
import numpy as np
from app.core.utils import overlay_image
from app.config import WINDOW_WIDTH, WINDOW_HEIGHT

class Fruit:
    """普通水果类 - 切开后分成两半"""
    def __init__(self, fruit_type, images):
        # 随机生成初始位置（屏幕底部，随机水平位置）
        self.x = random.randint(100, WINDOW_WIDTH - 100)
        self.y = WINDOW_HEIGHT + 50
        self.fruit_type = fruit_type
        self.images = images
        
        # 计算半径，用于碰撞检测
        if self.images and self.images['whole'] is not None:
            h, w = self.images['whole'].shape[:2]
            self.width, self.height = w, h
            self.radius = max(w, h) // 2
        else:
            self.radius, self.width, self.height = 50, 100, 100
            
        # 初始速度和重力
        self.velocity_x = random.uniform(-1, 1)  # 随机水平速度
        self.velocity_y = random.uniform(-18, -14) # 向上飞的初速度
        self.gravity = 0.3 # 重力加速度
        self.rotation = random.uniform(0, 360) # 初始旋转角度
        self.rotation_speed = random.uniform(-5, 5) # 旋转速度
        self.is_cut = False
        self.cut_pieces = []
        self.has_entered_screen = False # 是否进入过屏幕可视区域
        
    def update(self):
        """每一帧更新水果的位置"""
        if not self.is_cut:
            # 物理模拟：v = v0 + a*t, s = s0 + v*t
            self.velocity_y += self.gravity
            self.x += self.velocity_x
            self.y += self.velocity_y
            self.rotation += self.rotation_speed
            # 标记是否曾进入屏幕（用于判断是否算作“漏掉”的水果）
            if not self.has_entered_screen and 0 <= self.y <= WINDOW_HEIGHT - 50:
                self.has_entered_screen = True
        else:
            # 水果被切开后的碎片运动逻辑
            for piece in self.cut_pieces:
                piece['vy'] += self.gravity * 2.0 # 碎片掉落得更快一些
                piece['x'] += piece['vx']; piece['y'] += piece['vy']
                piece['rotation'] += piece['rotation_speed']; piece['alpha'] -= 3 # 逐渐变透明
                
    def draw(self, frame):
        """将水果绘制到画面上"""
        if not self.is_cut:
            if self.images and self.images['whole'] is not None:
                overlay_image(frame, self.images['whole'], int(self.x), int(self.y), self.rotation, 1.0)
            else:
                # 备用绘制：如果图片加载失败，画个圆圈
                cv2.circle(frame, (int(self.x), int(self.y)), 40, (0, 255, 255), -1)
        else:
            # 绘制碎片
            for piece in self.cut_pieces:
                if piece['alpha'] > 0:
                    overlay_image(frame, piece['image'], int(piece['x']), int(piece['y']), 
                                piece['rotation'], piece['alpha'] / 255.0)
                    
    def cut(self, cut_angle=0):
        """执行切开动作，生成两个碎片"""
        self.is_cut = True
        if not self.images: return
        for side, dx in [('left', -1), ('right', 1)]:
            if self.images[side] is not None:
                # 碎片会向左右飞开
                vx = random.uniform(5, 8) * dx
                self.cut_pieces.append({
                    'x': self.x, 'y': self.y, 'image': self.images[side],
                    'vx': vx, 'vy': random.uniform(2, 4), 'rotation': self.rotation,
                    'rotation_speed': random.uniform(4, 8) * dx,
                    'alpha': 255
                })
            
    def is_out_of_screen(self):
        if not self.is_cut: return self.y > WINDOW_HEIGHT + 150
        return all(piece['alpha'] <= 0 or piece['y'] > WINDOW_HEIGHT + 150 for piece in self.cut_pieces)
            
    def check_collision(self, trail):
        if self.is_cut or len(trail) < 2: return False
        for point in list(trail)[-15:]:
            if point and math.hypot(point[0] - self.x, point[1] - self.y) < self.radius * 0.8:
                return True
        return False

class MultiFruit(Fruit):
    def __init__(self, fruit_type, images):
        super().__init__(fruit_type, images)
        if self.images and self.images['whole'] is not None:
            h, w = self.images['whole'].shape[:2]
            self.width, self.height = w, h
            self.radius = max(w, h) // 2
        
    def cut(self, cut_angle=0):
        self.is_cut = True
        if not self.images or not self.images.get('pieces'): return
        piece_count = self.images['piece_count']
        angle_step = 360 / piece_count
        for i, piece_img in enumerate(self.images['pieces']):
            angle_rad = math.radians(i * angle_step)
            speed = random.uniform(6, 10)
            self.cut_pieces.append({
                'x': self.x, 'y': self.y, 'image': piece_img,
                'vx': math.cos(angle_rad) * speed, 'vy': math.sin(angle_rad) * speed - 2,
                'rotation': self.rotation + random.uniform(-30, 30),
                'rotation_speed': random.uniform(-10, 10), 'alpha': 255
            })

class Bomb:
    def __init__(self, bomb_type, image, explosion_images):
        self.x = random.randint(100, WINDOW_WIDTH - 100)
        self.y = WINDOW_HEIGHT + 50
        self.bomb_type = bomb_type
        self.image = image
        self.explosion_images = explosion_images
        
        if self.image is not None:
            h, w = self.image.shape[:2]
            self.width, self.height, self.radius = w, h, max(w, h) // 2
        else:
            self.radius, self.width, self.height = 50, 100, 100
            
        self.velocity_x, self.velocity_y = random.uniform(-1, 1), random.uniform(-18, -14)
        self.gravity, self.rotation = 0.3, random.uniform(0, 360)
        self.rotation_speed = random.uniform(-5, 5)
        self.is_exploded, self.has_entered_screen = False, False
        self.explosion_frame, self.explosion_max_frames = 0, 20
        
    def update(self):
        if not self.is_exploded:
            self.velocity_y += self.gravity
            self.x += self.velocity_x; self.y += self.velocity_y
            self.rotation += self.rotation_speed
            if not self.has_entered_screen and 0 <= self.y <= WINDOW_HEIGHT - 50:
                self.has_entered_screen = True
        else:
            self.explosion_frame += 1
                
    def draw(self, frame):
        if not self.is_exploded:
            if self.image is not None:
                overlay_image(frame, self.image, int(self.x), int(self.y), self.rotation, 1.0)
            else:
                color = (0, 0, 255) if self.bomb_type == 'deadly' else (0, 0, 0)
                cv2.circle(frame, (int(self.x), int(self.y)), 40, color, -1)
        elif self.explosion_frame < self.explosion_max_frames:
            key = 'explosion1' if self.explosion_frame % 4 < 2 else 'explosion2'
            boom_img = self.explosion_images.get(key)
            if boom_img is not None:
                alpha = 1.0 - (self.explosion_frame / self.explosion_max_frames)
                overlay_image(frame, boom_img, int(self.x), int(self.y), 0, alpha)
                    
    def explode(self):
        self.is_exploded, self.explosion_frame = True, 0
        
    def is_out_of_screen(self):
        if not self.is_exploded: return self.y > WINDOW_HEIGHT + 150
        return self.explosion_frame >= self.explosion_max_frames
            
    def check_collision(self, trail):
        if self.is_exploded or len(trail) < 2: return False
        for point in list(trail)[-15:]:
            if point and math.hypot(point[0] - self.x, point[1] - self.y) < self.radius * 0.8:
                return True
        return False

class ComboEffect:
    def __init__(self, combo_count, combo_images):
        self.combo_count = combo_count
        self.x, self.y = WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 100
        self.alpha, self.scale, self.frame, self.duration = 0, 0.5, 0, 60
        self.combo_images = combo_images
        self.key = ('combo3' if combo_count >= 20 else 'combo2' if combo_count >= 15 else 'combo1' if combo_count >= 10 else None)
    
    def update(self):
        self.frame += 1
        if self.frame < 15:
            self.alpha, self.scale = int(255 * (self.frame / 15)), 0.5 + (self.frame / 15) * 0.5
        elif self.frame < 45:
            self.alpha, self.scale = 255, 1.0
        else:
            fade = (self.frame - 45) / 15
            self.alpha, self.scale = int(255 * (1 - fade)), 1.0 + fade * 0.2
    
    def is_finished(self): return self.frame >= self.duration
    
    def draw(self, frame):
        if self.key and self.key in self.combo_images:
            img = self.combo_images[self.key]
            if img is not None:
                h, w = img.shape[:2]
                scaled = cv2.resize(img, (int(w * self.scale), int(h * self.scale)))
                overlay_image(frame, scaled, int(self.x), int(self.y), 0, self.alpha / 255.0)

class SlashEffect:
    def __init__(self, x, y, angle):
        self.x, self.y, self.angle = x, y, angle
        self.alpha, self.scale, self.duration, self.frame = 255, 1.2, 20, 0
        
    def update(self):
        self.frame += 1
        self.alpha = int(255 * (1 - self.frame / self.duration))
        self.scale = 1.2 + (self.frame / self.duration) * 0.3
        
    def is_finished(self): return self.frame >= self.duration
    
    def draw(self, frame, blade_img):
        if blade_img is not None:
            h, w = blade_img.shape[:2]
            scaled = cv2.resize(blade_img, (int(w * self.scale), int(h * self.scale)))
            overlay_image(frame, scaled, int(self.x), int(self.y), self.angle, self.alpha / 255.0)

class JuiceEffect:
    def __init__(self, x, y, juice_img):
        self.x, self.y, self.img = x, y, juice_img
        self.alpha, self.scale, self.duration, self.frame = 255, 1.2, 20, 0
        
    def update(self):
        self.frame += 1
        if self.frame < self.duration * 0.6: self.alpha = 255
        else: self.alpha = int(255 * (1 - (self.frame - self.duration * 0.6) / (self.duration * 0.4)))
        self.scale = 1.2 + (self.frame / self.duration) * 0.8
        
    def is_finished(self): return self.frame >= self.duration
    
    def draw(self, frame):
        if self.img is not None:
            h, w = self.img.shape[:2]
            scaled = cv2.resize(self.img, (int(w * self.scale), int(h * self.scale)))
            overlay_image(frame, scaled, int(self.x), int(self.y), 0, min(255, max(0, self.alpha)) / 255.0)
