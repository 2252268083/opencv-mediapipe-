import os
import cv2
import pygame
from app.config import (
    SUCAI_DIR, ZHADAN_DIR, DAOGUANG_DIR, TEXIAO_DIR, YINXIAO_DIR,
    FRUIT_TYPES, MULTI_FRUIT_TYPES
)
from app.core.logger import logger

class AssetManager:
    def __init__(self):
        self.fruit_images = {}
        self.multi_fruit_images = {}
        self.bomb_images = {}
        self.blade_images = {}
        self.combo_images = {}
        self.juice_images = {}
        self.sound_effects = {}
        self.has_sound = False

    def load_all(self):
        self.fruit_images = self._load_fruit_images()
        self.multi_fruit_images = self._load_multi_fruit_images()
        self.bomb_images = self._load_bomb_images()
        self.blade_images = self._load_blade_images()
        self.combo_images = self._load_combo_images()
        self.juice_images = self._load_juice_images()
        self.sound_effects, self.has_sound = self._load_sound_effects()
        return self

    def _load_fruit_images(self):
        images = {}
        for name in FRUIT_TYPES:
            whole_path = os.path.join(SUCAI_DIR, f'{name}.png')
            left_path = os.path.join(SUCAI_DIR, 'bl.png' if name == 'b1' else f'{name}l.png')
            right_path = os.path.join(SUCAI_DIR, 'br.png' if name == 'b1' else f'{name}r.png')
            
            if os.path.exists(whole_path):
                images[name] = {
                    'whole': cv2.imread(whole_path, cv2.IMREAD_UNCHANGED),
                    'left': cv2.imread(left_path, cv2.IMREAD_UNCHANGED),
                    'right': cv2.imread(right_path, cv2.IMREAD_UNCHANGED)
                }
        return images

    def _load_multi_fruit_images(self):
        images = {}
        # Watermelon
        whole = os.path.join(SUCAI_DIR, 'watermelon.png')
        if os.path.exists(whole):
            whole_img = cv2.imread(whole, cv2.IMREAD_UNCHANGED)
            if whole_img is not None:
                whole_img = cv2.resize(whole_img, None, fx=0.5, fy=0.5)
                pieces = [cv2.resize(cv2.imread(os.path.join(SUCAI_DIR, f'watermelon{i}.png'), cv2.IMREAD_UNCHANGED), None, fx=0.5, fy=0.5) for i in range(1, 9)]
                images['watermelon'] = {'whole': whole_img, 'pieces': pieces, 'piece_count': 8}
        
        # Dragonfruit
        whole = os.path.join(SUCAI_DIR, 'all.png')
        if os.path.exists(whole):
            whole_img = cv2.imread(whole, cv2.IMREAD_UNCHANGED)
            if whole_img is not None:
                whole_img = cv2.resize(whole_img, None, fx=0.5, fy=0.5)
                pieces = [cv2.resize(cv2.imread(os.path.join(SUCAI_DIR, f'00{i}.png'), cv2.IMREAD_UNCHANGED), None, fx=0.5, fy=0.5) for i in range(1, 9)]
                images['dragonfruit'] = {'whole': whole_img, 'pieces': pieces, 'piece_count': 8}
        return images

    def _load_bomb_images(self):
        images = {}
        paths = {
            'bomb1': os.path.join(ZHADAN_DIR, 'boom1.png'),
            'bomb2': os.path.join(ZHADAN_DIR, 'boom2.png'),
            'explosion1': os.path.join(ZHADAN_DIR, 'zha01.png'),
            'explosion2': os.path.join(ZHADAN_DIR, 'zha02.png')
        }
        for key, path in paths.items():
            if os.path.exists(path):
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if 'explosion' in key: img = cv2.resize(img, None, fx=2.0, fy=2.0)
                images[key] = img
        return images

    def _load_blade_images(self):
        images = {}
        for name in ['dao1', 'dao2']:
            path = os.path.join(DAOGUANG_DIR, f'{name}.png')
            if os.path.exists(path): images[name] = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        return images

    def _load_combo_images(self):
        images = {}
        for i in range(1, 4):
            path = os.path.join(TEXIAO_DIR, f'combo{i}.png')
            if os.path.exists(path): images[f'combo{i}'] = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        return images

    def _load_juice_images(self):
        images = {}
        colors = ['orange', 'green', 'pink', 'red']
        for i, color in enumerate(colors):
            path = os.path.join(TEXIAO_DIR, f'guozhi{i+1}.png')
            if os.path.exists(path): images[color] = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        return images

    def _load_sound_effects(self):
        sounds = {}
        has_sound = False
        try:
            pygame.mixer.init()
            paths = {
                'slice': os.path.join(YINXIAO_DIR, 'qieshuiguoyinxiao.mp3'),
                'explosion': os.path.join(YINXIAO_DIR, 'baozhayinxiao.mp3')
            }
            for key, path in paths.items():
                if os.path.exists(path):
                    sounds[key] = pygame.mixer.Sound(path)
            has_sound = True
        except Exception as e:
            logger.warning(f"音效加载失败: {e}")
        return sounds, has_sound
