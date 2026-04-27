"""
[配置管理层 - 游戏说明书]
集中存放所有的游戏常量和外部资源路径。
修改这里的参数可以直接调整游戏难度、画面大小或资源位置，而无需修改核心逻辑代码。
"""
import os

# 游戏窗口设置
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
FPS = 30

# 水果配置
FRUIT_TYPES = [
    'banana', 'boluo', 'iceBanana', 'Mango', 
    'mugua', 'peach', 'pear', 'pineapple', 'strawberry', 'b1'
]

MULTI_FRUIT_TYPES = [
    'watermelon',  # 西瓜：8个切片
    'dragonfruit'  # 火龙果：8个切片
]

# 游戏平衡配置
SPAWN_INTERVAL_INITIAL = 25
SPAWN_INTERVAL_MIN = 40 # 实际上原代码里 max(40, interval - 1) 有点奇怪，通常应该是逐渐减小间隔
MAX_MISSED = 10
MAX_BOMBS_HIT = 3
MAX_OBJECTS_ON_SCREEN = 10
BOMB_SPAWN_CHANCE = 0.2
DEADLY_BOMB_CHANCE = 0.3
MULTI_FRUIT_CHANCE = 0.15

# 资源路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(BASE_DIR, 'assets')
SUCAI_DIR = os.path.join(ASSETS_DIR, 'sucai')
ZHADAN_DIR = os.path.join(ASSETS_DIR, 'zhadan')
DAOGUANG_DIR = os.path.join(ASSETS_DIR, 'daoguang')
TEXIAO_DIR = os.path.join(ASSETS_DIR, 'texiao')
YINXIAO_DIR = os.path.join(ASSETS_DIR, 'yinxiao')
TASK_PATH = os.path.join(ASSETS_DIR, 'hand_landmarker.task')

# 手势追踪配置
HAND_CONFIDENCE = 0.55
TRACK_CONFIDENCE = 0.65
PRESENCE_CONFIDENCE = 0.55
TRAIL_LENGTH = 15

# PK 模式配置
PK_X_DIVIDER = WINDOW_WIDTH // 2
