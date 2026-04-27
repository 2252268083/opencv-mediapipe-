class GameException(Exception):
    """游戏基础异常类"""
    pass

class AssetLoadError(GameException):
    """资源加载失败异常类"""
    pass

class HandTrackerError(GameException):
    """手部检测失败异常类"""
    pass
