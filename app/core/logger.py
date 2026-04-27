import logging
import sys

def setup_logger(name="fruit_ninja"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 终端输出
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    
    return logger

logger = setup_logger()
