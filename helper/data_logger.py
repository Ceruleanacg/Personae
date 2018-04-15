# coding=utf-8

import logging
import os

from datetime import datetime
from static import LOGS_DIR

DATETIME_NOW = datetime.now().strftime("%Y%m%d%H%M%S")

STOCK_MARKET_LOG_PATH = '{}-{}'.format(DATETIME_NOW, 'stock_market.log')

stock_market_logger = logging.getLogger('stock_market_logger')
stock_market_logger.setLevel(logging.DEBUG)
stock_market_log_sh = logging.StreamHandler()
stock_market_log_sh.setLevel(logging.WARNING)
stock_market_log_fh = logging.FileHandler(os.path.join(LOGS_DIR, STOCK_MARKET_LOG_PATH))
stock_market_log_fh.setLevel(logging.DEBUG)
stock_market_log_fh.setFormatter(logging.Formatter('[{}] {}'.format('%(asctime)s', '%(message)s')))
stock_market_logger.addHandler(stock_market_log_sh)
stock_market_logger.addHandler(stock_market_log_fh)

ALGORITHM_LOG_PATH = '{}-{}'.format(DATETIME_NOW, 'algorithm.log')

algorithm_logger = logging.getLogger('algorithm_logger')
algorithm_logger.setLevel(logging.DEBUG)
algorithm_log_sh = logging.StreamHandler()
algorithm_log_sh.setLevel(logging.WARNING)
algorithm_log_fh = logging.FileHandler(os.path.join(LOGS_DIR, ALGORITHM_LOG_PATH))
algorithm_log_fh.setLevel(logging.DEBUG)
algorithm_log_fh.setFormatter(logging.Formatter('[{}] {}'.format('%(asctime)s', '%(message)s')))
algorithm_logger.addHandler(algorithm_log_sh)
algorithm_logger.addHandler(algorithm_log_fh)
