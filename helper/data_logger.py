# coding=utf-8

import logging
import os

from datetime import datetime
from static import LOGS_DIR

DATETIME_NOW = datetime.now().strftime("%Y%m%d%H%M%S")


def generate_market_logger(model_name):

    market_log_path = '{}-{}-{}'.format(model_name, DATETIME_NOW, 'stock_market.log')

    market_logger = logging.getLogger('stock_market_logger')
    market_logger.setLevel(logging.DEBUG)
    market_log_sh = logging.StreamHandler()
    market_log_sh.setLevel(logging.WARNING)
    market_log_fh = logging.FileHandler(os.path.join(LOGS_DIR, market_log_path))
    market_log_fh.setLevel(logging.DEBUG)
    market_log_fh.setFormatter(logging.Formatter('[{}] {}'.format('%(asctime)s', '%(message)s')))
    market_logger.addHandler(market_log_sh)
    market_logger.addHandler(market_log_fh)

    return market_logger


def generate_algorithm_logger(model_name):

    algorithm_log_path = '{}-{}-{}'.format(model_name, DATETIME_NOW, 'algorithm.log')

    algorithm_logger = logging.getLogger('algorithm_logger')
    algorithm_logger.setLevel(logging.DEBUG)
    algorithm_log_sh = logging.StreamHandler()
    algorithm_log_sh.setLevel(logging.WARNING)
    algorithm_log_fh = logging.FileHandler(os.path.join(LOGS_DIR, algorithm_log_path))
    algorithm_log_fh.setLevel(logging.DEBUG)
    algorithm_log_fh.setFormatter(logging.Formatter('[{}] {}'.format('%(asctime)s', '%(message)s')))
    algorithm_logger.addHandler(algorithm_log_sh)
    algorithm_logger.addHandler(algorithm_log_fh)

    return algorithm_logger
