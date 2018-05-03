# coding=utf-8

import os

LOGS_DIR = os.path.join(os.path.dirname(__file__), 'logs')

if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)
