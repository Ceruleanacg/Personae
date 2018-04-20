# coding=utf-8

import rqalpha
import logging

from rqalpha.api import *
from datetime import datetime
from base.model.document import Future
from helper.args_parser import future_spider_parser


def init(context):
    context.has_save_data = False


def before_trading(context):
    if not context.has_save_data:
        print(all_instruments(type='Future'))
        for code in config['args'].codes:
            items = history_bars(code, 200000, '1d')
            for item in items:
                future = Future()
                future.code = code
                future.date = datetime.strptime(str(item[0]), "%Y%m%d%H%M%S")
                future.open, future.high, future.low, future.close = item[1], item[2], item[3], item[4]
                future.volume = item[5]
                future.save_if_need()
            logging.warning("Finish crawling code: {}, items count: {}".format(code, len(items)))
    context.has_save_data = True


def handle_bar(context, bar_dict):
    pass


def after_trading(context):
    pass


if __name__ == '__main__':

    args = future_spider_parser.parse_args()

    config = {
        "base": {
            "start_date": "2018-01-01",
            "end_date": "20018-01-02",
            "benchmark": "AU88",
            "accounts": {
                "future": 100000
            }
        },
        "extra": {
            "log_level": "warning",
        },
        "args": args,
    }

    rqalpha.run_func(**globals())
