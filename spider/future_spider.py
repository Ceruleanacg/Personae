# coding=utf-8

import logging

from base.model.document import Future
from helper.args_parser import future_spider_parser


class FutureSpider(object):
    def __init__(self, code, start="2008-01-01", end="2018-01-01"):
        self.code = code
        self.start = start
        self.end = end

    def crawl(self):
        pass
        # logging.warning("Finish crawling code: {}, items count: {}".format(self.code, stock_frame.shape[0]))


if __name__ == '__main__':
    args = future_spider_parser.parse_args()
    for _code in args.codes:
        FutureSpider(_code, args.start, args.end).crawl()
