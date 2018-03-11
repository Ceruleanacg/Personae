# coding=utf-8

import tushare as ts

from base.model.finance import Stock


class StockSpider(object):
    def __init__(self, code, start, end):
        self.code = code
        self.start = start
        self.end = end

    def crawl(self):
        stock_frame = ts.get_k_data(code=self.code, start=self.start, end=self.end, retry_count=30)
        for index in stock_frame.index:
            stock_series = stock_frame.loc[index]
            stock_dict = stock_series.to_dict()
            stock = Stock(**stock_dict)
            stock.save_if_need()
