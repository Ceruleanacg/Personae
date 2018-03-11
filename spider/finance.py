# coding=utf-8

import tushare as ts

from base.model.finance import Stock
from helper.args_parser import spider_parser

if __name__ == '__main__':

    args = spider_parser.parse_args()

    code, start, end = args.code, args.start, args.end

    stock_frame = ts.get_k_data(code=code, start=start, end=end, retry_count=30)

    for index in stock_frame.index:
        stock_series = stock_frame.loc[index]
        stock_dict = stock_series.to_dict()
        stock = Stock(**stock_dict)
        stock.save()
