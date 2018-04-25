# coding=utf-8

import pandas as pd

from base.model.document import Stock


def generate_sample_data():

    dates = pd.date_range(start="2008-01-01", end="2008-01-30")

    for index, date in enumerate(dates):
        stock = Stock()
        stock.code = "T9999"
        stock.date = date
        stock.open = index
        stock.high = index + 1
        stock.low = index - 0.5
        stock.close = index + 1
        stock.volume = 100
        stock.save_if_need()

    for index, date in enumerate(dates[::-1]):
        stock = Stock()
        stock.code = "T9998"
        stock.date = date
        stock.open = index
        stock.high = index + 1
        stock.low = index - 0.5
        stock.close = index + 1
        stock.volume = 100
        stock.save_if_need()


if __name__ == '__main__':
    generate_sample_data()
