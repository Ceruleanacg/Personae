# coding=utf-8

import logging
import random
import math

from base.model.finance import Stock
from enum import Enum


logging.basicConfig(level=logging.INFO)


class MarketStatus(Enum):
    Running = 0
    NotRunning = 1


class Market(object):

    def __init__(self, codes, start_date="2008-01-01", end_date="2018-01-01"):

        # Initialize vars.
        self.codes = codes
        self.dates = []
        self.code_date_map = dict()

        # Build code - date - stock map.
        for code in self.codes:
            stocks = Stock.get_k_data(code, start_date, end_date)
            date_stock_map = dict()
            for stock in stocks:
                date_stock_map[stock.date] = stock
                if stock.date not in self.dates:
                    self.dates.append(stock.date)
            self.code_date_map[code] = date_stock_map

        self.dates = sorted(self.dates)
        self.iter_dates = iter(self.dates)

        try:
            self.current_date = next(self.iter_dates)
        except StopIteration:
            raise ValueError("Initialize failed, dates is empty.")

    def get_cur_stock_data(self, code):
        date_stock_map = self.code_date_map[code]
        try:
            return date_stock_map[self.current_date]
        except KeyError:
            logging.info("Code: {}, not exists in Market on Date: {}.".format(code, self.current_date))
            raise ValueError

    def forward(self):
        try:
            self.current_date = next(self.iter_dates)
            return MarketStatus.Running
        except StopIteration:
            return MarketStatus.NotRunning


class Position(object):

    def __init__(self, code, buy_price, amount):
        self.code = code
        self.amount = amount
        self.buy_price = buy_price
        self.cur_price = buy_price
        self.holding_value = self.cur_price * self.amount

    def add(self, buy_price, amount):
        self.buy_price = (self.amount * self.buy_price + amount * buy_price) / (self.amount + amount)
        self.cur_price = buy_price
        self.amount += amount
        self.holding_value = self.cur_price * self.amount

    def sub(self, sell_price, amount):
        self.cur_price = sell_price
        self.amount -= amount
        self.holding_value = self.cur_price * self.amount


class Trader(object):

    def __init__(self, market, cash=100000.0):
        self.market = market
        self.cash = cash
        self.positions = []
        self.initial_cash = cash

    @property
    def holdings_value(self):
        holdings_value = 0
        for position in self.positions:
            holdings_value += position.holding_value
        return holdings_value

    @property
    def profits(self):
        return self.holdings_value + self.cash - self.initial_cash

    def buy(self, code, amount):

        # Get current stock data.
        try:
            stock = self.market.get_cur_stock_data(code)
        except ValueError:
            return logging.info("Buying {} failed, current date cannot trade.".format(code))

        # Check if amount is OK.
        amount = amount if self.cash > stock.close * amount else int(math.floor(self.cash / stock.close))

        # Check if position exists.
        if not self._exist_position(code):
            # Build position if possible.
            self.positions.append(Position(code, stock.close, amount))
        else:
            # Get position and update if possible.
            position = self._get_position(code)
            position.add(stock.close, amount)

        # Update cash and holding price.
        self.cash -= amount * stock.close

    def sell(self, code, amount):

        # Check if position exists.
        if not self._exist_position(code):
            return logging.info("Code: {}, not exists in Positions.".format(code))

        position = self._get_position(code)

        # Get current stock data.
        try:
            stock = self.market.get_cur_stock_data(code)
        except ValueError:
            return logging.info("Selling {} failed, current date cannot trade.".format(code))

        # Sell position if possible.
        amount = amount if amount < position.amount else position.amount
        position.sub(stock.close, amount)

        if position.amount == 0:
            self.positions.remove(position)

        # Update cash and holding price.
        self.cash += amount * stock.close

    @staticmethod
    def hold(code, _):
        pass

    def log_asset(self):
        logging.info("Cash: {0:.2f} | "
                     "Holdings: {1:.2f} | "
                     "Profits: {2:.2f}".format(self.cash, self.holdings_value, self.profits))

    def _exist_position(self, code):
        return True if len([position.code for position in self.positions if position.code == code]) else False

    def _get_position(self, code):
        return [position for position in self.positions if position.code == code][0]


def main():
    codes = ["600036", "601998"]
    market = Market(codes)
    trader = Trader(market)
    actions = [trader.buy, trader.sell, trader.hold]

    while True:
        code = random.choice(codes)
        action = random.choice(actions)
        action(code, random.randint(100, 200))
        trader.log_asset()
        if market.forward() == MarketStatus.NotRunning:
            break


if __name__ == '__main__':
    main()
