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
        self.date_index = 0
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

    def get_stock_data(self, code):
        try:
            date_stock_map = self.code_date_map[code]
            return date_stock_map[self.dates[self.date_index]]
        except KeyError:
            logging.info("Code: {}, not exists in Market on Date: {}.".format(code, self.dates[self.date_index]))
            raise ValueError

    def forward(self):
        if self.date_index < len(self.dates):
            self.date_index += 1
            return MarketStatus.Running
        else:
            return MarketStatus.NotRunning


class Position(object):

    def __init__(self, code, buy_price, amount):
        self.code = code
        self.amount = amount
        self.buy_price = buy_price
        self.cur_price = buy_price
        self.holding_price = self.cur_price * self.amount

    def add(self, buy_price, amount):
        self.buy_price = (self.amount * self.buy_price + amount * buy_price) / (self.amount + amount)
        self.cur_price = buy_price
        self.amount += amount
        self.holding_price = self.cur_price * self.amount

    def sub(self, sell_price, amount):
        self.cur_price = sell_price
        self.amount -= amount
        self.holding_price = self.cur_price * self.amount


class Trader(object):

    def __init__(self, market, cash=1000000.0):
        self.market = market
        self.cash = cash
        self.positions = []

    @property
    def holdings(self):
        holdings = 0
        for position in self.positions:
            holdings += position.holding_price
        return holdings

    def buy(self, code, amount):

        # Get current stock data.
        try:
            stock = self.market.get_stock_data(code)
        except ValueError:
            return

        # Check if position exists.
        if not self._exist_position(code):

            # Build position if possible.
            if self.cash > stock.close * amount:
                self.positions.append(Position(code, stock.close, amount))
            else:
                amount = int(math.floor(self.cash / stock.close))
                self.positions.append(Position(code, stock.close, amount))
        else:

            # Get position and update if possible.
            position = self._get_position(code)
            if self.cash > stock.close * amount:
                position.add(stock.close, amount)
            else:
                amount = int(math.floor(self.cash / stock.close))
                position.add(stock.close, amount)

        # Update cash and holding price.
        self.cash -= amount * stock.close

    def sell(self, code, amount):

        # Check if position exists.
        if not self._exist_position(code):
            return logging.info("Code: {}, not exists in Positions.".format(code))

        # Get current stock data.
        try:
            stock = self.market.get_stock_data(code)
        except ValueError:
            return

        # Sell position if possible.
        position = self._get_position(code)
        if amount > position.amount:
            amount = position.amount
        position.sub(stock.close, amount)

        if position.amount == 0:
            self.positions.remove(position)

        # Update cash and holding price.
        self.cash += amount * stock.close

    def hold(self, code, amount):
        pass

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

        logging.info("Cash: {} | Holdings: {}".format(trader.cash, trader.holdings))

        if market.forward() == MarketStatus.NotRunning:
            break


if __name__ == '__main__':
    main()
