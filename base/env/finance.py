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
        self.trader = Trader(self)
        self.data_dim = None
        self.code_date_map = dict()

        if not len(self.codes):
            raise ValueError("Initialize, codes cannot be empty.")

        # Build code - date - stock map.
        for code in self.codes:
            stocks = Stock.get_k_data(code, start_date, end_date)
            if not self.data_dim:
                self.data_dim = self._get_data_dim(stocks.first())
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

    def reset(self):
        self.trader.reset()
        self.iter_dates = iter(self.dates)

    def forward(self, action_sheet):

        # Check trader.
        if not self.trader:
            raise ValueError("Trader cannot be None.")

        # Here, action_sheet is like: [0, 1]
        code, action = self.trader.codes[action_sheet[0]], self.trader.actions[action_sheet[1]]

        # Get Stock for current date with code.
        try:
            stock = self.get_cur_stock_data(code)
            action(stock, 100)
        except ValueError:
            logging.info("Buying {} failed, current date cannot trade.".format(code))

        try:
            self.current_date = next(self.iter_dates)
            stock_next = self.get_cur_stock_data(code)
            return stock_next, self.trader.profits, MarketStatus.Running, "Running."
        except StopIteration:
            return None, self.trader.profits, MarketStatus.NotRunning, "Not Running."

    @staticmethod
    def _get_data_dim(stock):
        stock = stock.to_mongo()
        stock.pop('date')
        stock.pop('_id')
        stock.pop('code')
        return len(stock.keys())


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
        self.cash = cash
        self.codes = market.codes
        self.market = market
        self.positions = []
        self.initial_cash = cash
        self.actions = [self.buy, self.sell, self.hold]

    @property
    def codes_count(self):
        return len(self.codes)

    @property
    def action_space(self):
        return len(self.actions)

    @property
    def holdings_value(self):
        holdings_value = 0
        for position in self.positions:
            holdings_value += position.holding_value
        return holdings_value

    @property
    def profits(self):
        return self.holdings_value + self.cash - self.initial_cash

    def buy(self, stock, amount):

        # Check if amount is OK.
        amount = amount if self.cash > stock.close * amount else int(math.floor(self.cash / stock.close))

        # Check if position exists.
        if not self._exist_position(stock.code):
            # Build position if possible.
            self.positions.append(Position(stock.code, stock.close, amount))
        else:
            # Get position and update if possible.
            position = self._get_position(stock.code)
            position.add(stock.close, amount)

        # Update cash and holding price.
        self.cash -= amount * stock.close

    def sell(self, stock, amount):

        # Check if position exists.
        if not self._exist_position(stock.code):
            return logging.info("Code: {}, not exists in Positions.".format(stock.code))

        position = self._get_position(stock.code)

        # Get current stock data.
        try:
            stock = self.market.get_cur_stock_data(stock.code)
        except ValueError:
            return logging.info("Selling {} failed, current date cannot trade.".format(stock.code))

        # Sell position if possible.
        amount = amount if amount < position.amount else position.amount
        position.sub(stock.close, amount)

        if position.amount == 0:
            self.positions.remove(position)

        # Update cash and holding price.
        self.cash += amount * stock.close

    @staticmethod
    def hold(stock, _):
        pass

    def reset(self):
        self.cash = self.initial_cash
        self.positions = []

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

    while True:

        code_index, action_index = random.choice(range(trader.codes_count)), random.choice(range(trader.action_space))

        # market_status = market.forward(trader, [code_index, action_index])

        trader.log_asset()

        # if market_status == MarketStatus.NotRunning:
        #     break


if __name__ == '__main__':
    main()
