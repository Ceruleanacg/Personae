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
        self.date_stocks_map = dict()
        self.current_date = None

        if not len(self.codes):
            raise ValueError("Initialize, codes cannot be empty.")

        for code in self.codes:
            # Get stocks date by code.
            stocks = Stock.get_k_data(code, start_date, end_date)
            if not self.data_dim:
                # Init date dim.
                self.data_dim = self._get_data_dim(stocks.first())
            # Init date - stock map.
            for stock in stocks:
                if stock.date not in self.dates:
                    self.dates.append(stock.date)
                try:
                    stocks_data = self.date_stocks_map[stock.date]
                    stocks_data.append(stock)
                except KeyError:
                    stocks_data = [stock]
                    self.date_stocks_map[stock.date] = stocks_data

        for index, date in enumerate(self.date_stocks_map):
            stocks = self.date_stocks_map[date]
            if len(stocks) != len(self.codes):
                last_valid_stocks = self._get_last_valid_stocks_data(index)
                for stock in last_valid_stocks:
                    if stock.code not in stocks:
                        stocks.append(stock)

        self.dates = sorted(self.dates)
        self.iter_dates = iter(self.dates)

    def get_cur_stock_data(self, code_index):
        # stocks_data will never be None or [].
        stocks_data = self.date_stocks_map[self.current_date]
        try:
            return stocks_data[code_index]
        except IndexError:
            code = self.codes[code_index]
            raise IndexError("Code: {}, not exists in Market on Date: {}.".format(code, self.current_date))

    def reset(self):
        self.trader.reset()
        self.iter_dates = iter(self.dates)
        try:
            self.current_date = next(self.iter_dates)
        except StopIteration:
            raise ValueError("Initialize failed, dates is empty.")
        return self.state

    def forward(self, action_indices):
        # Check trader.
        if not self.trader:
            raise ValueError("Trader cannot be None.")

        # Here, action_sheet is like: [0, 1, ..., 1, 2]
        for index, code in enumerate(self.codes):
            # Get Stock for current date with code.
            action_index = action_indices[index]
            action = self.trader.actions[action_index]
            try:
                stock = self.get_cur_stock_data(index)
                action(stock, 100)
            except IndexError:
                logging.info("Current date cannot trade for code: {}.".format(code))

        # Update and return the next state.
        try:
            self.current_date = next(self.iter_dates)
            return self.state, self.trader.profits, MarketStatus.Running, "Running."
        except StopIteration:
            return None, self.trader.profits, MarketStatus.NotRunning, "Not Running."

    @property
    def state(self):
        stocks = [stock.to_state() for stock in self.date_stocks_map[self.current_date]]
        return stocks

    def _get_data_dim(self, stock):
        return len(self.codes), len(stock.to_state())

    def _get_last_valid_stocks_data(self, index):
        _index = index - 1
        try:
            date = list(self.date_stocks_map.keys())[_index]
        except IndexError:
            _index = index + 1
            date = list(self.date_stocks_map.keys())[_index]
        stocks = self.date_stocks_map[date]
        if len(stocks) == len(self.codes):
            return stocks
        else:
            return self._get_last_valid_stocks_data(_index)


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

        # Sell position if possible.
        amount = amount if amount < position.amount else position.amount
        position.sub(stock.close, amount)

        if position.amount == 0:
            self.positions.remove(position)

        # Update cash and holding price.
        self.cash += amount * stock.close

    def hold(self, stock, _):
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


def main():

    codes = ["600036", "601998"]
    market = Market(codes)
    market.reset()

    while True:

        actions_indices = [random.choice(range(market.trader.action_space)) for _ in codes]

        s_next, r, status, info = market.forward(actions_indices)

        market.trader.log_asset()

        if status == MarketStatus.NotRunning:
            break


if __name__ == '__main__':
    main()
