# coding=utf-8

import numpy as np

import logging
import math

from spider.finance import StockSpider
from base.model.finance import Stock
from sklearn import preprocessing
from enum import Enum


logging.basicConfig(level=logging.WARNING)


class StockEnv(object):

    def __init__(self, session, codes, agent_class, start_date="2008-01-01", end_date="2018-01-01", **options):
        self.market = Market(codes, start_date, end_date, **options)
        self.agent = agent_class(session, self.market.trader.action_space, self.market.data_dim)

        try:
            self.episodes = options['episodes']
        except KeyError:
            self.episodes = 300

    def run(self):
        for episode in range(self.episodes):
            self.market.trader.log_asset(episode)
            s = self.market.reset()
            while True:
                a = self.agent.predict_action(s)
                a_indices = self._get_a_indices(a)
                s_next, r, status, info = self.market.forward(a_indices)
                self.agent.save_transition(s, a, r, s_next)
                self.agent.train_if_need()
                s = s_next
                if status == MarketStatus.NotRunning:
                    break

    @staticmethod
    def _get_a_indices(a):
        return np.where(a > 1 / 3, 1, np.where(a < - 1 / 3, -1, 0)).astype(np.int32)[0].tolist()


class MarketStatus(Enum):
    Running = 0
    NotRunning = 1


class Market(object):

    def __init__(self, codes, start_date="2008-01-01", end_date="2018-01-01", **options):

        self.codes = codes
        self.dates = []

        self.stocks_list = []

        self.date_stocks_map = dict()
        self.date_states_map = dict()
        self.dates_stocks_pairs = []

        self.data_dim = None
        self.current_date = None

        try:
            self.use_one_hot = options['use_one_hot']
        except KeyError:
            self.use_one_hot = True

        try:
            self.use_normalized = options['use_normalized']
        except KeyError:
            self.use_normalized = True

        try:
            self.use_state_mix_cash = options['state_mix_cash']
        except KeyError:
            self.use_state_mix_cash = True

        self._init_stocks_data(start_date, end_date)
        self._init_date_stocks_map()
        self._init_date_states_map()

        self.trader = Trader(self)

        self.dates = sorted(self.dates)
        self.iter_dates = iter(self.dates)

    def _init_stocks_data(self, start_date, end_date):
        # Check if codes are valid.
        if not len(self.codes):
            raise ValueError("Initialize, codes cannot be empty.")
        # Init stocks data.
        for code in self.codes:
            # Get stocks data by code.
            stocks = Stock.get_k_data(code, start_date, end_date)
            self.stocks_list.append(stocks)
            if not self.data_dim:
                # Init date dim.
                self.data_dim = self._get_data_dim(stocks.first())
            # Init stocks dicts.
            stock_dicts = [stock.to_dic() for stock in stocks]
            # Get dates and stock data.
            dates, stocks = [stock[1] for stock in stock_dicts], [stock[2:] for stock in stock_dicts]
            # Build date map.
            [self.dates.append(date) for date in dates if date not in self.dates]
            # Normalize data.
            scaler = preprocessing.MinMaxScaler()
            stocks_scaled = scaler.fit_transform(stocks)
            # Cache stock data.
            self.dates_stocks_pairs.append((dates, stocks_scaled))

    def _init_date_stocks_map(self):
        for index, stocks in enumerate(self.stocks_list):
            for stock in stocks:
                try:
                    stock_dic = self.date_stocks_map[stock.date]
                    stock_dic[stock.code] = stock
                except KeyError:
                    self.date_stocks_map[stock.date] = {stock.code: stock}

    def _init_date_states_map(self):
        for date in self.dates:
            self.date_states_map[date] = []
            for pair_index, (dates, stocks) in enumerate(self.dates_stocks_pairs):
                try:
                    stock_index = dates.index(date)
                    stock = stocks[stock_index]
                    stocks_list = self.date_states_map[date]
                    stocks_list.append(stock)
                except ValueError:
                    self._fill_stop_date_data(date, pair_index)

    def get_cur_stock_data(self, code):
        return self.date_stocks_map[self.current_date][code]

    def reset(self):
        self.trader.reset()
        self.iter_dates = iter(self.dates)
        try:
            self.current_date = next(self.iter_dates)
        except StopIteration:
            raise ValueError("Initialize failed, dates is empty.")
        return self.state

    def forward(self, action_keys):
        # Check trader.
        if not self.trader:
            raise ValueError("Trader cannot be None.")

        # Here, action_sheet is like: [-1, 1, ..., -1, 0]
        for index, code in enumerate(self.codes):
            # Get Stock for current date with code.
            action_key = action_keys[index]
            action = self.trader.action_dic[action_key]
            try:
                stock = self.get_cur_stock_data(code)
                action(stock, 100)
            except KeyError:
                logging.info("Current date cannot trade for code: {}.".format(code))

        # Update and return the next state.
        try:
            self.current_date = next(self.iter_dates)
            return self.state, self.trader.reward, MarketStatus.Running, 0
        except StopIteration:
            return self.state, self.trader.reward, MarketStatus.NotRunning, -1

    @property
    def state(self):
        stocks = np.array([stock for stock in self.date_states_map[self.current_date]])
        if self.use_one_hot:
            stocks = stocks.reshape((1, -1))
            if self.use_state_mix_cash:
                stocks = np.insert(stocks, 0, self.trader.cash / self.trader.initial_cash, axis=1)
                stocks = np.insert(stocks, 0, self.trader.holdings_value / self.trader.initial_cash, axis=1)
        return stocks

    def _fill_stop_date_data(self, date, stock_index):
        index = list(self.date_states_map.keys()).index(date)
        last_valid_stocks = self._get_last_valid_stocks_data(index)
        stocks_list = self.date_states_map[date]
        stocks_list.insert(stock_index, last_valid_stocks[stock_index])

    def _get_data_dim(self, stock):
        if self.use_one_hot:
            data_dim = len(self.codes) * len(stock.to_state())
            if self.use_state_mix_cash:
                data_dim += 2
        else:
            data_dim = len(self.codes) * len(stock.to_state())
        return data_dim

    def _get_last_valid_stocks_data(self, index, backward=True):
        _index = index - 1 if backward else index + 1
        try:
            date = list(self.date_states_map.keys())[_index]
        except IndexError:
            return self._get_last_valid_stocks_data(_index, backward=False)
        stocks = self.date_states_map[date]
        if len(stocks) == len(self.codes):
            return stocks
        else:
            return self._get_last_valid_stocks_data(_index, backward=backward)


class Trader(object):

    ActionBuy = 1
    ActionHold = 0
    ActionSell = -1

    def __init__(self, market, cash=100000.0):
        self.cash = cash
        self.codes = market.codes
        self.market = market
        self.profits = 0
        self.positions = []
        self.last_profits = 0
        self.initial_cash = cash
        self.action_dic = {Trader.ActionBuy: self.buy, Trader.ActionHold: self.hold, Trader.ActionSell: self.sell}

    @property
    def codes_count(self):
        return len(self.codes)

    @property
    def action_space(self):
        return self.codes_count

    @property
    def holdings_value(self):
        holdings_value = 0
        for position in self.positions:
            holdings_value += position.holding_value
        return holdings_value

    @property
    def reward(self):
        floating_profits = self.profits - self.last_profits
        if floating_profits > 0:
            return 10
        elif floating_profits == 0:
            return -1
        else:
            return -5

    def buy(self, stock, amount):

        # Check if amount is OK.
        amount = amount if self.cash > stock.close * amount else int(math.floor(self.cash / stock.close))

        if amount == 0:
            return logging.info("Code: {}, not enough cash.".format(stock.code))

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
        self._update_profits()

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
        self._update_profits()

    def hold(self, stock, amount):
        self._update_profits()

    def reset(self):
        self.cash = self.initial_cash
        self.positions = []
        self.profits = 0
        self.last_profits = 0

    def log_asset(self, episode):
        logging.warning(
            "Episode: {0} | "
            "Cash: {1:.2f} | "
            "Holdings: {2:.2f} | "
            "Profits: {3:.2f}".format(episode, self.cash, self.holdings_value, self.profits))

    def _update_profits(self):
        self.last_profits = self.profits
        self.profits = self.holdings_value + self.cash - self.initial_cash

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

        actions_indices = [np.random.choice([-1, 0, 1]) for _ in codes]

        s_next, r, status, info = market.forward(actions_indices)

        market.trader.log_asset("1")

        if status == MarketStatus.NotRunning:
            break


if __name__ == '__main__':
    main()
