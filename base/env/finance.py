# coding=utf-8

import pandas as pd
import numpy as np

import logging
import math

from base.model.document import Stock
from sklearn import preprocessing
from enum import Enum


class Market(object):

    Running = 0
    Done = -1

    def __init__(self, codes, start_date="2008-01-01", end_date="2018-01-01", **options):

        # Initialize codes.
        self.codes = codes

        # Initialize dates.
        self.dates = []
        self.t_dates = []
        self.e_dates = []

        # Initialize stocks data frames.
        self.origin_stock_frames = dict()
        self.scaled_stock_frames = dict()

        # Initialize scaled stocks data x, y.
        self.stocks_x = None
        self.stocks_y = None

        # Initialize scaled seq stocks data x, y.
        self.seq_stocks_x = None
        self.seq_stocks_y = None

        # Initialize flag date.
        self.next_date = None
        self.iter_dates = None
        self.current_date = None

        # Initialize parameters.
        self._init_parameters(**options)

        # Initialize stock data.
        self._init_stocks_data(start_date, end_date)

    def forward(self, action_keys):
        # Check trader.
        self.trader.remove_invalid_positions()
        self.trader.reset_reward()
        # Init current prices.
        stocks_price = []
        # Here, action_sheet is like: [-1, 1, ..., -1, 0]
        for index, code in enumerate(self.codes):
            # Get Stock for current date with code.
            action_code = action_keys[index]
            action = self.trader.action_dic[ActionCode(action_code)]
            try:
                stock = self._get_origin_stock_data(code, self.current_date)
                stock_next = self._get_origin_stock_data(code, self.next_date)
                action(code, stock, 100, stock_next)
                stocks_price.append(stock.close)
            except KeyError:
                logging.info("Current date cannot trade for code: {}.".format(code))
        # Update and return the next state.
        self.trader.history_baseline_profits.append(np.sum(np.multiply(self.stocks_holding_baseline, stocks_price)))
        self.trader.history_profits.append(self.trader.profits + self.trader.initial_cash)
        try:
            self.current_date, self.next_date = next(self.iter_dates), next(self.iter_dates)
            return self._get_scaled_stock_data_as_state(self.current_date), self.trader.reward, self.Running, 0
        except StopIteration:
            return self._get_scaled_stock_data_as_state(self.current_date), self.trader.reward, self.Done, -1

    def reset(self, mode='train'):
        self.trader.reset()
        self.iter_dates = iter(self.t_dates) if mode == 'train' else iter(self.e_dates)
        try:
            self.current_date = next(self.iter_dates)
            self.next_date = next(self.iter_dates)
        except StopIteration:
            raise ValueError("Initialize failed, dates are too short.")
        self._reset_stocks_holding_baseline()
        return self._get_scaled_stock_data_as_state(self.current_date)

    def get_stock_batch_data(self, batch_size=32):
        batch_indices = np.random.choice(self.t_data_indices, batch_size)
        if not self.use_sequence:
            batch_x = self.stocks_x[batch_indices]
            batch_y = self.stocks_y[batch_indices]
        else:
            batch_x = self.seq_stocks_x[batch_indices]
            batch_y = self.seq_stocks_y[batch_indices]
        return batch_x, batch_y

    def get_stock_test_data(self):
        if not self.use_sequence:
            test_x = self.stocks_x[self.e_data_indices]
            test_y = self.stocks_y[self.e_data_indices]
        else:
            test_x = self.seq_stocks_x[self.e_data_indices]
            test_y = self.seq_stocks_y[self.e_data_indices]
        return test_x, test_y

    def _init_parameters(self, **options):

        try:
            self.init_cash = options['cash']
        except KeyError:
            self.init_cash = 100000

        try:
            self.use_sequence = options['use_sequence']
        except KeyError:
            self.use_sequence = False

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

        try:
            self.seq_length = options['seq_length']
        except KeyError:
            self.seq_length = 5
        finally:
            self.seq_length = self.seq_length if self.seq_length > 1 else 2

        try:
            self.training_data_ratio = options['training_data_ratio']
        except KeyError:
            self.training_data_ratio = 0.7

        self.trader = Trader(self, cash=self.init_cash)

    def _init_stocks_data(self, start_date, end_date):
        self._init_stock_frames_data(start_date, end_date)
        self._init_stock_env_data()
        self._init_stock_data_indices()

    def _remove_invalid_codes(self):
        if not len(self.codes):
            raise ValueError("Initialize, codes cannot be empty.")
        valid_codes = [code for code in self.codes if Stock.exist_in_db(code)]
        if not len(valid_codes):
            raise ValueError("Fatal Error: No valid codes or empty codes.")
        self.codes = valid_codes

    def _init_stock_frames_data(self, start_date, end_date):
        # Remove invalid codes first.
        self._remove_invalid_codes()
        # Init columns and data set.
        columns, dates_set = ['open', 'high', 'low', 'close', 'volume'], set()
        # Init stocks data.
        for code in self.codes:
            # Get stocks data by code.
            stocks = Stock.get_k_data(code, start_date, end_date)
            # Init stocks dicts.
            stock_dicts = [stock.to_dic() for stock in stocks]
            # Get dates and stock data, build frames, save date.
            stocks_date, stocks_data = [stock[1] for stock in stock_dicts], [stock[2:] for stock in stock_dicts]
            # Update dates set.
            dates_set = dates_set.union(stocks_date)
            # Cache stock data.
            stocks_scaled = preprocessing.MinMaxScaler().fit_transform(stocks_data)
            origin_stock_frame = pd.DataFrame(data=stocks_data, index=stocks_date, columns=columns)
            scaled_stock_frame = pd.DataFrame(data=stocks_scaled, index=stocks_date, columns=columns)
            self.origin_stock_frames[code] = origin_stock_frame
            self.scaled_stock_frames[code] = scaled_stock_frame
        # Init dates and date iter.
        self.dates = sorted(list(dates_set))
        # Rebuild index.
        for code in self.codes:
            origin_stock_frame = self.origin_stock_frames[code]
            scaled_stock_frame = self.scaled_stock_frames[code]
            self.origin_stock_frames[code] = origin_stock_frame.reindex(self.dates, method='bfill')
            self.scaled_stock_frames[code] = scaled_stock_frame.reindex(self.dates, method='bfill')

    def _init_stock_env_data(self):
        if not self.use_sequence:
            self._init_series_data()
        else:
            self._init_sequence_data()

    def _init_series_data(self):
        self.dates = self.dates[: -1 - 1]
        scaled_stocks_x, scaled_stocks_y = [], []
        for index, date in enumerate(self.dates):
            stock = [self.scaled_stock_frames[code].iloc[index] for code in self.codes]
            label = [self.scaled_stock_frames[code].iloc[index + 1] for code in self.codes]
            stock = np.array(stock)
            label = np.array(label)
            if self.use_one_hot:
                stock = stock.reshape((1, -1))
            scaled_stocks_x.append(stock)
            scaled_stocks_y.append(label)
        self.stocks_x = np.array(scaled_stocks_x)
        self.stocks_y = np.array(scaled_stocks_y)
        self.data_count = len(scaled_stocks_x)

    def _init_sequence_data(self):
        # Init seqs_x, seqs_y.
        scaled_stock_seqs_x, scaled_stock_seqs_y = [], []
        for date_index, date in enumerate(self.dates[:-1 - 1]):
            # wait until valid date index.
            if date_index < self.seq_length:
                continue
            stocks_data_x, stocks_data_y = [], []
            for code in self.codes:
                stocks = self.scaled_stock_frames[code].iloc[date_index - self.seq_length:date_index + 1]
                stocks_data_x.append(np.array(stocks[:-1]))
                stocks_data_y.append(np.array(stocks.iloc[-1]['close']))
            stocks_data_x = np.array(stocks_data_x)
            stocks_data_y = np.array(stocks_data_y)
            stock_seq_x, stock_seq_y = [], stocks_data_y
            for seq_index in range(self.seq_length):
                stock_seq_x.append(stocks_data_x[:, seq_index, :].reshape((-1)))
            stock_seq_x = np.array(stock_seq_x)
            scaled_stock_seqs_x.append(np.array(stock_seq_x))
            scaled_stock_seqs_y.append(stock_seq_y)
        self.seq_stocks_x = np.array(scaled_stock_seqs_x)
        self.seq_stocks_y = np.array(scaled_stock_seqs_y)
        self.data_count = len(scaled_stock_seqs_x)

    def _init_stock_data_indices(self):
        self.data_indices = np.arange(0, self.data_count)
        self.t_data_indices = self.data_indices[:int(self.data_count * self.training_data_ratio)]
        self.e_data_indices = self.data_indices[int(self.data_count * self.training_data_ratio):]
        self.t_dates = self.dates[:int(len(self.dates) * self.training_data_ratio)]
        self.e_dates = self.dates[int(len(self.dates) * self.training_data_ratio):]

    def _get_origin_stock_data(self, code, date):
        return self.origin_stock_frames[code].loc[date]

    def _get_scaled_stock_data_as_state(self, date):
        if self.use_sequence:
            return self.seq_stocks_x[self.dates.index(date)]
        else:
            stock = self.stocks_x[self.dates.index(date)]
            if self.use_state_mix_cash:
                stock = np.insert(stock, 0, self.trader.cash / self.trader.initial_cash, axis=1)
                stock = np.insert(stock, 0, self.trader.holdings_value / self.trader.initial_cash, axis=1)
            return stock

    def _reset_stocks_holding_baseline(self):
        # Calculate cash piece.
        cash_piece = self.init_cash / self.code_count
        # Get stocks data.
        stocks = [self._get_origin_stock_data(code, self.current_date) for code in self.codes]
        # Init stocks baseline.
        self.stocks_holding_baseline = [int(math.floor(cash_piece / stock.close)) for stock in stocks]

    @property
    def code_count(self):
        return len(self.codes)

    @property
    def data_dim(self):
        if self.use_sequence:
            data_dim = self.code_count * self.scaled_stock_frames[self.codes[0]].shape[1]
            return data_dim
        else:
            if self.use_one_hot:
                data_dim = self.code_count * self.scaled_stock_frames[self.codes[0]].shape[1]
                if self.use_state_mix_cash:
                    data_dim += 2
            else:
                data_dim = self.code_count * self.scaled_stock_frames[self.codes[0]].shape[1]
            return data_dim


class ActionCode(Enum):
    Buy = 1
    Hold = 0
    Sell = -1


class ActionStatus(Enum):
    Success = 0
    Failed = -1


class Trader(object):

    def __init__(self, market, cash=100000.0):
        self.cash = cash
        self.codes = market.codes
        self.market = market
        self.reward = 0
        self.positions = []
        self.initial_cash = cash
        self.history_profits = []
        self.cur_action_code = None
        self.cur_action_status = None
        self.history_baseline_profits = []
        self.action_dic = {ActionCode.Buy: self.buy, ActionCode.Hold: self.hold, ActionCode.Sell: self.sell}

    @property
    def codes_count(self):
        return len(self.codes)

    @property
    def action_space(self):
        return self.codes_count

    @property
    def profits(self):
        return self.cash + self.holdings_value - self.initial_cash

    @property
    def holdings_value(self):
        holdings_value = 0
        for position in self.positions:
            holdings_value += position.cur_value
        return holdings_value

    def buy(self, code, stock, amount, stock_next):
        # Check if amount is valid.
        amount = amount if self.cash > stock.close * amount else int(math.floor(self.cash / stock.close))
        # If amount > 0, means cash is enough.
        if amount > 0:
            # Check if position exists.
            if not self._exist_position(code):
                # Build position if possible.
                position = Position(code, stock.close, amount, stock_next.close)
                self.positions.append(position)
            else:
                # Get position and update if possible.
                position = self._get_position(code)
                position.add(stock.close, amount, stock_next.close)
            # Update cash and holding price.
            self.cash -= amount * stock.close
            self._update_reward(ActionCode.Buy, ActionStatus.Success, position)
        else:
            logging.info("Code: {}, not enough cash, cannot buy.".format(code))
            if self._exist_position(code):
                # If position exists, update status.
                position = self._get_position(code)
                position.update_status(stock.close, stock_next.close)
                self._update_reward(ActionCode.Buy, ActionStatus.Failed, position)

    def sell(self, code, stock, amount, stock_next):
        # Check if position exists.
        if not self._exist_position(code):
            logging.info("Code: {}, not exists in Positions, sell failed.".format(code))
            return self._update_reward(ActionCode.Sell, ActionStatus.Failed, None)
        # Sell position if possible.
        position = self._get_position(code)
        amount = amount if amount < position.amount else position.amount
        position.sub(stock.close, amount, stock_next.close)
        # Update cash and holding price.
        self.cash += amount * stock.close
        self._update_reward(ActionCode.Sell, ActionStatus.Success, position)

    def hold(self, code, stock, _, stock_next):
        if not self._exist_position(code):
            logging.info("Code: {}, not exists in Positions, hold failed.")
            return self._update_reward(ActionCode.Hold, ActionStatus.Failed, None)
        position = self._get_position(code)
        position.update_status(stock.close, stock_next.close)
        self._update_reward(ActionCode.Hold, ActionStatus.Failed, position)

    def reset(self):
        self.cash = self.initial_cash
        self.positions = []
        self.history_profits = []
        self.history_baseline_profits = []

    def reset_reward(self):
        self.reward = 0

    def remove_invalid_positions(self):
        self.positions = [position for position in self.positions if position.amount > 0]

    def log_asset(self, episode):
        logging.warning(
            "Episode: {0} | "
            "Cash: {1:.2f} | "
            "Holdings: {2:.2f} | "
            "Profits: {3:.2f}".format(episode, self.cash, self.holdings_value, self.profits)
        )

    def log_reward(self):
        logging.info("Reward: {}".format(self.reward))

    def _update_reward(self, action_code, action_status, position):
        if action_code == ActionCode.Buy:
            if action_status == ActionStatus.Success:
                if position.pro_value > position.cur_value:
                    self.reward += 5
                else:
                    self.reward -= 7
            else:
                self.reward -= 10
        elif action_code == ActionCode.Sell:
            if action_status == ActionStatus.Success:
                if position.pro_value > position.cur_value:
                    self.reward -= 7
                else:
                    self.reward += 5
            else:
                self.reward -= 10
        else:
            if action_status == ActionStatus.Success:
                if position.pro_value > position.cur_value:
                    self.reward += 5
                else:
                    self.reward -= 7
            else:
                self.reward -= 10

    def _exist_position(self, code):
        return True if len([position.code for position in self.positions if position.code == code]) else False

    def _get_position(self, code):
        return [position for position in self.positions if position.code == code][0]


class Position(object):

    def __init__(self, code, buy_price, amount, next_price):
        self.code = code
        self.amount = amount
        self.buy_price = buy_price
        self.cur_price = buy_price
        self.cur_value = self.cur_price * self.amount
        self.pro_value = next_price * self.amount

    def add(self, buy_price, amount, next_price):
        self.buy_price = (self.amount * self.buy_price + amount * buy_price) / (self.amount + amount)
        self.amount += amount
        self.update_status(buy_price, next_price)

    def sub(self, sell_price, amount, next_price):
        self.cur_price = sell_price
        self.amount -= amount
        self.update_status(sell_price, next_price)

    def hold(self, cur_price, next_price):
        self.update_status(cur_price, next_price)

    def update_status(self, cur_price, next_price):
        self.cur_price = cur_price
        self.cur_value = self.cur_price * self.amount
        self.pro_value = next_price * self.amount


def main():

    codes = ["600036", "601328", "601998", "601288"]
    market = Market(codes)
    market.reset()

    logging.basicConfig(level=logging.INFO)

    while True:

        actions_indices = [np.random.choice([-1, 0, 1]) for _ in codes]

        s_next, r, status, info = market.forward(actions_indices)

        market.trader.log_asset("1")
        market.trader.log_reward()

        if status == market.Done:
            break


if __name__ == '__main__':
    main()
