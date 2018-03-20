# coding=utf-8

import pandas as pd
import numpy as np

import logging
import math

from base.model.finance import Stock
from sklearn import preprocessing
from enum import Enum


class StockEnv(object):

    ModeRL, ModeSL, ModeMIX = 0, 1, 2

    def __init__(self, codes, start_date="2008-01-01", end_date="2018-01-01", **options):
        self.market = Market(codes, start_date, end_date, **options)

        try:
            self.session = options['session']
        except KeyError:
            self.session = None

        try:
            self.agent = options['agent_class'](self.session,
                                                self.market.trader.action_space,
                                                self.market.data_dim)
        except KeyError:
            self.agent = None

        try:
            self.predictor = options['predictor_class'](self.session,
                                                        self.market.seq_length,
                                                        self.market.data_dim,
                                                        self.market.code_count)
        except KeyError:
            self.predictor = None

        try:
            self.episodes = options['episodes']
        except KeyError:
            self.episodes = 300

        try:
            logging.basicConfig(level=options['log_level'])
        except KeyError:
            logging.basicConfig(level=logging.WARNING)

        self._run_methods = [self._run_rl_mode, self._run_sl_mode]

    def run(self, mode=ModeRL):
        self._run_methods[mode]()

    def _run_rl_mode(self):
        if not self.agent:
            raise ValueError("In RL mode, predictor cannot be None")
        for episode in range(self.episodes):
            self.market.trader.log_asset(episode)
            self.agent.log_loss(episode)
            s = self.market.reset()
            while True:
                a = self.agent.predict_action(s)
                a_indices = self._get_a_indices(a)
                s_next, r, status, info = self.market.forward(a_indices)
                a = np.array(a_indices).reshape((1, -1))
                self.agent.save_transition(s, a, r, s_next)
                self.agent.train()
                s = s_next
                if status == MarketStatus.NotRunning:
                    break

    def _run_sl_mode(self):
        if not self.predictor:
            raise ValueError("In SL mode, predictor cannot be None")
        self.predictor.train(self.market.get_batch_sequences)

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

        self.origin_stock_frames = dict()
        self.scaled_stock_frames = dict()

        self.scaled_stock_seqs_x = None
        self.scaled_stock_seqs_y = None

        self.next_date = None
        self.current_date = None

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

        self._init_stocks_data(start_date, end_date)

        self.trader = Trader(self)

        self.iter_dates = iter(self.dates)

    @property
    def state(self):
        return self._get_state(self.current_date)

    @property
    def code_count(self):
        return len(self.codes)

    @property
    def data_dim(self):
        if self.use_sequence:
            data_frame = self.scaled_stock_frames[self.codes[0]]
            data_dim = len(self.codes) * data_frame.shape[1]
            return data_dim
        else:
            data_frame = self.scaled_stock_frames[self.codes[0]]
            if self.use_one_hot:
                data_dim = len(self.codes) * data_frame.shape[1]
                if self.use_state_mix_cash:
                    data_dim += 2
            else:
                data_dim = len(self.codes) * data_frame.shape[1]
            return data_dim

    def _init_stocks_data(self, start_date, end_date):
        self._remove_invalid_codes()
        self._init_stock_frames(start_date, end_date)
        self._init_sequences()

    def forward(self, action_keys):
        # Check trader.
        self.trader.remove_invalid_positions()
        self.trader.reset_reward()
        # Here, action_sheet is like: [-1, 1, ..., -1, 0]
        for index, code in enumerate(self.codes):
            # Get Stock for current date with code.
            action_code = action_keys[index]
            action = self.trader.action_dic[ActionCode(action_code)]
            try:
                stock = self._get_stock_data(code, self.current_date)
                stock_next = self._get_stock_data(code, self.next_date)
                action(code, stock, 100, stock_next)
            except KeyError:
                logging.info("Current date cannot trade for code: {}.".format(code))

        # Update and return the next state.
        try:
            self.current_date, self.next_date = next(self.iter_dates), next(self.iter_dates)
            return self.state, self.trader.reward, MarketStatus.Running, 0
        except StopIteration:
            return self.state, self.trader.reward, MarketStatus.NotRunning, -1

    def reset(self):
        self.trader.reset()
        self.iter_dates = iter(self.dates)
        try:
            self.current_date = next(self.iter_dates)
            self.next_date = next(self.iter_dates)
        except StopIteration:
            raise ValueError("Initialize failed, dates are too short.")
        return self.state

    def get_batch_sequences(self, batch_size=32):
        batch_indices = np.random.choice(self.sequence_indices, batch_size)
        batch_x_sequences = self.scaled_stock_seqs_x[batch_indices]
        batch_y_sequences = self.scaled_stock_seqs_y[batch_indices]
        return batch_x_sequences, batch_y_sequences

    def _get_stock_data(self, code, date):
        return self.origin_stock_frames[code].loc[date]

    def _remove_invalid_codes(self):
        if not len(self.codes):
            raise ValueError("Initialize, codes cannot be empty.")
        valid_codes = [code for code in self.codes if Stock.exist_in_db(code)]
        if not len(valid_codes):
            raise ValueError("Fatal Error: No valid codes or empty codes.")
        self.codes = valid_codes

    def _init_stock_frames(self, start_date, end_date):
        dates_stocks_pairs = []
        # Init stocks data.
        for code in self.codes:
            # Get stocks data by code.
            stocks = Stock.get_k_data(code, start_date, end_date)
            # Init stocks dicts.
            stock_dicts = [stock.to_dic() for stock in stocks]
            # Get dates and stock data, build frames, save date.
            dates, stocks = [stock[1] for stock in stock_dicts], [stock[2:] for stock in stock_dicts]
            dates_stocks_pairs.append((dates, stocks))
            [self.dates.append(date) for date in dates if date not in self.dates]
            # Cache stock data.
            scaler = preprocessing.MinMaxScaler()
            stocks_scaled = scaler.fit_transform(stocks)
            columns = ['open', 'high', 'low', 'close', 'volume']
            origin_stock_frame = pd.DataFrame(data=stocks, index=dates, columns=columns)
            scaled_stock_frame = pd.DataFrame(data=stocks_scaled, index=dates, columns=columns)
            self.origin_stock_frames[code] = origin_stock_frame
            self.scaled_stock_frames[code] = scaled_stock_frame

        self.dates = sorted(self.dates)

        for code in self.codes:
            origin_stock_frame = self.origin_stock_frames[code].reindex(self.dates, method='bfill')
            scaled_stock_frame = self.scaled_stock_frames[code].reindex(self.dates, method='bfill')
            self.origin_stock_frames[code] = origin_stock_frame
            self.scaled_stock_frames[code] = scaled_stock_frame

    def _init_sequences(self):
        if self.use_sequence:
            self.dates = self.dates[self.seq_length - 1: -1 - (self.seq_length - 1)]
            scaled_stock_seqs_x, scaled_stock_seqs_y = [], []
            for date_index, date in enumerate(self.dates):
                # Get x.
                stock_seq_x = []
                stock_seq_y = []
                for seq_index in range(self.seq_length):
                    stocks_x = []
                    for code in self.codes:
                        loc_index = date_index + seq_index + self.seq_length - 1
                        stock = self.scaled_stock_frames[code].iloc[loc_index]
                        stocks_x.append(stock)
                    stocks_x = np.array(stocks_x).reshape((-1))

                    stock_seq_x.append(stocks_x)
                stocks_y = []
                for code in self.codes:
                    label = self.scaled_stock_frames[code]['close'].iloc[date_index + self.seq_length]
                    stocks_y.append(label)
                stocks_y = np.array(stocks_y)
                stock_seq_y.append(stocks_y)
                scaled_stock_seqs_x.append(np.array(stock_seq_x))
                scaled_stock_seqs_y.append(np.array(stock_seq_y).reshape(-1))
            self.scaled_stock_seqs_x = np.array(scaled_stock_seqs_x)
            self.scaled_stock_seqs_y = np.array(scaled_stock_seqs_y)
            self.sequence_indices = np.arange(0, len(scaled_stock_seqs_x))

    def _get_state(self, date):
        if not self.use_sequence:
            state = [self.scaled_stock_frames[code].loc[date] for code in self.codes]
            state = np.array(state)
            if self.use_one_hot:
                state = state.reshape((1, -1))
                if self.use_state_mix_cash:
                    state = np.insert(state, 0, self.trader.cash / self.trader.initial_cash, axis=1)
                    state = np.insert(state, 0, self.trader.holdings_value / self.trader.initial_cash, axis=1)
            return state
        else:
            return self.scaled_stock_seqs_x[self.dates.index(date)]


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
        self.cur_action_code = None
        self.cur_action_status = None
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
                    self.reward += 10
                else:
                    self.reward -= 10
            else:
                self.reward -= 50
        elif action_code == ActionCode.Sell:
            if action_status == ActionStatus.Success:
                if position.pro_value > position.cur_value:
                    self.reward -= 10
                else:
                    self.reward += 10
            else:
                self.reward -= 50
        else:
            if action_status == ActionStatus.Success:
                if position.pro_value > position.cur_value:
                    self.reward += 10
                else:
                    self.reward -= 10
            else:
                self.reward -= 50

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

        if status == MarketStatus.NotRunning:
            break


if __name__ == '__main__':
    main()
