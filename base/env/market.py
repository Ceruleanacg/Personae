# coding=utf-8

import pandas as pd
import numpy as np
import math

from sklearn.preprocessing import MinMaxScaler
from base.env.trader import Trader
from base.model.document import Stock, Future


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

        # Initialize data frames.
        self.origin_frames = dict()
        self.scaled_frames = dict()

        # Initialize scaled  data x, y.
        self.data_x = None
        self.data_y = None

        # Initialize scaled seq data x, y.
        self.seq_data_x = None
        self.seq_data_y = None

        # Initialize flag date.
        self.next_date = None
        self.iter_dates = None
        self.current_date = None

        # Init scalers.
        self.scalers = [MinMaxScaler() for _ in self.codes]

        # Initialize parameters.
        self._init_options(**options)

        # Initialize stock data.
        self._init_data(start_date, end_date)

    def _init_options(self, **options):

        try:
            self.m_type = options['market']
        except KeyError:
            self.m_type = 'stock'

        try:
            self.init_cash = options['cash']
        except KeyError:
            self.init_cash = 100000
        finally:
            self.max_profits = self.init_cash * 5

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
        self.doc_class = Stock if self.m_type == 'stock' else Future

    def _init_data(self, start_date, end_date):
        self._init_data_frames(start_date, end_date)
        self._init_env_data()
        self._init_data_indices()

    def _remove_invalid_codes(self):
        if not len(self.codes):
            raise ValueError("Fatal error, odes cannot be empty.")
        valid_codes = [code for code in self.codes if self.doc_class.exist_in_db(code)]
        if not len(valid_codes):
            raise ValueError("Fatal error, no valid codes in database.")
        self.codes = valid_codes

    def _init_data_frames(self, start_date, end_date):
        # Remove invalid codes first.
        self._remove_invalid_codes()
        # Init columns and data set.
        columns, dates_set = ['open', 'high', 'low', 'close', 'volume'], set()
        # Load data.
        for index, code in enumerate(self.codes):
            # Load instrument docs by code.
            instrument_docs = self.doc_class.get_k_data(code, start_date, end_date)
            # Init instrument dicts.
            instrument_dicts = [instrument.to_dic() for instrument in instrument_docs]
            # Split dates.
            dates = [instrument[1] for instrument in instrument_dicts]
            # Split instruments.
            instruments = [instrument[2:] for instrument in instrument_dicts]
            # Update dates set.
            dates_set = dates_set.union(dates)
            # Build origin and scaled frames.
            scaler = self.scalers[index]
            scaler.fit(instruments)
            instruments_scaled = scaler.transform(instruments)
            origin_frame = pd.DataFrame(data=instruments, index=dates, columns=columns)
            scaled_frame = pd.DataFrame(data=instruments_scaled, index=dates, columns=columns)
            # Build code - frame map.
            self.origin_frames[code] = origin_frame
            self.scaled_frames[code] = scaled_frame
        # Init date iter.
        self.dates = sorted(list(dates_set))
        # Rebuild index.
        for code in self.codes:
            origin_frame = self.origin_frames[code]
            scaled_frame = self.scaled_frames[code]
            self.origin_frames[code] = origin_frame.reindex(self.dates, method='bfill')
            self.scaled_frames[code] = scaled_frame.reindex(self.dates, method='bfill')

    def _init_env_data(self):
        if not self.use_sequence:
            self._init_series_data()
        else:
            self._init_sequence_data()

    def _init_series_data(self):
        # Calculate data count.
        self.data_count = len(self.dates[: -1])
        # Calculate bound index.
        self.bound_index = int(self.data_count * self.training_data_ratio)
        # Init scaled_x, scaled_y.
        scaled_data_x, scaled_data_y = [], []
        for index, date in enumerate(self.dates[: -1]):
            # Get current x, y.
            x = [self.scaled_frames[code].iloc[index] for code in self.codes]
            y = [self.scaled_frames[code].iloc[index + 1] for code in self.codes]
            # Convert x, y to array.
            x = np.array(x)
            y = np.array(y)
            if self.use_one_hot:
                x = y.reshape((1, -1))
            # Append x, y
            scaled_data_x.append(x)
            scaled_data_y.append(y)
        # Convert list to array.
        self.data_x = np.array(scaled_data_x)
        self.data_y = np.array(scaled_data_y)

    def _init_sequence_data(self):
        # Calculate data count.
        self.data_count = len(self.dates[: -1 - self.seq_length])
        # Calculate bound index.
        self.bound_index = int(self.data_count * self.training_data_ratio)
        # Init seqs_x, seqs_y.
        scaled_seqs_x, scaled_seqs_y = [], []
        # Scale to valid dates.
        for date_index, date in enumerate(self.dates[: -1]):
            # Continue until valid date index.
            if date_index < self.seq_length:
                continue
            data_x, data_y = [], []
            for code in self.codes:
                # Get scaled frame by code.
                scaled_frame = self.scaled_frames[code]
                # Get instrument data x.
                instruments_x = scaled_frame.iloc[date_index - self.seq_length: date_index]
                # Get instrument data y.
                if date_index < self.bound_index:
                    # Get y, y is not at date index, but plus 1. (Training Set)
                    instruments_y = scaled_frame.iloc[date_index + 1]['close']
                else:
                    # Get y, y is at date index. (Test Set)
                    instruments_y = scaled_frame.iloc[date_index]['close']
                data_x.append(np.array(instruments_x))
                data_y.append(np.array(instruments_y))
            # Convert list to array.
            data_x = np.array(data_x)
            data_y = np.array(data_y)
            seq_x = []
            seq_y = data_y
            # Build seq x, y.
            for seq_index in range(self.seq_length):
                seq_x.append(data_x[:, seq_index, :].reshape((-1)))
            # Convert list to array.
            seq_x = np.array(seq_x)
            scaled_seqs_x.append(seq_x)
            scaled_seqs_y.append(seq_y)
        # Convert seq from list to array.
        self.seq_data_x = np.array(scaled_seqs_x)
        self.seq_data_y = np.array(scaled_seqs_y)

    def _init_data_indices(self):
        # Calculate indices range.
        self.data_indices = np.arange(0, self.data_count)
        # Calculate train and eval indices.
        self.t_data_indices = self.data_indices[:self.bound_index]
        self.e_data_indices = self.data_indices[self.bound_index:]
        # Generate train and eval dates.
        self.t_dates = self.dates[:self.bound_index]
        self.e_dates = self.dates[self.bound_index:]

    def _origin_data(self, code, date):
        return self.origin_frames[code].loc[date]

    def _scaled_data_as_state(self, date):
        if self.use_sequence:
            return self.seq_data_x[self.dates.index(date)]
        else:
            data = self.data_x[self.dates.index(date)]
            if self.use_state_mix_cash:
                data = np.insert(data, 0, self.trader.cash / self.max_profits, axis=1)
                data = np.insert(data, 0, self.trader.holdings_value / self.max_profits, axis=1)
            return data

    def reset(self, mode='train'):
        # Reset trader.
        self.trader.reset()
        # Reset iter dates by mode.
        self.iter_dates = iter(self.t_dates) if mode == 'train' else iter(self.e_dates)
        try:
            self.current_date = next(self.iter_dates)
            self.next_date = next(self.iter_dates)
        except StopIteration:
            raise ValueError("Reset error, dates are empty.")
        # Reset baseline.
        self._reset_baseline()
        return self._scaled_data_as_state(self.current_date)

    def get_batch_data(self, batch_size=32):
        batch_indices = np.random.choice(self.t_data_indices, batch_size)
        if not self.use_sequence:
            batch_x = self.data_x[batch_indices]
            batch_y = self.data_y[batch_indices]
        else:
            batch_x = self.seq_data_x[batch_indices]
            batch_y = self.seq_data_y[batch_indices]
        return batch_x, batch_y

    def get_test_data(self):
        if not self.use_sequence:
            test_x = self.data_x[self.e_data_indices]
            test_y = self.data_y[self.e_data_indices]
        else:
            test_x = self.seq_data_x[self.e_data_indices]
            test_y = self.seq_data_y[self.e_data_indices]
        return test_x, test_y

    def forward(self, stock_code, action_code):
        # Check Trader.
        self.trader.remove_invalid_positions()
        self.trader.reset_reward()
        # Get stock data.
        stock = self._origin_data(stock_code, self.current_date)
        stock_next = self._origin_data(stock_code, self.next_date)
        # Execute transaction.
        action = self.trader.action_by_code(action_code)
        action(stock_code, stock, 100, stock_next)
        # Init episode status.
        episode_done = self.Running
        # Add action times.
        self.trader.action_times += 1
        # Update date if need.
        if self.trader.action_times == self.code_count:
            self.trader.action_times = 0
            try:
                self.current_date, self.next_date = self.next_date, next(self.iter_dates)
            except StopIteration:
                episode_done = self.Done
            finally:
                self._update_profits_and_baseline()
        # Get next state.
        state_next = self._scaled_data_as_state(self.current_date)
        # Return s_n, r, d, info.
        return state_next, self.trader.reward, episode_done, self.trader.cur_action_status

    def _update_profits_and_baseline(self):
        prices = [self._origin_data(code, self.current_date).close for code in self.codes]
        self.trader.history_baselines.append(np.sum(np.multiply(self.stocks_holding_baseline, prices)))
        self.trader.history_profits.append(self.trader.profits + self.trader.initial_cash)

    def _reset_baseline(self):
        # Calculate cash piece.
        cash_piece = self.init_cash / self.code_count
        # Get stocks data.
        stocks = [self._origin_data(code, self.current_date) for code in self.codes]
        # Init stocks baseline.
        self.stocks_holding_baseline = [int(math.floor(cash_piece / stock.close)) for stock in stocks]

    @property
    def code_count(self):
        return len(self.codes)

    @property
    def data_dim(self):
        if self.use_sequence:
            data_dim = self.code_count * self.scaled_frames[self.codes[0]].shape[1]
            return data_dim
        else:
            if self.use_one_hot:
                data_dim = self.code_count * self.scaled_frames[self.codes[0]].shape[1]
                if self.use_state_mix_cash:
                    data_dim += 2
            else:
                data_dim = self.code_count * self.scaled_frames[self.codes[0]].shape[1]
            return data_dim
