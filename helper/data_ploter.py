# coding=utf-8

import matplotlib.pyplot as plt
import math


def plot_stock_series(codes, y, label, save_path, y_desc='Predict', label_desc='Real'):
    row, col = int(math.ceil(len(codes) / 2)), 1 if len(codes) == 1 else 2
    plt.figure(figsize=(20, 15))
    for index, code in enumerate(codes):
        plt.subplot(row * 100 + col * 10 + (index + 1))
        plt.title(code)
        plt.plot(y[:, index], label=y_desc)
        plt.plot(label[:, index], label=label_desc)
        # plt.plot(y[:, index], 'o-', label=y_desc)
        # plt.plot(label[:, index], 'o-', label=label_desc)
        plt.legend(loc='upper left')
    # plt.show()
    plt.savefig(save_path, dpi=200)


def plot_profits_series(base, profits, save_path):
    plt.figure(figsize=(20, 15))
    plt.subplot(111)
    plt.title("Profits - Baseline")
    plt.plot(base, label='base')
    plt.plot(profits, label='profits')
    plt.legend(loc='upper left')
    # plt.show()
    plt.savefig(save_path, dpi=200)
