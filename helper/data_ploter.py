# coding=utf-8

import matplotlib.pyplot as plt
import math


def plot_stock_series(codes, y, label, save_path):
    row, col = int(math.ceil(len(codes) / 2)), int(math.ceil(len(codes) / 2))
    plt.figure(figsize=(40, 25))
    for index, code in enumerate(codes):
        plt.subplot(row * 100 + col * 10 + (index + 1))
        plt.title(code)
        plt.plot(y[:, index], label='Predicted')
        plt.plot(label[:, index], label='Real')
        plt.legend(loc='upper left')
    # plt.show()
    plt.savefig(save_path, dpi=200)
