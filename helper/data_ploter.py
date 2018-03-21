# coding=utf-8

import matplotlib.pyplot as plt


def plot_stock_series(codes, y, label, save_path):
    plt.figure(figsize=(30, 10))
    plt.title('Stock Price Series')
    for index, code in enumerate(codes):
        plt.subplot(100 + 10 * len(codes) + index + 1)
        plt.plot(y[:, index], label='Predicted')
        plt.plot(label[:, index], label='Real')
        plt.legend(loc='upper left')
    # plt.show()
    plt.savefig(save_path, dpi=300)
