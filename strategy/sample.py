import rqalpha

from rqalpha.api import *
from strategy import config


# 在这个方法中编写任何的初始化逻辑。context对象将会在你的算法策略的任何方法之间做传递。
def init(context):
    context.has_save_data = False


# before_trading此函数会在每天策略交易开始前被调用，当天只会被调用一次
def before_trading(context):
    # if not context.has_save_data:
    au888 = history_bars('AU888', 20000, '1d').reshape((-1, 1))
    print(au888)
    # context.has_save_data = True


# 你选择的证券的数据更新将会触发此段逻辑，例如日或分钟历史数据切片或者是实时数据切片更新
def handle_bar(context, bar_dict):
    pass


# after_trading函数会在每天交易结束后被调用，当天只会被调用一次
def after_trading(context):
    pass


rqalpha.run_func(init=init,
                 before_trading=before_trading,
                 handle_bar=handle_bar,
                 after_trading=after_trading,
                 config=config)
