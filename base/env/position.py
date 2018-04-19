# coding=utf-8


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
