# coding=utf-8

from mongoengine import Document
from mongoengine import StringField, FloatField, DateTimeField


class Stock(Document):
    # 股票代码
    code = StringField(required=True)
    # 交易日
    date = DateTimeField(required=True)
    # 开盘价
    open = FloatField()
    # 最高价
    high = FloatField()
    # 最低价
    low = FloatField()
    # 收盘价
    close = FloatField()
    # 成交量
    volume = FloatField()
    # 成交金额
    amount = FloatField()
    # 涨跌幅
    p_change = FloatField()
    # 价格变动
    price_change = FloatField()
    # 5日均价
    ma5 = FloatField()
    # 10日均量
    ma10 = FloatField()
    # 20日均量
    ma20 = FloatField()
    # 5日均量
    v_ma5 = FloatField()
    # 10日均量
    v_ma10 = FloatField()
    # 20日均量
    v_ma20 = FloatField()
    # 换手率
    turnover = FloatField()

    meta = {
        'indexes': [
            'code',
            'date',
            ('code', 'date')
        ]
    }

    def save_if_need(self):
        return self.save() if len(self.__class__.objects(code=self.code, date=self.date)) < 1 else None

    def to_state(self):
        stock_dic = self.to_mongo()
        stock_dic.pop('_id')
        stock_dic.pop('code')
        stock_dic.pop('date')
        return stock_dic.values()

    @classmethod
    def get_k_data(cls, code, start, end):
        return cls.objects(code=code, date__gte=start, date__lte=end).order_by('date')

