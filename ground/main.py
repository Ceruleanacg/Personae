# coding=utf-8

from spider.finance import StockSpider


spider = StockSpider(code="601398", start="2008-01-01", end="2018-01-01")
spider.crawl()
