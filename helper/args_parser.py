# coding=utf-8

import argparse

stock_codes = ["600036", "601328", "601998", "601398"]
future_codes = ["AU88", "RB88", "CU88", "AL88"]

stock_spider_parser = argparse.ArgumentParser()
stock_spider_parser.add_argument("-c", "--codes", default=stock_codes, nargs="+")
stock_spider_parser.add_argument("-s", "--start", default="2008-01-01")
stock_spider_parser.add_argument("-e", "--end", default="2018-01-01")

future_spider_parser = argparse.ArgumentParser()
future_spider_parser.add_argument("-c", "--codes", default=future_codes, nargs="+")
future_spider_parser.add_argument("-s", "--start", default="2008-01-01")
future_spider_parser.add_argument("-e", "--end", default="2018-01-01")

model_launcher_parser = argparse.ArgumentParser()
model_launcher_parser.add_argument("-n", "--name", default="DoubleDQN")
model_launcher_parser.add_argument("-c", "--codes", default=stock_codes, nargs="+")
model_launcher_parser.add_argument("-s", "--start", default="2008-01-01")
model_launcher_parser.add_argument("-e", "--end", default="2018-01-01")
model_launcher_parser.add_argument("--mode", default="train")
model_launcher_parser.add_argument("--market", default="stock")
model_launcher_parser.add_argument("--episode", default=500, type=int)
model_launcher_parser.add_argument("--train_steps", default=100000, type=int)
model_launcher_parser.add_argument("--training_data_ratio", default=0.8, type=float)
