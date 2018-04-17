# coding=utf-8

import argparse

codes = ["600036", "601328", "601998", "601398"]
# codes = ["600036", "601328"]

spider_parser = argparse.ArgumentParser()
spider_parser.add_argument("-c", "--codes", default=codes, nargs="+")
spider_parser.add_argument("-s", "--start", default="2008-01-01")
spider_parser.add_argument("-e", "--end", default="2018-01-01")


model_launcher_parser = argparse.ArgumentParser()
model_launcher_parser.add_argument("-n", "--name", default="DoubleDQN")
model_launcher_parser.add_argument("-c", "--codes", default=codes, nargs="+")
model_launcher_parser.add_argument("-s", "--start", default="2008-01-01")
model_launcher_parser.add_argument("-e", "--end", default="2018-01-01")
model_launcher_parser.add_argument("--mode", default="train")
model_launcher_parser.add_argument("--episode", default=200, type=int)
