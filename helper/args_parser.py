# coding=utf-8

import argparse

spider_parser = argparse.ArgumentParser()
spider_parser.add_argument("-c", "--codes", default=["600036, ""601328", "601998", "601288"], type=list)
spider_parser.add_argument("-s", "--start", default="2008-01-01")
spider_parser.add_argument("-e", "--end", default="2018-01-01")


model_launcher_parser = argparse.ArgumentParser()
model_launcher_parser.add_argument("-n", "--name", default="DDPG")
model_launcher_parser.add_argument("-c", "--codes", default=["600036, ""601328", "601998", "601288"], type=list)
model_launcher_parser.add_argument("-s", "--start_date", default="2008-01-01")
model_launcher_parser.add_argument("-e", "--end_date", default="2018-01-01")