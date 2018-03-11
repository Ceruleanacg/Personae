# coding=utf-8

import argparse

spider_parser = argparse.ArgumentParser()
spider_parser.add_argument("-c", "--code", default="600036")
spider_parser.add_argument("-s", "--start", default="2008-01-01")
spider_parser.add_argument("-e", "--end", default="2018-01-01")
