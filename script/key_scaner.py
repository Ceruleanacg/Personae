# coding=utf-8

import sys
import os


if __name__ == '__main__':

    start_ip, end_ip = sys.argv[1], sys.argv[2]

    split_start_ip = start_ip.split('.')
    split_end_ip = end_ip.split('.')

    ip_list_str = ""

    ip_base = split_start_ip[0] + '.' + split_start_ip[1] + '.' + split_start_ip[2] + '.'

    ip_count = int(split_end_ip[-1]) - int(split_start_ip[-1]) + 1

    for ip_index in range(ip_count):
        ip_list_str += ip_base + str(int(split_start_ip[3]) + ip_index) + " "

    cmd_1 = "ssh-keyscan -t rsa %s" % ip_list_str
    os.system("%s > ~/.ssh/known_hosts" % cmd_1)
