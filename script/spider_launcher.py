import paramiko
import sys

from helper.args_parser import stock_codes, future_codes


def launch_model():
    # Spider name.
    spider_name = 'stock'
    # Codes.
    codes = stock_codes
    # Start date.
    start = "2008-01-01"
    # End date.
    end = "2018-01-01"

    # Mounted dir.
    mounted_dir = '/home/duser/shuyu/Personae:/app/Personae/'
    image_name = 'ceruleanwang/personae'

    rl_cmd = 'docker run -tv {} --network=quant {} spider/'.format(mounted_dir, image_name)
    rl_cmd += "{}_spider.py -c {} -s {} -e {}".format(
        spider_name, " ".join(codes), start, end
    )
    cmd = rl_cmd

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname='192.168.4.199', port=22, username='duser')

    trans = ssh.get_transport()

    channel = trans.open_session()
    channel.get_pty()
    channel.invoke_shell()

    std_in, std_out, std_err = ssh.exec_command(cmd)

    while True:
        line = std_out.readline()
        if line:
            sys.stdout.write(line)
        else:
            break

    ssh.close()


if __name__ == '__main__':
    launch_model()
