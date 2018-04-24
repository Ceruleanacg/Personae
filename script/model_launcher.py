import paramiko
import sys

from helper.args_parser import stock_codes, future_codes


def launch_model():
    # Model Name.
    model_name = 'PolicyGradient'
    # Market Type.
    market = 'stock'
    # Codes.
    codes = stock_codes
    # Start date.
    start = "2008-01-01"
    # End date.
    end = "2018-01-01"
    # Episodes.
    episode = 500
    # Train steps.
    train_steps = 100000
    # Training data ratio.
    training_data_ratio = 0.8

    # Mounted dir.
    mounted_dir = '/home/duser/shuyu/Personae:/app/Personae/'
    image_name = 'ceruleanwang/personae'

    rl_cmd = 'docker run -tv {} --network=quant {} algorithm/RL/'.format(mounted_dir, image_name)
    rl_cmd += "{}.py -c {} -s {} -e {}  --market {} --episode {} --train_steps {} --training_data_ratio {}".format(
        model_name, " ".join(codes), start, end, market, episode, train_steps, training_data_ratio
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
