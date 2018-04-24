import paramiko
import sys


def synchronize_model():

    project_dir = '/home/duser/shuyu/Personae'

    cmd = "cd {}; git reset --hard; git pull origin master".format(project_dir)

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
    synchronize_model()
