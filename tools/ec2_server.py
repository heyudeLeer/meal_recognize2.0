#  -*- coding=utf-8 -*-


"""
file: service.py
socket service
"""


import socket
import threading
import time
import sys
import os
import subprocess


def socket_service():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 防止socket server重启后端口被占用（socket.error: [Errno 98] Address already in use）
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('0.0.0.0', 6699))
        #s.bind(('127.0.0.1', 6699))
        s.listen(10)
    except socket.error as msg:
        print (msg)
        sys.exit(1)
    print ('Waiting connection...')

    while 1:
        conn, addr = s.accept()
        t = threading.Thread(target=deal_data, args=(conn, addr))
        t.start()

def deal_data(conn, addr):
    print ('Accept new connection from {0}'.format(addr))
    #conn.send('Hi, Welcome to the ec2!')
    while 1:
        data = conn.recv(1024)
        print ('{0} client send data is: {1}'.format(addr, data))

        ret,output = process(cmd=data)

        #if 'exit' == data or not data:
        #    print ('{0} connection close'.format(addr))
        #    conn.send('Connection closed!')
        #    break
        if ret != 0:
            conn.send('err: {0}'.format(output))
        else:
            conn.send('{0}'.format(ret))

        #conn.send('{0}'.format(output))
        break
    conn.close()

def process(cmd=None):
    try:
        res = subprocess.check_output(cmd,
                                      stderr=subprocess.STDOUT,
                                      shell=True)
        #print  'res:', res
        output = res
        return (0,output)
    except subprocess.CalledProcessError, exc:
        #print 'returncode:', exc.returncode
        #print 'cmd:', exc.cmd
        #print 'output:', exc.output
        output = exc.output
        return (1,output)

if __name__ == '__main__':
    socket_service()
    #process(cmd='ls -l split.sh')

