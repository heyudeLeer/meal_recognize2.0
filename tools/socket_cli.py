#!/usr/bin/env python
# -*- coding=utf-8 -*-


"""
file: client.py
socket client
"""

import socket
import sys


def socket_client():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(('127.0.0.1', 6699)) # 54.223.39.84
    except socket.error as msg:
        print msg
        sys.exit(1)
    print s.recv(1024)
    while 1:
        data = raw_input('please input work: ')
        s.send(data)
        print s.recv(1024)
        if data == 'exit':
            break
    s.close()

def send(data=None):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(('54.223.39.84', 6699)) # 54.223.39.84
    except socket.error as msg:
        print msg
        sys.exit(1)
    #print s.recv(1024)

    s.send(data)
    output = s.recv(1024)
    s.close()

    if len(output) == 1:
        return (0,output)

    else:
        return (1,output)


if __name__ == '__main__':
    #socket_client()
    send(data='ls -l xx')