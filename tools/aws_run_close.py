#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import time
import argparse
import subprocess
import socket_cli as sc
import ec2_server as ec2

def clean(output=None):
    #ec2.process('scp gpu:/home/ubuntu/ec2_server/ec2_server.log ~/;cat ~/ec2_server.log')
    print output
    ret, output = ec2.process('aws ec2 stop-instances  --instance-ids i-01903079ea58c34c2')
    if ret != 0:
        print '\033[1;32m error:aws ec2 is running,please stop it by yourself!!!\033[0m'
        print output
    exit(ret)

def stop_ec2():
    ret, output = ec2.process('aws ec2 stop-instances  --instance-ids i-01903079ea58c34c2')
    if ret != 0:
        print '\033[1;32m error:aws ec2 is running,please stop it by yourself!!!\033[0m'
        print output


parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--exe', type=str, default = None)
parser.add_argument('--init', type=bool, default= False)
args = parser.parse_args()
if args.exe is None:
    print "no code to run..."
    exit(0)
else:
    print 'run file is: ' + args.exe


#os.system('git add .')
#os.system('git commit --amend --no-edit')
ret,output = ec2.process('git add .')
if ret != 0:
    print output
    exit(ret)
ret,output = ec2.process('git commit --amend --no-edit')
if ret != 0:
    print output
    exit(ret)

ret,output = ec2.process('git symbolic-ref --short HEAD')
if ret != 0:
    print output
    exit(ret)
branch = output
print "branch is: " + branch

cwd = os.getcwd()
project_name = os.path.basename(cwd)
print "project is: " + project_name

ret,output=ec2.process('aws ec2 start-instances  --instance-ids i-01903079ea58c34c2')
if ret != 0:
    print output
    exit(ret)
print 'waite ec2 starting'
#time.sleep(28)
#while(True):
#    os.popen('aws ec2 describe-instances  --instance-ids i-01903079ea58c34c2')
#    time.sleep(1)

if args.init is True:
    print 'create project on aws ec2 at the beginning'
    ret,output = sc.send('cd /home/ubuntu;mkdir %s;cd %s;mkdir bare;cd bare;git init --bare' % (project_name,project_name))
    if ret is not 0:
        clean(output)
    else:
        print 'success...'

    print 'push code...'
    ret,output = ec2.process('git remote add aws_ec2 gpu:/home/ubuntu/%s/bare;git push aws_ec2' % (project_name))
    if ret is not 0:
        clean(output)
    else:
        print 'success...'

    print 'pull code...'
    ret,out = sc.send('cd /home/ubuntu/%s;git clone bare src;cd src;git pull origin %s' % (project_name,branch))
    if ret is not 0:
        clean(output)
    else:
        print 'success...'

    print 'install...'
    ret,output = sc.send('cd /home/ubuntu/%s/src;python setup.py install' % (project_name))
    if ret is not 0:
        clean(output)
    else:
        print 'success...'

    print 'exe code...'
    ret,output = sc.send("cd /home/ubuntu/%s/src;python %s" % (project_name,args.exe))
    if ret is not 0:
        clean(output)
    else:
        print 'success...'

else:
    print 'git push...'
    ret, output = ec2.process('git push aws_ec2 -f')
    if ret != 0:
        clean(output)
    else:
        print 'success...'

    print 'git fetch and reset...'
    ret, output = sc.send('cd /home/ubuntu/%s/src;git fetch --all;git reset --hard origin/%s' % (project_name,branch))
    if ret is not 0:
        clean(output)
    else:
        print 'success...'

    print 'exe code...'
    ret, output = sc.send('cd /home/ubuntu/%s/src;pwd;python %s' % (project_name,args.exe))
    if ret is not 0:
        clean(output)
    else:
        print 'success...'

stop_ec2()







