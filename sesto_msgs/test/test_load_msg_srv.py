#!/usr/bin/env python
import unittest

import rospy
import rostest
import sys
import os
import subprocess
from sesto_msgs import *


class LoadMsg(unittest.TestCase):
    def test_load_msg(self):
        # we will try to import all the msg in sesto_msg/msg
        msg_list = os.listdir(os.path.dirname(__file__) + '/../msg')
        command = ""
        for each_msg in msg_list:
            each_msg = each_msg.rsplit('.', 1)[0]
            command = command + "from sesto_msgs.msg import " + each_msg + "\n"
        command += 'print "OK"'
        # we run a python script via bash
        # sys.stderr.write('\n[LoadMsg] Will run python script via bash: \n' + command)
        output = subprocess.check_output(['bash', '-c', 'python -c \'' + command + '\''])
        sys.stderr.write('\n[LoadMsg] Bash output: ' + output)
        self.assertEqual(output, 'OK\n')


class LoadSrv(unittest.TestCase):
    def test_load_srv(self):
        # we will try to import all the srv in sesto_msg/srv
        msg_list = os.listdir(os.path.dirname(__file__) + '/../srv')
        command = ""
        for each_srv in msg_list:
            each_srv = each_srv.rsplit('.', 1)[0]
            command = command + "from sesto_msgs.srv import " + each_srv + "\n"
        command += 'print "OK"'
        # we run a python script via bash
        # sys.stderr.write('\n[LoadSrv] Will run python script via bash: \n' + command)
        output = subprocess.check_output(['bash', '-c', 'python -c \'' + command + '\''])
        sys.stderr.write('\n[LoadSrv] Bash output: ' + output)
        self.assertEqual(output, 'OK\n')


if __name__ == '__main__':
    rostest.rosrun('test_load_msg_srv', 'load_msg', LoadMsg, sys.argv)
    rostest.rosrun('test_load_msg_srv', 'load_srv', LoadSrv, sys.argv)
