import socket
import getpass
from subprocess import Popen, PIPE, call
import re
import os
import sys, getopt
from subprocess import check_output


def parse_output(output):
    result = {}
    for item in output.split("\n"):
        if item == "#verification: 1":
            result["verification"] = True
        if "#perf" in item:
            result["perf"] = item.split(" ")[-1]
    print result
    return result


username = str(getpass.getuser())
print username

average_TEPS = 0
rating_value = 0

tests = [{"params": "--rmat --s 20 --e 32 --check", "weight": 5.0},
         {"params": "--random_uniform --s 21 --e 32 --check", "weight": 4.0},
         {"params": "--random_uniform --s 20 --e 32 --check", "weight": 5.0},
         {"params": "--rmat --s 22 --e 32 --check", "weight": 5.0}]

for test in tests:
    #params = test["params"].split()
    #print(params)
    out = check_output(["./sssp", test["params"]])
    print out
    result = parse_output(out)
    if "verification" not in result:
        average_TEPS = 0
        rating_value = 0
        break
    average_TEPS += float(result["perf"])
    rating_value += float(result["perf"]) * test["weight"]

average_TEPS = average_TEPS / len(tests)
print "avg teps"
print average_TEPS
host = socket.gethostname()
port = 1026                  # The same port as used by the server
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(('graphit.parallel.ru', port))
print "connected"
data_dict = {"user_name": username,
             "average_TEPS": average_TEPS,
             "rating_value": rating_value,
             "submission_time": ""}
print "test"
data = str(data_dict)
s.sendall(data)
print "1"
data = s.recv(1024)
print "2"
s.close()
print "3"
print('Received', repr(data))
