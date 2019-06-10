
from typing import List
import sys
from pathlib import Path


UNK = "*UNKNOWN*"
OOR2 = "OOR2"
OOR3 = "OOR3"
OOR4 = "OOR4"
START = "*START*"
END = "*END*"
IGNORE = -1
MISS = -2


def sum_by(f, *args):
    return sum(f(*vs) for vs in zip(*args))


def log(msg):
    print("log: ", msg, file=sys.stderr)


def get_suffix(word):
    return [word[-1],
           word[-2:] if len(word) > 1 else OOR2,
           word[-3:] if len(word) > 2 else OOR3,
           word[-4:] if len(word) > 3 else OOR4]


def get_prefix(word):
    return [word[0],
            word[:2] if len(word) > 1 else OOR2,
            word[:3] if len(word) > 2 else OOR3,
            word[:4] if len(word) > 3 else OOR4]


def normalize(word):
    if word == "-LRB-":
        return "("
    elif word == "-RRB-":
        return ")"
    elif word == "-LCB-":
        return "("
    elif word == "-RCB-":
        return ")"
    else:
        return word


def denormalize(word):
    if word == "(":
        return "-LRB-"
    elif word == ")":
        return "-RRB-"
    elif word == "{":
        return "-LCB-"
    elif word == "}":
        return "-RCB-"
    word = word.replace(">", "-RAB-")
    word = word.replace("<", "-LAB-")
    return word


