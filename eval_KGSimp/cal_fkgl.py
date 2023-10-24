#!/usr/bin/python
#-*-coding:utf-8 -*-
#Author   : Xuanli
#Version  : 1.0
#Filename : cal_fkgl.py
from __future__ import print_function

import textstat
import sys


def main(input_file):
    sents = []
    with open(input_file) as reader:
        for sent in reader:
            sents.append(sent.strip())
        print("FKGL:", textstat.flesch_kincaid_grade((" ".join(sents))))


if __name__ == "__main__":
    main(sys.argv[1])

