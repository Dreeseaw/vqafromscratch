#!/bin/bash

RUN_ID=$1

mkdir -pv "logs/$RUN_ID" && python3 -u train.py "logs/$RUN_ID" | tee "logs/$RUN_ID/logfile.txt"
