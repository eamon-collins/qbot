#!/bin/bash

ulimit -c unlimited

LD_LIBRARY_PATH=/home/ecollins/.conda/envs/qenv/lib:$LD_LIBRARY_PATH /home/ecollins/practice/qbot/qbot -b -t 2
