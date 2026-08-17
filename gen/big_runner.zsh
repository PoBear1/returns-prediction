#!/bin/zsh
caffeinate -i python train_runner.py AAPL 9 > output/AAPL_output.txt &
caffeinate -i python train_runner.py AGG 9 > output/AGG_output.txt & 
caffeinate -i python train_runner.py AMZN 9 > output/AMZN_output.txt & 
caffeinate -i python train_runner.py BND 9 > output/BND_output.txt & 
caffeinate -i python train_runner.py ETHA 9 > output/ETHA_output.txt & 
caffeinate -i python train_runner.py GLD 9 > output/GLD_output.txt & 
caffeinate -i python train_runner.py GOOG 9 > output/GOOG_output.txt & 
caffeinate -i python train_runner.py IBIT 9 > output/IBIT_output.txt & 
caffeinate -i python train_runner.py META 9 > output/META_output.txt & 
caffeinate -i python train_runner.py MSFT 9 > output/MSFT_output.txt & 
caffeinate -i python train_runner.py NFLX 9 > output/NFLX_output.txt & 
caffeinate -i python train_runner.py NVDA 9 > output/NVDA_output.txt & 
caffeinate -i python train_runner.py QQQ 9 > output/QQQ_output.txt & 
caffeinate -i python train_runner.py SLV 9 > output/SLV_output.txt & 
caffeinate -i python train_runner.py SPY 9 > output/SPY_output.txt & 
caffeinate -i python train_runner.py TSLA 9 > output/TSLA_output.txt & 
caffeinate -i python train_runner.py VXX 9 > output/VXX_output.txt & 
ps
