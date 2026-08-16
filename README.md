# Predicting and generating returns of assets/portfolios of assets

This is a side project of mine that attempts to use more powerful and modern ML models on returns data, with a fair bit of previous data analysis/modelling, to predict the returns of single assets, and as a future goal, to predict the returns of a collection of possibly dependent assets. This was moved from an earlier repo that more general ML because this project grew too big to be a sub-project, rather now it is its own project. 

All of this was run on a MacBook Pro (2024) M4 Max, however CUDA/the CPU also works out-of-the-box (not empirically tested due to hardware constraints, sorry!)

## On the lack of data

This repo does not include the data I used in the experiment because they were too big to put in, rather I am going to leave a hint as to what I used:

1. I used `alpaca-py` as the main client to get market data (both historical and real-time), on the free tier. You can probably guess the quality of the data from that.
2. I used minute-by-minute asset bars, rather than day-to-day or hour-by-hour, and if possible my training/testing data came from the very start of 2020 to the end of the 31st of July, 2026.
3. I used the following assets (no particular reason why, I just wanted to see how well my model performed over each type of asset):
	- Garden-variety stocks (AAPL, AMZN, GOOG, META, MSFT, NFLX, NVDA, TSLA)
	- Bond ETFs (AGG, BND)
	- ETFs on country equities (EWA, EWC, EWD, EWG, EWH, EWI, EWJ, EWK, EWL, EWM, EWN, EWO, EWP, EWQ, EWS, EWT, EWU, EWW, EWY, EWZ)
	- Index ETFs (QQQ, SPY)
	- Volatility ETFs (VXX, VIXY, VIXM)
	- Crypto ETFs (IBIT, ETHA)
	- Commodity ETFs (GLD, SLV)