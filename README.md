# Bubble Predictor
When I started investing in August 2025, it seemed the online community was divided between whether federal rate cuts would be great news for stocks or if a lack of consumer spending would cause a market crash. There was also plenty of discourse about whether or not an AI bubble existed. 
I quickly found myself engrossed in what caused major bubble crashes in 2007 with the housing market and 2000 with the dot com hype. I wanted to find out if I was buying in at an incredible time in the market, or if everything that goes up must come down. This is what gave me the inspiration to create a model to forecast the risk of a bubble in the market and to determine when exactly it would pop.

What makes this undertaking unique is that it utilizes market data across major indices and market ETFs alongside sentimental analyses of news headlines on that day, a combination that to my knowledge has not been used in training a financial forecast model, especially not one specifically aimed at predicting market volatility and the chance of a crash. The model's primary objective is to assign a rating from 0.0 to 2.0 on the chance of a bubble existing and popping in the next year or month. The model also has the ability (although not nearly as accurate) to predict the next week's percentage change for major sectors and indices, as well as the VIX of the following week of trading.

# Datasets
I pulled from the incredible FNSPID dataset of news headlines from 1999 to 2000, and supplemented it with market data from yfinance and more articles scraped using Tiingo's API. I used TextBlob for my headline sentiment analysis, and trained models using Google Vertex AI to train Time Series Dense Autoencoders (TIDE) for predicting off of the time range of 2007-2014, a period of recession and bear markets with high volatility, and 2020-2025, which started with the COVID crash and would reflect modern markets better. I also trained a simple Random Forest Regression model with scikit-learn. 

# Results
Bubble risk was assigned to training data as 0 for low risk, 1 for medium risk, and 2 for high risk. In classification, this was a float in that range. I assigned bubble risk as 1.0 for a year before major crashes and a month before minor crashes. I set risk as 2.0 for a month before major crashes and a week before minor crashes. I set it as 0.0 for all else.

2007 Model:
| Metric | Value        |
|--------|--------------|
| MAE    | 0.295        |
| MAPE   | 132,665,580  |
| RMSE   | 0.476        |
| RMSLE  | 0.28         |
| R²     | 0.398        |

2020 Model:
| Metric | Value        |
|--------|--------------|
| MAE    | 0.262        |
| MAPE   | 113,815,790  |
| RMSE   | 0.468        |
| RMSLE  | 0.246        |
| R²     | 0.609        |

2020 Model (Random Forest):
Test RMSE: 0.1775, Test R2: 0.9114

2020 (next week's SPY change prediction):
| Metric | Value        |
|--------|--------------|
| MAE    | 1.529        |
| MAPE   | 129,066      |
| RMSE   | 2.181        |
| RMSLE  | 0.644        |
| R²     | 0.37         |


Consensus between models: 
As of September 14, 2025, our models are at a consensus that we are around 1.0 in bubble risk with about a 20% deviation in either direction. This means that we are either at least a month out from a major or minor crash, and its probable that the AI bubble concerns and concerns about a market crash do have weight! Only time will tell if the predictions are correct!

# Conclusions
Overall, the project was a lot of fun to do! The importance the model placed on specific industries was particularly interesting. It seems that the most important predictive indicator was energy and health care sector ETFs being very low in ratio to SPY, which was rather interesting as its not the first indicator one would think of when trying to predict a bubble. High trading in technology, high volume in consumer and retail sectors, high QQQ to SPY ratio, and high VIX were also indicators the model placed heavy weight on.
The RMSE was quite good for a financial forecasting model, so I'm quite happy with the results. Even if I can't fully base my trading patterns off this model, I feel like I've learned more about macroeconomics and can go into future investments with the understanding that a market crash may indeed be on the horizon!

@misc{dong2024fnspid,
      title={FNSPID: A Comprehensive Financial News Dataset in Time Series}, 
      author={Zihan Dong and Xinyu Fan and Zhiyuan Peng},
      year={2024},
      eprint={2402.06698},
      archivePrefix={arXiv},
      primaryClass={q-fin.ST}
}
