import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, LinearConstraint, Bounds, BFGS

def plot_equity_curve(tickers, per_weights, start_date, end_date):
    equity_curve = get_equity_curve(tickers, per_weights, start_date, end_date)
    plt.plot(equity_curve)
    plt.show()

def get_equity_curve(tickers, per_weights, start_date, end_date):
    cp = get_closing_prices(tickers, start_date, end_date)
    weighted_cp = cp.values*per_weights.T/cp.values[0,:]
    equity_curve = np.sum(weighted_cp, axis=1)
    return equity_curve

def get_expected_return(tickers, per_weights, start_date, end_date, ndays):
    equity_curve = get_equity_curve(tickers, per_weights, start_date, end_date)
    ecdf = pd.DataFrame({'ec':equity_curve})
    ecdf['P1'] = ecdf['ec'].shift(1)
    ecdf['dr'] = ecdf['ec']/ecdf['P1']-1
    ecdf.dropna(inplace=True)
    return ndays*ecdf['dr'].mean()

def get_closing_prices(tickers, start_date, end_date):
    priceVolData = {}
    for ticker in tickers:
        priceVolData[ticker]= pd.read_csv('HISTORICAL_DATA/'+ticker+'_data.csv', index_col=0)
        priceVolData[ticker] = priceVolData[ticker][(priceVolData[ticker].index>=start_date)&(priceVolData[ticker].index<=end_date)]

    data = pd.DataFrame()
    for ticker in tickers:
        data[ticker] = priceVolData[ticker]['close']

    return data

def get_daily_returns(tickers, start_date, end_date):
    # dates in formate : 'YYYY-MM-DD'
    data = get_closing_prices(tickers, start_date, end_date)
    stocks = data.columns.values

    for i, stock in enumerate(stocks):
        data['P'+str(i+1)] = data[stock].shift(1)
        data[tickers[i]] = data[tickers[i]]/data['P'+str(i+1)]-1
    data = data[[ticker for ticker in tickers]]
    data.dropna(inplace=True)
    return data

def get_portfolio_variance(tickers, per_weights, start_date, end_date):
    equity_curve = get_equity_curve(tickers, per_weights, start_date, end_date)
    ecdf = pd.DataFrame({'ec':equity_curve})
    ecdf['P1'] = ecdf['ec'].shift(1)
    ecdf['dr'] = ecdf['ec']/ecdf['P1']-1
    ecdf.dropna(inplace=True)
    return ecdf['dr'].std()

def portfolio_risk(per_weights, tickers, start_date, end_date):
    return get_portfolio_variance(tickers, per_weights, start_date, end_date)

def optimizeWeights(per_weights, tickers, start_date, end_date):
    n = len(per_weights)
    bounds = Bounds([0 for i in range(n)],[1 for i in range(n)])
    lc = LinearConstraint([[1 for i in range(n)]],[1],[1])
    res = minimize(portfolio_risk, per_weights,args=(tickers, start_date, end_date),method='trust-constr', jac='2-point', hess=BFGS(),constraints=[lc],bounds=bounds)
    return res.x

def get_corr_matrix(tickers, start_date, end_date):
    # dates in formate : 'YYYY-MM-DD'
    data = get_daily_returns(tickers, start_date, end_date)
    return data.corr()

tickers = ['ASHOKLEY','RELIANCE','MARUTI','TATAMOTORS']
start_date = '2016-09-01'
end_date = '2017-09-01'
per_weights = np.array([25,25,25,25])
opt_weights = optimizeWeights(per_weights,tickers,start_date,end_date)
print(opt_weights)
plot_equity_curve(tickers, opt_weights, start_date, end_date)
