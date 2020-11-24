import numpy as np
import pandas as pd
from pandas_datareader import data as wb
from scipy.stats import norm, gmean, cauchy
import plotly.express as px
from plotly.offline import plot

def import_stock_data(tickers, start = '2015-1-1'):
    data = pd.DataFrame()
    if len([tickers]) ==1:
        data[tickers] = wb.DataReader(tickers, data_source='yahoo', start = start)['Adj Close']
        data = pd.DataFrame(data)
    else:
        for t in tickers:
            data[t] = wb.DataReader(t, data_source='yahoo', start = start)['Adj Close']
    return(data)

def log_returns(data):
    return (np.log(1+data.pct_change()))

def simple_returns(data):
    return ((data/data.shift(1))-1)

def drift_calc(data, return_type='log'):
    if return_type=='log':
        lr = log_returns(data)
    elif return_type=='simple':
        lr = simple_returns(data)
    u = lr.mean()
    var = lr.var()
    drift = u-(0.5*var)
    try:
        return drift.values
    except:
        return drift

def daily_returns(data, days, iterations, return_type='log'):
    ft = drift_calc(data, return_type)
    if return_type == 'log':
        try:
            stv = log_returns(data).std().values
        except:
            stv = log_returns(data).std()
    elif return_type=='simple':
        try:
            stv = simple_returns(data).std().values
        except:
            stv = simple_returns(data).std()    
    #Oftentimes, we find that the distribution of returns is a variation of the normal distribution where it has a fat tail
    # This distribution is called cauchy distribution
    dr = np.exp(ft + stv * norm.ppf(np.random.rand(days, iterations)))
    return dr

def probs_find(predicted, higherthan, on = 'value'):
    if on == 'return':
        predicted0 = predicted.iloc[0,0]
        predicted = predicted.iloc[-1]
        predList = list(predicted)
        over = [(i*100)/predicted0 for i in predList if ((i-predicted0)*100)/predicted0 >= higherthan]
        less = [(i*100)/predicted0 for i in predList if ((i-predicted0)*100)/predicted0 < higherthan]
    elif on == 'value':
        predicted = predicted.iloc[-1]
        predList = list(predicted)
        over = [i for i in predList if i >= higherthan]
        less = [i for i in predList if i < higherthan]
    else:
        print("'on' must be either value or return")
    return (len(over)/(len(over)+len(less)))

def monte_carlo_simulation(data, days, iterations=3000):
    ticks = list(data.columns)
    #Log returns
    lr = np.log(1+data.pct_change())
    # Drift Cont
    mu = lr.mean()
    sig = lr.var()
    drift = mu - (0.5*sig)
    # Standard Dev
    stv = lr.std()
    # Generated Daily Returns
    dr = pd.DataFrame()
    stats = pd.DataFrame(index=['Days',
                                'Last Closing Price', 
                                f"Expected Stock Price After {days} Days",
                                f"Expected Return After {days} days",
                                'Probability of At Least Breakeven'])
    for i in ticks:
        last_price = data.iloc[-1][i]
        ret = np.exp(drift[i] + stv[i] * norm.ppf(np.random.rand(days+1, iterations)))
        price_list = np.zeros_like(ret)
        # Put the last actual price in the first row of matrix. 
        price_list[0] = data.iloc[-1][i]
        # Calculate the price of each day
        for t in range(1, days+1):
            price_list[t] = price_list[t-1]*ret[t]

        s_returns = pd.DataFrame()
        s_returns[i] = pd.DataFrame(price_list).iloc[-1]/pd.DataFrame(price_list).iloc[0]
        s_returns.loc[s_returns[i]<0 ,i] = 0

        dr = pd.concat([dr, s_returns], axis=1)

        # Create dataframe with stats
        plist = pd.DataFrame(price_list)
        stats[i] = [f"{days} days",
                    f"${round(last_price,2)}",
                    f"${round(plist.iloc[-1].mean(),2)}",
                    f"{round(100*(plist.iloc[-1].mean()-price_list[0,1])/plist.iloc[-1].mean(),2)}%",
                    f"{round(probs_find(plist,0, on='return')*100,2)}%"]
        
    return dr, stats


def get_probabilities_per_return_graph(returns_df):
    mn = min(returns_df.min())
    mn = min(round(mn, 1), round(mn-0.1, 1))
    mean = max(returns_df.mean())
    sdev = max(returns_df.std())
    mx = min(mean+(3*sdev), 6)
    

    probabilities = pd.DataFrame()
    ret_range = list(np.arange(mn, mx, 0.01))
    for c in returns_df.columns:
        s_d = returns_df[c].to_list()
        prob = []
        for i in ret_range:
            prob.append((len([j for j in s_d if j > i])/len(s_d)))
        probabilities[c] = prob
    probabilities['Return (%)'] = ret_range
    probabilities = probabilities.set_index('Return (%)')
    # mx = min(probabilities.stack()[probabilities.stack()<0.0001].reset_index(level=1, drop=True).index.to_list())
    probabilities = probabilities[probabilities.index <= mx]
    probabilities.index = (probabilities.index-1) * 100
    
    fig = px.line(probabilities, labels={"value":"Probability","variable":"Stock"})
    fig.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="right",
        x=0.99
    ), legend_title_text="")
    fig.update_layout(title='"At Least" Probability versus Return (%) per Stock')
#     fig.to_html()
    return fig

def multiple_monte_carlo(tickers, days):
    tickers = [i.replace(".","-") for i in tickers]
    data = import_stock_data(tickers)
    dr, stats = monte_carlo_simulation(data, days)
    fig = get_probabilities_per_return_graph(dr)
    html = plot(fig, include_plotlyjs=False, output_type='div')
    html = html.replace('style="height:100%; width:100%;"', "")
    return html, stats.to_dict('split')

# def simulate_mc(data, days, iterations, return_type='log', plot=True):
#     # Generate daily returns
#     returns = daily_returns(data, days, iterations, return_type)
#     # Create empty matrix
#     price_list = np.zeros_like(returns)
#     # Put the last actual price in the first row of matrix. 
#     last_price = data.iloc[-1].values[0]
#     price_list[0] = data.iloc[-1]
#     # Calculate the price of each day
#     for t in range(1,days):
#         price_list[t] = price_list[t-1]*returns[t]
    
#     s_returns = pd.DataFrame(price_list).iloc[-1]/pd.DataFrame(price_list).iloc[0]

#     fig = px.histogram(s_returns)
            
#     plist = pd.DataFrame(price_list)
#     info = {"Stock":[data.columns.values[0]],
#             "Last Closing Price":[f"${round(last_price,2)}"],
#             "Days": [f"{days} days"],
#             f"Expected Stock Price After {days} Days":[f"${round(plist.iloc[-1].mean(),2)}"],
#             f"Expected Return After {days} days":[f"{round(100*(plist.iloc[-1].mean()-price_list[0,1])/plist.iloc[-1].mean(),2)}%"],
#             "Probability of Breakeven":[f"{round(probs_find(plist,0, on='return')*100,2)}%"]}
       
#     return fig, info

# def single_monte_carlo(ticker, days, iterations=10000):
#     data = import_stock_data(ticker)
#     fig, info =  simulate_mc(data, days=days, iterations=iterations)
#     html = plot(fig, include_plotlyjs=False, output_type='div')
#     return html, info

# def multiple_monte_carlo(tickers, days=252, iterations=10000):
#     if type(tickers)!=list:
#         tickers = [tickers]
    