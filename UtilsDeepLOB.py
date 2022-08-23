import pandas as pd
import ast
import numpy as np
from datetime import timedelta

def string_to_nplist(x):
    if pd.isnull(x):
        return []
    else:
        return np.array(ast.literal_eval(x))

def read_data(file_name, col_names, col_list):
    data = pd.read_csv(file_name, names=col_names, delimiter="|")
    for col in col_list:
        data[col] = data[col].apply(lambda x: string_to_nplist(x))
    data["DateTime"] = pd.to_datetime(data["DateTime"])
    data.set_index(["DateTime"], inplace=True)
    return data

def read_data_mbo(file_name, col_names, col_list):
    data = pd.read_csv(file_name, names=col_names, delimiter="|")
    for col in col_list:
        data[col] = data[col].apply(lambda x: string_to_nplist(x))
    data["DateTime"] = pd.to_datetime(data["DateTime"])
    data.set_index(["DateTime"], inplace=True)
    data = convert_MBO_index(data)
    data = data.groupby(level=0).filter(lambda x: len(x) == 1)
    return data

def clean_lob(data, weight_mid_price=0.5, cols_need=["BidPrices","BidVolumes","AskPrices","AskVolumes"], num_level=10):
    lst_valid_samples = []
    mid_prices = []
    valid_bid_prices = []
    valid_bid_volumes = []
    valid_ask_prices = []
    valid_ask_volumes = []
    for ind, row in data.iterrows():
        if len(row["BidPrices"]) and len(row["AskPrices"]):
            if (row["BidPrices"].shape[0] >= num_level) and (row["AskPrices"].shape[0] >= num_level):
                lst_valid_samples.append(ind)
                valid_bid_prices.append(row["BidPrices"][:num_level])
                valid_bid_volumes.append(row["BidVolumes"][:num_level])
                valid_ask_prices.append(row["AskPrices"][:num_level])
                valid_ask_volumes.append(row["AskVolumes"][:num_level])
                mid_p = weight_mid_price * row["BidPrices"][0] + (1 - weight_mid_price) * row["AskPrices"][0]
                mid_prices.append(mid_p)
    ret_data = pd.DataFrame(index=lst_valid_samples, data=data.loc[lst_valid_samples, cols_need])
    ret_data["Midprice"] = mid_prices
    ret_data["BidPrices"] = valid_bid_prices
    ret_data["BidVolumes"] = valid_bid_volumes
    ret_data["AskPrices"] = valid_ask_prices
    ret_data["AskVolumes"] = valid_ask_volumes
    return ret_data

def clean_mbo(data, weight_mid_price=0.5, cols_need=["BidPrices","BidVolumes","AskPrices","AskVolumes"], num_level=10):
    lst_valid_samples = []
    mid_prices = []
    valid_bid_prices = []
    valid_bid_volumes = []
    valid_ask_prices = []
    valid_ask_volumes = []
    for ind, row in data.iterrows():
        if len(row["BidPrices"]) and len(row["AskPrices"]):
            if (row["BidPrices"].shape[0] >= num_level) and (row["AskPrices"].shape[0] >= num_level):
                lst_valid_samples.append(ind)
                valid_bid_prices.append(row["BidPrices"][:num_level])
                valid_bid_volumes.append(row["BidVolumes"][:num_level])
                valid_ask_prices.append(row["AskPrices"][:num_level])
                valid_ask_volumes.append(row["AskVolumes"][:num_level])
                mid_p = weight_mid_price * row["BidPrices"][0] + (1 - weight_mid_price) * row["AskPrices"][0]
                mid_prices.append(mid_p)
    ret_data = pd.DataFrame(index=lst_valid_samples, data=data.loc[lst_valid_samples, cols_need])
    ret_data["Midprice"] = mid_prices
    ret_data["BidPrices"] = valid_bid_prices
    ret_data["BidVolumes"] = valid_bid_volumes
    ret_data["AskPrices"] = valid_ask_prices
    ret_data["AskVolumes"] = valid_ask_volumes
    return ret_data
        
    
def func_cc(x):
    ret = np.concatenate((x.ZscoreAskPrices, x.ZscoreAskVolumes, x.ZscoreBidPrices, x.ZscoreBidVolumes))
    return ret
    
def zscore_nomalization(data, freq="1D", min_periods=60):
    data["AvgBidPrices"] = data["BidPrices"].apply(lambda x: np.mean(x))
    data["AvgBidVolumes"] = data["BidVolumes"].apply(lambda x: np.mean(x))
    data["AvgAskPrices"] = data["AskPrices"].apply(lambda x: np.mean(x))
    data["AvgAskVolumes"] = data["AskVolumes"].apply(lambda x: np.mean(x))
    data["MuBidPrice"] = data["AvgBidPrices"].rolling(window=freq, min_periods=min_periods).mean()
    data["STDBidPrice"] = data["AvgBidPrices"].rolling(window=freq, min_periods=min_periods).std()
    data["MuBidVolume"] = data["AvgBidVolumes"].rolling(window=freq, min_periods=min_periods).mean()
    data["STDBidVolume"] = data["AvgBidVolumes"].rolling(window=freq, min_periods=min_periods).std()
    data["MuAskPrice"] = data["AvgAskPrices"].rolling(window=freq, min_periods=min_periods).mean()
    data["STDAskPrice"] = data["AvgAskPrices"].rolling(window=freq, min_periods=min_periods).std()
    data["MuAskVolume"] = data["AvgAskVolumes"].rolling(window=freq, min_periods=min_periods).mean()
    data["STDAskVolume"] = data["AvgAskVolumes"].rolling(window=freq, min_periods=min_periods).std()
    data["ZscoreBidPrices"] = (data["BidPrices"] - data["MuBidPrice"]) / data["STDBidPrice"]
    data["ZscoreBidVolumes"] = (data["BidVolumes"] - data["MuBidVolume"]) / data["STDBidVolume"]
    data["ZscoreAskPrices"] = (data["AskPrices"] - data["MuAskPrice"]) / data["STDAskPrice"]
    data["ZscoreAskVolumes"] = (data["AskVolumes"] - data["MuAskVolume"]) / data["STDAskVolume"]
    data["ConcatLOB"] =  data[["ZscoreAskPrices", "ZscoreAskVolumes", "ZscoreBidPrices", "ZscoreBidVolumes"]].apply(lambda x: func_cc(x), axis=1)
    
def zscore_nomalization_mbo(data, freq=500):
    data["AvgBidPrices"] = data["BidPrices"].apply(lambda x: np.mean(x))
    data["AvgBidVolumes"] = data["BidVolumes"].apply(lambda x: np.mean(x))
    data["AvgAskPrices"] = data["AskPrices"].apply(lambda x: np.mean(x))
    data["AvgAskVolumes"] = data["AskVolumes"].apply(lambda x: np.mean(x))
    data["MuBidPrice"] = data["AvgBidPrices"].rolling(window=freq).mean()
    data["STDBidPrice"] = data["AvgBidPrices"].rolling(window=freq).std()
    data["MuBidVolume"] = data["AvgBidVolumes"].rolling(window=freq).mean()
    data["STDBidVolume"] = data["AvgBidVolumes"].rolling(window=freq).std()
    data["MuAskPrice"] = data["AvgAskPrices"].rolling(window=freq).mean()
    data["STDAskPrice"] = data["AvgAskPrices"].rolling(window=freq).std()
    data["MuAskVolume"] = data["AvgAskVolumes"].rolling(window=freq).mean()
    data["STDAskVolume"] = data["AvgAskVolumes"].rolling(window=freq).std()
    data["ZscoreBidPrices"] = (data["BidPrices"] - data["MuBidPrice"]) / data["STDBidPrice"]
    data["ZscoreBidVolumes"] = (data["BidVolumes"] - data["MuBidVolume"]) / data["STDBidVolume"]
    data["ZscoreAskPrices"] = (data["AskPrices"] - data["MuAskPrice"]) / data["STDAskPrice"]
    data["ZscoreAskVolumes"] = (data["AskVolumes"] - data["MuAskVolume"]) / data["STDAskVolume"]
    data["ConcatLOB"] =  data[["ZscoreAskPrices", "ZscoreAskVolumes", "ZscoreBidPrices", "ZscoreBidVolumes"]].apply(lambda x: func_cc(x), axis=1)


def get_Daily_Volatility(close,span0=20):
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    #df0 = close.index.searchsorted(close.index - 1)
    df0 = df0[df0 > 0]
    df0 = (pd.Series(close.index[df0 - 1], index=close.index[close.shape[0] - df0.shape[0]:]))
    df0 = close.loc[df0.index] / close.loc[df0.array].array - 1  # daily returns
    df0 = df0.ewm(span=span0).std()
    return df0

def get_3_barriers(daily_volatility, prices, upper_lower_multipliers, t_final):
    #create a container
    barriers = pd.DataFrame(columns=['days_passed', 
              'price', 'vert_barrier', 'top_barrier', 'bottom_barrier'],
              index = daily_volatility.index)
    for day, vol in daily_volatility.iteritems():
        #set the vertical barrier
        days_passed = len(daily_volatility.loc[daily_volatility.index[0] : day])
        if (days_passed + t_final < len(daily_volatility.index)and t_final != 0):
            vert_barrier = daily_volatility.index[days_passed + t_final]
        else:
            vert_barrier = np.nan        #set the top barrier
        if upper_lower_multipliers[0] > 0:
            top_barrier = prices.loc[day] + prices.loc[day] * upper_lower_multipliers[0] * vol
        else:
            #set it to NaNs
            top_barrier = pd.Series(index=prices.index)        #set the bottom barrier
        if upper_lower_multipliers[1] > 0:
            bottom_barrier = prices.loc[day] - prices.loc[day] * upper_lower_multipliers[1] * vol
        else: 
            #set it to NaNs
            bottom_barrier = pd.Series(index=prices.index)
        barriers.loc[day, ['days_passed', 'price', 'vert_barrier','top_barrier', 'bottom_barrier']] = \
        days_passed, prices.loc[day], vert_barrier, top_barrier, bottom_barrier
    return barriers

def get_labels(barriers):
    '''
    start: first day of the window
    end:last day of the window
    price_initial: first day stock price
    price_final:last day stock price
    top_barrier: profit taking limit
    bottom_barrier:stop loss limt
    condition_pt:top_barrier touching conditon
    condition_sl:bottom_barrier touching conditon'''
    
    barriers['DePrado'] = None
    copy_barriers = barriers.copy().reset_index()
    for i in range(len(barriers.index)):
        start = barriers.index[i]
        #end = barriers.reset_index().vert_barrier[i]############################
        end = copy_barriers.vert_barrier[i]
        if pd.notna(end):
            # assign the initial and final price
            price_initial = barriers.price[start]
            price_final = barriers.price[end]           
            # assign the top and bottom barriers
            #top_barrier = barriers.reset_index().top_barrier[i]
            #bottom_barrier = barriers.reset_index().bottom_barrier[i]     
            top_barrier = copy_barriers.top_barrier[i]
            bottom_barrier = copy_barriers.bottom_barrier[i]
            #set the profit taking and stop loss conditons
            condition_pt = (barriers.price[start: end] >= top_barrier).any()
            condition_sl = (barriers.price[start: end] <= bottom_barrier).any()
            #assign the labels
            if condition_pt: 
                barriers['DePrado'][i] = 1
            elif condition_sl: 
                barriers['DePrado'][i] = -1    
            else: 
                barriers['DePrado'][i] = max(
                          [(price_final - price_initial)/ 
                           (top_barrier - price_initial), \
                           (price_final - price_initial)/ \
                           (price_initial - bottom_barrier)],\
                            key=abs)
    return barriers


def deprado(data):
    volatility = get_Daily_Volatility(data.Midprice)   
    barriers = get_3_barriers(volatility, data.Midprice, upper_lower_multipliers=[2,2], t_final=10)
    labels = get_labels(barriers)
    data['key'] = data.index
    labels['key'] = labels.index
    return data.merge(labels, on='key', how='left')


def convert_MBO_index(data):  
    i = 0
    data['time'] = data.index
    while i < len(data):
      current_time = data['time'][i]
      j = i + 1
      count = 1
      while j < len(data) and data['time'][j] == current_time:
        count += 1
        j += 1
      milliseconds = 1000 // count
      for k in range(i, j):
          data['time'][k] = data['time'][k] + timedelta(milliseconds=(k-i)*milliseconds)
      i = j
      data.index = data['time']
    return data
    

def main(data_path):
    pass
    
if __name__ == "__main__":
    main()
