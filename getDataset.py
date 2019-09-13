import pandas as pd
def get_historical_data(symbol):
    ''' Alphavantage api  '''
    symbol = symbol.upper()
    # American Airlines stock market prices

    api_key = 'WRT04EFZKZG0W8WU'
    # JSON file with all the stock market data for AAL from the last 20 years
    url_string = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%s&datatype=csv&outputsize=full&apikey=%s"%(symbol,api_key)
    
    col_names = ['Date','Open','High','Low','Close','Volume']
    stocks = pd.read_csv(url_string, header=0, names=col_names) 
    
    df = pd.DataFrame(stocks)
    return df
ticker=input('enter stock ticker')
str1=ticker+'.csv'


data = get_historical_data(ticker)
'''
import os
if str1 not in os.listdir():
    data = get_historical_data(ticker) # from January 1, 2005 to June 30, 2017
else:
    data=pd.DataFrame(pd.read_csv(str1))
'''

#Write the data to a csv file.
data.to_csv('dataset/'+'NSE_'+ticker[4:]+'.csv',index = False)#NSE:INFY NSE:SBIN
