# -*- coding: utf-8 -*-
"""
Spyder Editor

Este é um arquivo de script temporário.
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 20:42:20 2019

@author: dfcas
"""
import plotly as py
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from datetime import datetime
from plotly import graph_objs as go
from plotly import tools
from talib import abstract

SMA = abstract.SMA
SAR = abstract.SAR

df1 = pd.read_csv('.\EURUSD\EURUSD1.csv',names=['date','hour','open','high','low','close','volume'],parse_dates=[['date', 'hour']])
df1.date_hour=pd.to_datetime(df1.date_hour, format='%d.%m.%Y %H:%M:%S.%f')
df1.set_index(df1.date_hour)
graphic1MCandle = go.Figure(data=[go.Candlestick(x=df1['date_hour'],open=df1['open'],high=df1['high'],low=df1['low'],close=df1['close'],increasing_line_color= 'blue', decreasing_line_color= 'red')])
sma5 = SMA(df1, timeperiod=5, price='close')
sma15 = SMA(df1, timeperiod=15, price='close')
sma60 = SMA(df1, timeperiod=60, price='close')
sma89 = SMA(df1, timeperiod=89, price='close')

df1.insert(5, "SMA5", sma5, True) 
df1.insert(5, "SMA15", sma15, True)
df1.insert(5, "SMA60", sma60, True)
df1.insert(5, "SMA89", sma89, True) 

movimento = []
tp = []
sl = []

columns = list(df1)
j=0
top = len(df1)

try:
    for i in range(top):
        j = i+14
        high=df1.loc[i:j,"high"].values
        low=df1.loc[i:j,"low"].values
        localmax = max(high)
        localmin = min(low)

        if df1.loc[j,"close"] > df1.loc[i,"close"]:
            movimento.append(1)
            tp.append(localmax)
            sl.append(localmin)
        elif df1.loc[j,"close"] < df1.loc[i,"close"]:
            movimento.append(-1)
            tp.append(localmin)
            sl.append(localmax)
        else:
            movimento.append(0)
            tp.append(0)
            sl.append(0)            
except KeyError:
    pass

comp = len(movimento)
print('Tamannho do TP: '+ str(len(tp)))
print('Tamannho do SL: '+ str(len(sl)))
print('Tamannho do MOVIMENTO: '+ str(comp))

for i in range(top - comp):
    movimento.insert(0, np.nan)
    tp.insert(0, np.nan)
    sl.insert(0, np.nan)   

df1.insert(5, "ORDER", movimento, True)
del i
del j
del comp
del high
del low
del localmax
del localmin

df1.insert(5, "TP", tp, True) 
df1.insert(5, "SL", sl, True)

"Colocar em um mesmo DAtaset todas colunas de data e OHLC para 1M, 5M, 15M, 1H, 4H e D"
"Processar SMA, SAR e ATR para 1M, 5M, 15M, 1H, 4H e D"

"""
- Para cada 15 valores eu devo identificar o maior e o menor da série para com isso saber o TP e SL da ordem


py.offline.plot(graphic1MCandle, filename='tutorial1M.html')"""


