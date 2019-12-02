import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
from plotly import graph_objs as go
from plotly import tools
from talib import abstract

SMA = abstract.SMA
SAR = abstract.SAR


pd.set_option('display.max_columns', 30)

isBusinessDay = BDay().onOffset
candle5MinOPEN = []
candle5MinHIGH = []
candle5MinLOW = []
candle5MinCLOSE = []

candle15MinOPEN = []
candle15MinHIGH = []
candle15MinLOW = []
candle15MinCLOSE = []

candle30MinOPEN = []
candle30MinHIGH = []
candle30MinLOW = []
candle30MinCLOSE = []

candle60MinOPEN = []
candle60MinHIGH = []
candle60MinLOW = []
candle60MinCLOSE = []

candle5MinVOL = []


def cria_velas_para_tfs_superiores(df1minute):
    cria_velas_5mins(df1minute)
    cria_velas_15mins(df1minute)
    cria_velas_30mins(df1minute)
    cria_velas_60mins(df1minute)


def cria_velas_5mins(df1minute):
    top = len(df1minute)
    tempo = 0
    low = np.nan
    high = np.nan
    open = np.nan
    close = np.nan
    volume = np.nan
    for i in range(top):
        tempo = df1.loc[i, 'hour']
        if i+4 < top:
            if tempo.minute % 5 == 0:
                low = min(df1.loc[i:i+4, "low"].values)
                high = max(df1.loc[i:i+4, "high"].values)
                open = df1.loc[i, "open"]
                close = df1.loc[i+4, "close"]
                volume = sum(df1.loc[i:i+4, "volume"].values)
            candle5MinOPEN.append(open)
            candle5MinHIGH.append(high)
            candle5MinLOW.append(low)
            candle5MinCLOSE.append(close)
        else:
            candle5MinOPEN.append(np.nan)
            candle5MinHIGH.append(np.nan)
            candle5MinLOW.append(np.nan)
            candle5MinCLOSE.append(np.nan)
    df1minute.insert(len(df1.columns)-1, 'OPEN5M', candle5MinOPEN)
    df1minute.insert(len(df1.columns)-1, 'HIGH5M', candle5MinHIGH)
    df1minute.insert(len(df1.columns)-1, 'LOW5M', candle5MinLOW)
    df1minute.insert(len(df1.columns)-1, 'CLOSE5M', candle5MinCLOSE)


def cria_velas_15mins(df1minute):
    top = len(df1minute)
    tempo = 0
    low = np.nan
    high = np.nan
    open = np.nan
    close = np.nan
    volume = np.nan
    for i in range(top):
        tempo = df1.loc[i, 'hour']
        if i+14 < top:
            if tempo.minute % 15 == 0:
                low = min(df1.loc[i:i+14, "low"].values)
                high = max(df1.loc[i:i+14, "high"].values)
                open = df1.loc[i, "open"]
                close = df1.loc[i+14, "close"]
                volume = sum(df1.loc[i:i+14, "volume"].values)
            candle15MinOPEN.append(open)
            candle15MinHIGH.append(high)
            candle15MinLOW.append(low)
            candle15MinCLOSE.append(close)
        else:
            candle15MinOPEN.append(np.nan)
            candle15MinHIGH.append(np.nan)
            candle15MinLOW.append(np.nan)
            candle15MinCLOSE.append(np.nan)
    df1minute.insert(len(df1.columns)-1, 'OPEN15M', candle15MinOPEN)
    df1minute.insert(len(df1.columns)-1, 'HIGH15M', candle15MinHIGH)
    df1minute.insert(len(df1.columns)-1, 'LOW15M', candle15MinLOW)
    df1minute.insert(len(df1.columns)-1, 'CLOSE15M', candle15MinCLOSE)


def cria_velas_30mins(df1minute):
    top = len(df1minute)
    tempo = 0
    low = np.nan
    high = np.nan
    open = np.nan
    close = np.nan
    volume = np.nan
    for i in range(top):
        tempo = df1.loc[i, 'hour']
        if i+29 < top:
            if tempo.minute % 30 == 0:
                low = min(df1.loc[i:i+29, "low"].values)
                high = max(df1.loc[i:i+29, "high"].values)
                open = df1.loc[i, "open"]
                close = df1.loc[i+29, "close"]
                volume = sum(df1.loc[i:i+29, "volume"].values)
            candle30MinOPEN.append(open)
            candle30MinHIGH.append(high)
            candle30MinLOW.append(low)
            candle30MinCLOSE.append(close)
        else:
            candle30MinOPEN.append(np.nan)
            candle30MinHIGH.append(np.nan)
            candle30MinLOW.append(np.nan)
            candle30MinCLOSE.append(np.nan)
    df1minute.insert(len(df1.columns)-1, 'OPEN30M', candle30MinOPEN)
    df1minute.insert(len(df1.columns)-1, 'HIGH30M', candle30MinHIGH)
    df1minute.insert(len(df1.columns)-1, 'LOW30M', candle30MinLOW)
    df1minute.insert(len(df1.columns)-1, 'CLOSE30M', candle30MinCLOSE)


def cria_velas_60mins(df1minute):
    top = len(df1minute)
    tempo = 0
    low = np.nan
    high = np.nan
    open = np.nan
    close = np.nan
    volume = np.nan
    for i in range(top):
        tempo = df1.loc[i, 'hour']
        if i+59 < top:
            if tempo.minute % 60 == 0:
                low = min(df1.loc[i:i+59, "low"].values)
                high = max(df1.loc[i:i+59, "high"].values)
                open = df1.loc[i, "open"]
                close = df1.loc[i+59, "close"]
                volume = sum(df1.loc[i:i+59, "volume"].values)
            candle60MinOPEN.append(open)
            candle60MinHIGH.append(high)
            candle60MinLOW.append(low)
            candle60MinCLOSE.append(close)
        else:
            candle60MinOPEN.append(np.nan)
            candle60MinHIGH.append(np.nan)
            candle60MinLOW.append(np.nan)
            candle60MinCLOSE.append(np.nan)
    df1minute.insert(len(df1.columns)-1, 'OPEN60M', candle60MinOPEN)
    df1minute.insert(len(df1.columns)-1, 'HIGH60M', candle60MinHIGH)
    df1minute.insert(len(df1.columns)-1, 'LOW60M', candle60MinLOW)
    df1minute.insert(len(df1.columns)-1, 'CLOSE60M', candle60MinCLOSE)


def le_csv_1min():
    df1 = pd.read_csv('.\EURUSD\EURUSD1.csv', names=['date', 'hour', 'open', 'high', 'low', 'close', 'volume'])
    """ df1.date_hour = pd.to_datetime(df1.date_hour, format='%d.%m.%Y %H:%M') """

    df1.date = pd.to_datetime(df1.date, format='%Y.%m.%d', infer_datetime_format=True)
    df1.hour = pd.to_datetime(df1.hour, format='%H:%M')
    start_date = '2019-09-10'
    end_date = '2019-10-11'
    df1 = df1[(df1['date'] >= start_date) & (df1['date'] <= end_date)]
    df1.set_index('date', 'hour')
    df1.date = pd.to_datetime(df1.date, format='%Y.%m.%d', infer_datetime_format=True)
    df1.hour = pd.to_datetime(df1.hour, format='%H:%M')
    df1 = df1.dropna()
    match_series = pd.to_datetime(df1['date']).map(isBusinessDay)
    df1[match_series]
    return df1


def calcula_tp_sl_ordem():
    movimento = []
    tp = []
    sl = []

    columns = list(df1)
    j = 0
    top = len(df1)

    try:
        for i in range(top):
            j = i + 14
            high = df1.loc[i:j, "high"].values
            low = df1.loc[i:j, "low"].values
            localmax = max(high)
            localmin = min(low)

            if df1.loc[j, "close"] > df1.loc[i, "close"]:
                movimento.append(1)
                tp.append(localmax)
                sl.append(localmin)
            elif df1.loc[j, "close"] < df1.loc[i, "close"]:
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

    for i in range(top - comp):
        movimento.insert(0, np.nan)
        tp.insert(0, np.nan)
        sl.insert(0, np.nan)

    df1.insert(len(df1.columns)-1, "ORDER", movimento, True)
    df1.insert(len(df1.columns)-1, "TP", tp, True)
    df1.insert(len(df1.columns)-1, "SL", sl, True)


df1 = le_csv_1min()
cria_velas_para_tfs_superiores(df1)
calcula_tp_sl_ordem(df1)

"""

sma5 = SMA(df1, timeperiod=5, price='close')
sma15 = SMA(df1, timeperiod=15, price='close')
sma60 = SMA(df1, timeperiod=60, price='close')
sma89 = SMA(df1, timeperiod=89, price='close')

df1.insert(5, "SMA5", sma5, True)
df1.insert(5, "SMA15", sma15, True)
df1.insert(5, "SMA60", sma60, True)
df1.insert(5, "SMA89", sma89, True)
"""



graphic1MCandle = go.Candlestick(x=df1.index, open=df1['open'], high=df1['high'], low=df1['low'], close=df1['close'],
                                 increasing_line_color='blue', decreasing_line_color='red')
graphicTP = go.Scatter(x=df1.index, y=df1['TP'], name='TP', connectgaps=False, mode='markers')

fig = tools.make_subplots(rows=2, cols=1, shared_yaxes=True, shared_xaxes=False)
fig.add_trace(graphic1MCandle, 1, 1)
fig.add_trace(graphicTP, 1, 1)
fig.update_layout(title='EURUSD - M1 - TP @ 15M',
                  xaxis_title='DATE AND TIME',
                  yaxis_title='EURUSD')


def update_figure(selected):
    dff = df1[(df["date_hour"] >= selected[0]) & (df1["date_hour"] <= selected[1])]
    trace = go.Candlestick(x=dff['date_hour'], open=dff['open'], high=dff['high'], low=dff['low'], close=dff['close'],
                           increasing={'line': {'color': '#00CC94'}}, decreasing={'line': {'color': '#F50030'}})
    return {'date_hour': [trace],
            'layout': go.Layout(title=f"Stock Values for the period:{'-'.join(str(i) for i in selected)}",
                                xaxis={'rangeslider': {'visible': False}, 'autorange': "reversed", },
                                yaxis={"title": f'Stock Price (USD)'}
                                )}


fig.layout.on_change(update_figure, 'xaxis.range')
fig.update_layout(xaxis_rangeslider_visible=False)

"""py.offline.plot(fig, filename='tutorial.html')"""