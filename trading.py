import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
from plotly import graph_objs as go
from plotly import tools
from talib import RSI
from talib import abstract

SMA = abstract.SMA

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

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
    __cria_velas_5mins(df1minute)
    __cria_velas_15mins(df1minute)
    __cria_velas_30mins(df1minute)
    __cria_velas_60mins(df1minute)


def __cria_velas_5mins(df1minute):
    top = len(df1minute)
    tempo = 0
    low = np.nan
    high = np.nan
    open = np.nan
    close = np.nan
    volume = np.nan
    for i in range(top):
        tempo = df1.loc[i, 'date_hour']
        if i + 4 < top:
            if tempo.minute % 5 == 0:
                low = min(df1.loc[i:i + 4, "LOW1M"].values)
                high = max(df1.loc[i:i + 4, "HIGH1M"].values)
                open = df1.loc[i, "OPEN1M"]
                close = df1.loc[i + 4, "CLOSE1M"]
                volume = sum(df1.loc[i:i + 4, "volume"].values)
            candle5MinOPEN.append(open)
            candle5MinHIGH.append(high)
            candle5MinLOW.append(low)
            candle5MinCLOSE.append(close)
        else:
            candle5MinOPEN.append(np.nan)
            candle5MinHIGH.append(np.nan)
            candle5MinLOW.append(np.nan)
            candle5MinCLOSE.append(np.nan)
    df1minute.insert(len(df1.columns) - 1, 'OPEN5M', candle5MinOPEN)
    df1minute.insert(len(df1.columns) - 1, 'HIGH5M', candle5MinHIGH)
    df1minute.insert(len(df1.columns) - 1, 'LOW5M', candle5MinLOW)
    df1minute.insert(len(df1.columns) - 1, 'CLOSE5M', candle5MinCLOSE)


def __cria_velas_15mins(df1minute):
    top = len(df1minute)
    tempo = 0
    low = np.nan
    high = np.nan
    open = np.nan
    close = np.nan
    volume = np.nan
    for i in range(top):
        tempo = df1.loc[i, 'date_hour']
        if i + 14 < top:
            if tempo.minute % 15 == 0:
                low = min(df1.loc[i:i + 14, "LOW1M"].values)
                high = max(df1.loc[i:i + 14, "HIGH1M"].values)
                open = df1.loc[i, "OPEN1M"]
                close = df1.loc[i + 14, "CLOSE1M"]
                volume = sum(df1.loc[i:i + 14, "volume"].values)
            candle15MinOPEN.append(open)
            candle15MinHIGH.append(high)
            candle15MinLOW.append(low)
            candle15MinCLOSE.append(close)
        else:
            candle15MinOPEN.append(np.nan)
            candle15MinHIGH.append(np.nan)
            candle15MinLOW.append(np.nan)
            candle15MinCLOSE.append(np.nan)
    df1minute.insert(len(df1.columns) - 1, 'OPEN15M', candle15MinOPEN)
    df1minute.insert(len(df1.columns) - 1, 'HIGH15M', candle15MinHIGH)
    df1minute.insert(len(df1.columns) - 1, 'LOW15M', candle15MinLOW)
    df1minute.insert(len(df1.columns) - 1, 'CLOSE15M', candle15MinCLOSE)


def __cria_velas_30mins(df1minute):
    top = len(df1minute)
    tempo = 0
    low = np.nan
    high = np.nan
    open = np.nan
    close = np.nan
    volume = np.nan
    for i in range(top):
        tempo = df1.loc[i, 'date_hour']
        if i + 29 < top:
            if tempo.minute % 30 == 0:
                low = min(df1.loc[i:i + 29, "LOW1M"].values)
                high = max(df1.loc[i:i + 29, "HIGH1M"].values)
                open = df1.loc[i, "OPEN1M"]
                close = df1.loc[i + 29, "CLOSE1M"]
                volume = sum(df1.loc[i:i + 29, "volume"].values)
            candle30MinOPEN.append(open)
            candle30MinHIGH.append(high)
            candle30MinLOW.append(low)
            candle30MinCLOSE.append(close)
        else:
            candle30MinOPEN.append(np.nan)
            candle30MinHIGH.append(np.nan)
            candle30MinLOW.append(np.nan)
            candle30MinCLOSE.append(np.nan)
    df1minute.insert(len(df1.columns) - 1, 'OPEN30M', candle30MinOPEN)
    df1minute.insert(len(df1.columns) - 1, 'HIGH30M', candle30MinHIGH)
    df1minute.insert(len(df1.columns) - 1, 'LOW30M', candle30MinLOW)
    df1minute.insert(len(df1.columns) - 1, 'CLOSE30M', candle30MinCLOSE)


def __cria_velas_60mins(df1minute):
    top = len(df1minute)
    tempo = 0
    low = np.nan
    high = np.nan
    open = np.nan
    close = np.nan
    volume = np.nan
    for i in range(top):
        tempo = df1.loc[i, 'date_hour']
        if i + 59 < top:
            if tempo.minute % 60 == 0:
                low = min(df1.loc[i:i + 59, "LOW1M"].values)
                high = max(df1.loc[i:i + 59, "HIGH1M"].values)
                open = df1.loc[i, "OPEN1M"]
                close = df1.loc[i + 59, "CLOSE1M"]
                volume = sum(df1.loc[i:i + 59, "volume"].values)
            candle60MinOPEN.append(open)
            candle60MinHIGH.append(high)
            candle60MinLOW.append(low)
            candle60MinCLOSE.append(close)
        else:
            candle60MinOPEN.append(np.nan)
            candle60MinHIGH.append(np.nan)
            candle60MinLOW.append(np.nan)
            candle60MinCLOSE.append(np.nan)
    df1minute.insert(len(df1.columns) - 1, 'OPEN60M', candle60MinOPEN)
    df1minute.insert(len(df1.columns) - 1, 'HIGH60M', candle60MinHIGH)
    df1minute.insert(len(df1.columns) - 1, 'LOW60M', candle60MinLOW)
    df1minute.insert(len(df1.columns) - 1, 'CLOSE60M', candle60MinCLOSE)


def le_csv_1min():
    dataFrameCSV = pd.read_csv('.\EURUSD\EURUSD1.csv',
                               names=['date', 'hour', 'OPEN1M', 'HIGH1M', 'LOW1M', 'CLOSE1M', 'volume'],
                               parse_dates=[['date', 'hour']])
    dataFrameCSV.date_hour = pd.to_datetime(dataFrameCSV.date_hour, format='%d.%m.%Y %H:%M')
    """
    df1.date = pd.to_datetime(df1.date, format='%Y.%m.%d', infer_datetime_format=True)
    df1.hour = pd.to_datetime(df1.hour, format='%H:%M')
    
    start_date = '2019-09-10'
    end_date = '2019-10-11'
    df1 = df1[(df1['date'] >= start_date) & (df1['date'] <= end_date)]
    df1.set_index('date', 'hour')
    df1.date = pd.to_datetime(df1.date, format='%Y.%m.%d', infer_datetime_format=True)
    df1.hour = pd.to_datetime(df1.hour, format='%H:%M')
    """
    dataFrameCSV = dataFrameCSV.dropna()
    match_series = pd.to_datetime(dataFrameCSV['date_hour']).map(isBusinessDay)
    dataFrameCSV[match_series]
    return dataFrameCSV


def calcula_tp_sl_ordem(df1):
    movimento = []
    tp = []
    sl = []

    columns = list(df1)
    top = len(df1)

    try:
        for i in range(top):
            if df1.loc[i, "CLOSE60M"] > df1.loc[i, "CLOSE1M"]:
                movimento.append(1)
                tp.append(df1.loc[i, "CLOSE60M"])
                sl.append(df1.loc[i, "LOW60M"])
            elif df1.loc[i, "CLOSE60M"] < df1.loc[i, "CLOSE1M"]:
                movimento.append(-1)
                sl.append(df1.loc[i, "CLOSE60M"])
                tp.append(df1.loc[i, "LOW60M"])
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

    df1.insert(len(df1.columns) - 1, "ORDER", movimento, True)
    df1.insert(len(df1.columns) - 1, "TP", tp, True)
    df1.insert(len(df1.columns) - 1, "SL", sl, True)


def cria_media_movel(df1):
    sma5_1m = SMA(df1, timeperiod=5, price='CLOSE1M')
    sma15_1m = SMA(df1, timeperiod=15, price='CLOSE1M')
    sma60_1m = SMA(df1, timeperiod=60, price='CLOSE1M')
    sma89_1m = SMA(df1, timeperiod=89, price='CLOSE1M')

    sma6_5m = SMA(df1, timeperiod=5, price='CLOSE5M')
    sma12_5m = SMA(df1, timeperiod=12, price='CLOSE5M')
    sma24_5m = SMA(df1, timeperiod=24, price='CLOSE5M')
    sma48_5m = SMA(df1, timeperiod=48, price='CLOSE5M')

    sma50_15m = SMA(df1, timeperiod=50, price='CLOSE15M')
    sma100_15m = SMA(df1, timeperiod=100, price='CLOSE15M')
    sma200_15m = SMA(df1, timeperiod=200, price='CLOSE15M')

    sma50_1h = SMA(df1, timeperiod=50, price='CLOSE60M')
    sma100_1h = SMA(df1, timeperiod=100, price='CLOSE60M')
    sma200_1h = SMA(df1, timeperiod=200, price='CLOSE60M')

    df1.insert(len(df1.columns) - 1, "SMA5_1M", sma5_1m, True)
    df1.insert(len(df1.columns) - 1, "SMA15_1M", sma15_1m, True)
    df1.insert(len(df1.columns) - 1, "SMA60_1M", sma60_1m, True)
    df1.insert(len(df1.columns) - 1, "SMA89_1M", sma89_1m, True)
    df1.insert(len(df1.columns) - 1, "SMA6_5M", sma6_5m, True)
    df1.insert(len(df1.columns) - 1, "SMA12_5M", sma12_5m, True)
    df1.insert(len(df1.columns) - 1, "SMA24_5M", sma24_5m, True)
    df1.insert(len(df1.columns) - 1, "SMA48_5M", sma48_5m, True)
    df1.insert(len(df1.columns) - 1, "SMA50_15M", sma50_15m, True)
    df1.insert(len(df1.columns) - 1, "SMA100_15M", sma100_15m, True)
    df1.insert(len(df1.columns) - 1, "SMA200_15M", sma200_15m, True)
    df1.insert(len(df1.columns) - 1, "SMA50_1H", sma50_1h, True)
    df1.insert(len(df1.columns) - 1, "SMA100_1H", sma100_1h, True)
    df1.insert(len(df1.columns) - 1, "SMA200_1H", sma200_1h, True)

def cria_rsi(df1):
    rsi_1m = RSI(df1['CLOSE1M'].values, timeperiod=14)
    rsi_5m = RSI(df1['CLOSE5M'].values, timeperiod=14)
    rsi_15m = RSI(df1['CLOSE15M'].values, timeperiod=14)
    rsi_30m = RSI(df1['CLOSE30M'].values, timeperiod=14)
    rsi_60m = RSI(df1['CLOSE60M'].values, timeperiod=14)

    top = len(df1)
    comp = len(rsi_1m)
    for i in range(top - comp):
        rsi_1m.insert(0, np.nan)
        rsi_5m.insert(0, np.nan)
        rsi_15m.insert(0, np.nan)
        rsi_30m.insert(0, np.nan)
        rsi_60m.insert(0, np.nan)

    df1.insert(len(df1.columns) - 1, "RSI_1M", rsi_1m, True)
    df1.insert(len(df1.columns) - 1, "RSI_5M", rsi_5m, True)
    df1.insert(len(df1.columns) - 1, "RSI_15M", rsi_15m, True)
    df1.insert(len(df1.columns) - 1, "RSI_30M", rsi_30m, True)
    df1.insert(len(df1.columns) - 1, "RSI_60M", rsi_60m, True)
def cria_grafico(df1):
    graphic_1m_candle = go.Candlestick(x=df1.index, open=df1['open'], high=df1['high'], low=df1['low'],
                                       close=df1['close'], increasing_line_color='blue', decreasing_line_color='red')
    graphic_tp = go.Scatter(x=df1.index, y=df1['TP'], name='TP', connectgaps=False, mode='markers')
    fig = tools.make_subplots(rows=2, cols=1, shared_yaxes=True, shared_xaxes=False)
    fig.add_trace(graphic_1m_candle, 1, 1)
    fig.add_trace(graphic_tp, 1, 1)
    fig.update_layout(title='EURUSD - M1 - TP @ 15M',
                      xaxis_title='DATE AND TIME',
                      yaxis_title='EURUSD')
    py.offline.plot(fig, filename='tutorial.html')


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


df1 = le_csv_1min()
df1.round(5)
cria_velas_para_tfs_superiores(df1)
df1 = df1.dropna()
calcula_tp_sl_ordem(df1)
cria_media_movel(df1)
cria_rsi(df1)
print(df1.loc[320:1420, :])

"""
- BUGS
    - A colocação em um DAtaFrame de dados de vários TimeFrames distorce as médias e o RSI
    é preciso separar em outros DataFrames 

"""
