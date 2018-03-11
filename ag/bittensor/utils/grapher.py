import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.finance import candlestick_ohlc
from matplotlib.finance import volume_overlay
from matplotlib.dates import (MONDAY, DateFormatter, MonthLocator,
                              WeekdayLocator, date2num)
import matplotlib.cbook as cbook
import matplotlib.image as image


def candleChart(dataframe, title, ax=None):
    """This works"""
    plot_this = False
    if ax is None:
        plot_this = True
        fig = plt.figure(figsize=(12,12), facecolor='white',
            edgecolor='black')
        fig.suptitle(title)
        datafile = cbook.get_sample_data(
            os.path.join(os.getcwd(), 'images\\logo.png'), asfileobj=False)
        im = image.imread(datafile)
        fig.figimage(im, 10, 10, zorder=3, alpha=.3)
    ax = plt.axes()
    candledata = zip(
        range(len(dataframe)),
        # date2num(dataframe.index),
        # dataframe.index,
        dataframe['Open'],
        dataframe['High'],
        dataframe['Low'],
        dataframe['Close'],
        dataframe['Volume']
        )
    candlestick_ohlc(ax, candledata, colorup = "black", 
        colordown = "orange",
        width = 2 * .4)
    try:  # plot Moving AVGS
        ax.plot([x for x in dataframe['slow']], linestyle='--')
        ax.plot([x for x in dataframe['fast']], linestyle=':')
    except:
        print('Failed to make Moving Average.')
        pass
    
    try:  # color fill on MA.
        pass
    except:
        pass
    
    try:  # plot MACD crossover
        for i, e in enumerate(dataframe['macd_signal']):
            if e <= -1:
                ax.plot(i, dataframe['Close'].iloc[i], 'ro', ms=15, lw=0, alpha=0.7, mfc='red')
            if e >= 1:
                ax.plot(i, dataframe['Close'].iloc[i], 'ro', ms=15, lw=0, alpha=0.7, mfc='green')
    except:
        print('Failed to make MACD signal dots.')
        pass
    try:  # plot Volume Bars
        vax = ax.twinx()
        v = volume_overlay(
            ax=vax, 
            opens=dataframe['Open'], 
            closes=dataframe['Close'], 
            volumes=dataframe['baseVolume'],
            colorup='k', 
            colordown='r',
            width=4, 
            alpha=1.0 )
        vax.add_collection(v)
    except Exception as e:
        print('Failed to make Volume Bars.\n', e)
        pass
    
    ax.axis('off')
    if plot_this is None:
        plt.show()
    return ax


