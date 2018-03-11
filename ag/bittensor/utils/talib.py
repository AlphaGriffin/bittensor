#!/usr/bin/python3
"""
Bittensor TA-LIB.
by: AlphaGriffin

@ SOURCE Inspiration:
    TA LIB basics... this came in a very pandas2 kind of way.
    https://www.quantopian.com/posts/technical-analysis-indicators-without-talib-code
    @author: Bruno Franca
    @author: Peter Bakker

@ SOURCE Inspiration:
    Analysis comes from this source.
    https://www.quantinsti.com/blog/build-technical-indicators-in-python/
    By Milind Paradkar :: https://www.linkedin.com/in/milind-paradkar-b37292107/
"""

__author__ = "Eric Petersen @Ruckusist"
__copyright__ = "Copyright 2018, The Alpha Griffin Project"
__credits__ = ["Eric Petersen", "Shawn Wilson", "@alphagriffin"]
__license__ = "***"
__version__ = "0.0.2"
__maintainer__ = "Eric Petersen"
__email__ = "ruckusist@alphagriffin.com"
__status__ = "Beta"

import numpy
import pandas as pd
import math as m

class TALib(object):
    def __init__(self, options=None):
        self.options = options
        self.available = {
            'Moving Average': self.MA,
            'Bollinger Bands': self.BBANDS,
            'Standard Deviation': self.STD,
            'Exp Moving Average': self.EMA,
            'Momentum': self.MOM,
            'Rate of Change': self.ROC,
            'Average True Range': self.ATR,
            'Commodity Channel Index': self.CCI,
            'Realitive Strength Index': self.RSI
        }

    def main(self):
        pass

    def unzero(self, df):
        """Necessary for any df rolling calculation."""
        df = df.replace(0,'NaN')
        df = df.dropna(how='all',axis=0)
        df = df.replace('NaN', 0)
        df.len = len(df)
        return df

    def MA(self, df, period=12, column='Close'):
        """BASIC MOVING AVERAGE"""
        MA = pd.Series(
                self.unzero(  # unzero seems to be useless after join
                    df[column].rolling(
                        window=period
                        ).mean()
                ),
            name='MA_{}_{}'.format(column, period))
        del df
        # df = df.join(MA)
        return MA  # , df

    def STD(self, df, period=12, column='Close'):
        """BASIC MOVING AVERAGE"""
        STD = pd.Series(
                self.unzero(  # unzero seems to be useless after join
                    df[column].rolling(
                        window=period
                        ).std()
                ),
            name='STD_{}_{}'.format(column, period))
        del df
        # df = df.join(MA)
        return STD  # , df

    def EMA(self, df, period=12, column='Close'):
        """ EWM | Expontentially Weighted Moving Average
        looks like this is only working with the mean() call after it
        """
        EMA = pd.Series(
                df[column].ewm(span=period, min_periods=period-1).mean(),
                name='EMA_{}_{}'.format(column, period))
        del df
        # df = df.join(EMA)
        return EMA  # , df

    def MOM(self, df, period=12, column='Close'):
        """Momentum, or rolling Diff..."""
        MOM = pd.Series(
            df[column].diff(period),
            name="Momentum_{}_{}".format(column,period)
        )
        del df
        # df = df.join(MOM)
        return MOM  # , df

    def ROC(self, df, period=12, column='Close'):
        """ Rate of Change
        Momentum over Close price.
        """
        M = self.MOM(df, period+1, column)
        L = df[column].shift(period - 1)

        ROC = pd.Series(
            M / L,
            name="RateOfChange_{}_{}".format(column,period)
        )
        del df
        # df = df.join(ROC)
        return ROC  # , df

    def ATR(self, df, period=12):
        TR_l = [0]
        for i in range(len(df)-1):
            TR = max(
                df['High'].iloc[i + 1],
                df['Close'].iloc[i] - df['Low'].iloc[i + 1],
                df['Close'].iloc[i]
            )
            TR_l.append(TR)
        TR_s = pd.Series(TR_l)
        ATR = pd.Series(TR_s.ewm(span=period, min_periods=period,adjust=True,ignore_na=True).mean(),
                        name='TrueRange_{}'.format(period))
        # ATR = ATR.reindex(df.index)
        return ATR

    def BBANDS(self, df, period=12):
        MA = self.MA(df, period, 'Close')
        STD = self.STD(df, period, 'Close')

        b1 = MA + (2*STD)
        B1 = pd.Series(b1, name='BollingerB1_{}'.format(period))
        b2 =  MA - (2*STD)
        B2 = pd.Series(b2, name='BollingerB2_{}'.format(period))
        return (B1, MA, B2)

    def CCI(self, df, period=12):
        # wtf is a PP??
        PP = pd.Series((df['High'] + df['Low'] + df['Close']) / 3)
        MA_TP = PP.rolling(window=period, center=False).mean()
        x = PP - MA_TP
        ST_TP = PP.rolling(window=period, center=False).std()
        change_ST = ST_TP * 0.015
        CCI = pd.Series(x / change_ST, name='CCI_{}'.format(period))

        analysis = None
        analysis = 'overbought' if CCI[-1] > 100 else analysis
        analysis = 'oversold' if CCI[-1] < -100 else analysis
        return CCI  #, analysis

    def RSI(self, df, period=12):
        Up_I = []
        Do_I = []
        for i in range(len(df) - 1):
            Up_M = df['High'].iloc[i + 1] - df['High'].iloc[i]
            Do_M = df['Low'].iloc[i + 1] - df['Low'].iloc[i]
            if Up_M > Do_M and Up_M > 0:
                Up_D = Up_M
            else:
                Up_D = 0
            Up_I.append(Up_D)

            if Do_M > Up_M and Do_M > 0:
                Do_D = Do_M
            else:
                Do_D = 0
            Do_I.append(Do_D)
        UPI = pd.Series(Up_I)
        DOI = pd.Series(Do_I)
        PosDI = pd.Series(UPI.ewm(span=period, min_periods=period-1).mean())
        NegDI = pd.Series(DOI.ewm(span=period, min_periods=period-1).mean())
        RSI = pd.Series(PosDI / (PosDI + NegDI), name='RSI_{}'.format(period))

        analysis = None
        analysis = 'overbought' if RSI.iloc[-1] > 0.7 else analysis
        analysis = 'oversold' if RSI.iloc[-1] < 0.3 else analysis

        return RSI  #, analysis


#Pivot Points, Supports and Resistances
def PPSR(df):
    PP = pd.Series((df['High'] + df['Low'] + df['Close']) / 3)
    R1 = pd.Series(2 * PP - df['Low'])
    S1 = pd.Series(2 * PP - df['High'])
    R2 = pd.Series(PP + df['High'] - df['Low'])
    S2 = pd.Series(PP - df['High'] + df['Low'])
    R3 = pd.Series(df['High'] + 2 * (PP - df['Low']))
    S3 = pd.Series(df['Low'] - 2 * (df['High'] - PP))
    psr = {'PP':PP, 'R1':R1, 'S1':S1, 'R2':R2, 'S2':S2, 'R3':R3, 'S3':S3}
    PSR = pd.DataFrame(psr)
    df = df.join(PSR)
    return df

#Stochastic oscillator %K
def STOK(df):
    SOk = pd.Series((df['Close'] - df['Low']) / (df['High'] - df['Low']), name = 'SO%k')
    df = df.join(SOk)
    return df

#Stochastic oscillator %D
def STO(df, n):
    SOk = pd.Series((df['Close'] - df['Low']) / (df['High'] - df['Low']), name = 'SO%k')
    SOd = pd.Series(pd.ewma(SOk, span = n, min_periods = n - 1), name = 'SO%d_' + str(n))
    df = df.join(SOd)
    return df

#Trix
def TRIX(df, n):
    EX1 = pd.ewma(df['Close'], span = n, min_periods = n - 1)
    EX2 = pd.ewma(EX1, span = n, min_periods = n - 1)
    EX3 = pd.ewma(EX2, span = n, min_periods = n - 1)
    i = 0
    ROC_l = [0]
    while i + 1 <= df.index[-1]:
        ROC = (EX3[i + 1] - EX3[i]) / EX3[i]
        ROC_l.append(ROC)
        i = i + 1
    Trix = pd.Series(ROC_l, name = 'Trix_' + str(n))
    df = df.join(Trix)
    return df

#Average Directional Movement Index
def ADX(df, n, n_ADX):
    i = 0
    UpI = []
    DoI = []
    while i + 1 <= df.index[-1]:
        UpMove = df.get_value(i + 1, 'High') - df.get_value(i, 'High')
        DoMove = df.get_value(i, 'Low') - df.get_value(i + 1, 'Low')
        if UpMove > DoMove and UpMove > 0:
            UpD = UpMove
        else: UpD = 0
        UpI.append(UpD)
        if DoMove > UpMove and DoMove > 0:
            DoD = DoMove
        else: DoD = 0
        DoI.append(DoD)
        i = i + 1
    i = 0
    TR_l = [0]
    while i < df.index[-1]:
        TR = max(df.get_value(i + 1, 'High'), df.get_value(i, 'Close')) - min(df.get_value(i + 1, 'Low'), df.get_value(i, 'Close'))
        TR_l.append(TR)
        i = i + 1
    TR_s = pd.Series(TR_l)
    ATR = pd.Series(pd.ewma(TR_s, span = n, min_periods = n))
    UpI = pd.Series(UpI)
    DoI = pd.Series(DoI)
    PosDI = pd.Series(pd.ewma(UpI, span = n, min_periods = n - 1) / ATR)
    NegDI = pd.Series(pd.ewma(DoI, span = n, min_periods = n - 1) / ATR)
    ADX = pd.Series(pd.ewma(abs(PosDI - NegDI) / (PosDI + NegDI), span = n_ADX, min_periods = n_ADX - 1), name = 'ADX_' + str(n) + '_' + str(n_ADX))
    df = df.join(ADX)
    return df

#MACD, MACD Signal and MACD difference
def MACD(df, n_fast, n_slow):
    EMAfast = pd.Series(pd.ewma(df['Close'], span = n_fast, min_periods = n_slow - 1))
    EMAslow = pd.Series(pd.ewma(df['Close'], span = n_slow, min_periods = n_slow - 1))
    MACD = pd.Series(EMAfast - EMAslow, name = 'MACD_' + str(n_fast) + '_' + str(n_slow))
    MACDsign = pd.Series(pd.ewma(MACD, span = 9, min_periods = 8), name = 'MACDsign_' + str(n_fast) + '_' + str(n_slow))
    MACDdiff = pd.Series(MACD - MACDsign, name = 'MACDdiff_' + str(n_fast) + '_' + str(n_slow))
    df = df.join(MACD)
    df = df.join(MACDsign)
    df = df.join(MACDdiff)
    return df

#Mass Index
def MassI(df):
    Range = df['High'] - df['Low']
    EX1 = pd.ewma(Range, span = 9, min_periods = 8)
    EX2 = pd.ewma(EX1, span = 9, min_periods = 8)
    Mass = EX1 / EX2
    MassI = pd.Series(pd.rolling_sum(Mass, 25), name = 'Mass Index')
    df = df.join(MassI)
    return df

#Vortex Indicator: http://www.vortexindicator.com/VFX_VORTEX.PDF
def Vortex(df, n):
    i = 0
    TR = [0]
    while i < df.index[-1]:
        Range = max(df.get_value(i + 1, 'High'), df.get_value(i, 'Close')) - min(df.get_value(i + 1, 'Low'), df.get_value(i, 'Close'))
        TR.append(Range)
        i = i + 1
    i = 0
    VM = [0]
    while i < df.index[-1]:
        Range = abs(df.get_value(i + 1, 'High') - df.get_value(i, 'Low')) - abs(df.get_value(i + 1, 'Low') - df.get_value(i, 'High'))
        VM.append(Range)
        i = i + 1
    VI = pd.Series(pd.rolling_sum(pd.Series(VM), n) / pd.rolling_sum(pd.Series(TR), n), name = 'Vortex_' + str(n))
    df = df.join(VI)
    return df

#KST Oscillator
def KST(df, r1, r2, r3, r4, n1, n2, n3, n4):
    M = df['Close'].diff(r1 - 1)
    N = df['Close'].shift(r1 - 1)
    ROC1 = M / N
    M = df['Close'].diff(r2 - 1)
    N = df['Close'].shift(r2 - 1)
    ROC2 = M / N
    M = df['Close'].diff(r3 - 1)
    N = df['Close'].shift(r3 - 1)
    ROC3 = M / N
    M = df['Close'].diff(r4 - 1)
    N = df['Close'].shift(r4 - 1)
    ROC4 = M / N
    KST = pd.Series(pd.rolling_sum(ROC1, n1) + pd.rolling_sum(ROC2, n2) * 2 + pd.rolling_sum(ROC3, n3) * 3 + pd.rolling_sum(ROC4, n4) * 4, name = 'KST_' + str(r1) + '_' + str(r2) + '_' + str(r3) + '_' + str(r4) + '_' + str(n1) + '_' + str(n2) + '_' + str(n3) + '_' + str(n4))
    df = df.join(KST)
    return df

#True Strength Index
def TSI(df, r, s):
    M = pd.Series(df['Close'].diff(1))
    aM = abs(M)
    EMA1 = pd.Series(pd.ewma(M, span = r, min_periods = r - 1))
    aEMA1 = pd.Series(pd.ewma(aM, span = r, min_periods = r - 1))
    EMA2 = pd.Series(pd.ewma(EMA1, span = s, min_periods = s - 1))
    aEMA2 = pd.Series(pd.ewma(aEMA1, span = s, min_periods = s - 1))
    TSI = pd.Series(EMA2 / aEMA2, name = 'TSI_' + str(r) + '_' + str(s))
    df = df.join(TSI)
    return df

#Accumulation/Distribution
def ACCDIST(df, n):
    ad = (2 * df['Close'] - df['High'] - df['Low']) / (df['High'] - df['Low']) * df['Volume']
    M = ad.diff(n - 1)
    N = ad.shift(n - 1)
    ROC = M / N
    AD = pd.Series(ROC, name = 'Acc/Dist_ROC_' + str(n))
    df = df.join(AD)
    return df

#Chaikin Oscillator
def Chaikin(df):
    ad = (2 * df['Close'] - df['High'] - df['Low']) / (df['High'] - df['Low']) * df['Volume']
    Chaikin = pd.Series(pd.ewma(ad, span = 3, min_periods = 2) - pd.ewma(ad, span = 10, min_periods = 9), name = 'Chaikin')
    df = df.join(Chaikin)
    return df

#Money Flow Index and Ratio
def MFI(df, n):
    PP = (df['High'] + df['Low'] + df['Close']) / 3
    i = 0
    PosMF = [0]
    while i < df.index[-1]:
        if PP[i + 1] > PP[i]:
            PosMF.append(PP[i + 1] * df.get_value(i + 1, 'Volume'))
        else:
            PosMF.append(0)
        i = i + 1
    PosMF = pd.Series(PosMF)
    TotMF = PP * df['Volume']
    MFR = pd.Series(PosMF / TotMF)
    MFI = pd.Series(pd.rolling_mean(MFR, n), name = 'MFI_' + str(n))
    df = df.join(MFI)
    return df

#On-balance Volume
def OBV(df, n):
    i = 0
    OBV = [0]
    while i < df.index[-1]:
        if df.get_value(i + 1, 'Close') - df.get_value(i, 'Close') > 0:
            OBV.append(df.get_value(i + 1, 'Volume'))
        if df.get_value(i + 1, 'Close') - df.get_value(i, 'Close') == 0:
            OBV.append(0)
        if df.get_value(i + 1, 'Close') - df.get_value(i, 'Close') < 0:
            OBV.append(-df.get_value(i + 1, 'Volume'))
        i = i + 1
    OBV = pd.Series(OBV)
    OBV_ma = pd.Series(pd.rolling_mean(OBV, n), name = 'OBV_' + str(n))
    df = df.join(OBV_ma)
    return df

#Force Index
def FORCE(df, n):
    F = pd.Series(df['Close'].diff(n) * df['Volume'].diff(n), name = 'Force_' + str(n))
    df = df.join(F)
    return df

#Ease of Movement
def EOM(df, n):
    EoM = (df['High'].diff(1) + df['Low'].diff(1)) * (df['High'] - df['Low']) / (2 * df['Volume'])
    Eom_ma = pd.Series(pd.rolling_mean(EoM, n), name = 'EoM_' + str(n))
    df = df.join(Eom_ma)
    return df

#Coppock Curve
def COPP(df, n):
    M = df['Close'].diff(int(n * 11 / 10) - 1)
    N = df['Close'].shift(int(n * 11 / 10) - 1)
    ROC1 = M / N
    M = df['Close'].diff(int(n * 14 / 10) - 1)
    N = df['Close'].shift(int(n * 14 / 10) - 1)
    ROC2 = M / N
    Copp = pd.Series(pd.ewma(ROC1 + ROC2, span = n, min_periods = n), name = 'Copp_' + str(n))
    df = df.join(Copp)
    return df

#Keltner Channel
def KELCH(df, n):
    KelChM = pd.Series(pd.rolling_mean((df['High'] + df['Low'] + df['Close']) / 3, n), name = 'KelChM_' + str(n))
    KelChU = pd.Series(pd.rolling_mean((4 * df['High'] - 2 * df['Low'] + df['Close']) / 3, n), name = 'KelChU_' + str(n))
    KelChD = pd.Series(pd.rolling_mean((-2 * df['High'] + 4 * df['Low'] + df['Close']) / 3, n), name = 'KelChD_' + str(n))
    df = df.join(KelChM)
    df = df.join(KelChU)
    df = df.join(KelChD)
    return df

#Ultimate Oscillator
def ULTOSC(df):
    i = 0
    TR_l = [0]
    BP_l = [0]
    while i < df.index[-1]:
        TR = max(df.get_value(i + 1, 'High'), df.get_value(i, 'Close')) - min(df.get_value(i + 1, 'Low'), df.get_value(i, 'Close'))
        TR_l.append(TR)
        BP = df.get_value(i + 1, 'Close') - min(df.get_value(i + 1, 'Low'), df.get_value(i, 'Close'))
        BP_l.append(BP)
        i = i + 1
    UltO = pd.Series((4 * pd.rolling_sum(pd.Series(BP_l), 7) / pd.rolling_sum(pd.Series(TR_l), 7)) + (2 * pd.rolling_sum(pd.Series(BP_l), 14) / pd.rolling_sum(pd.Series(TR_l), 14)) + (pd.rolling_sum(pd.Series(BP_l), 28) / pd.rolling_sum(pd.Series(TR_l), 28)), name = 'Ultimate_Osc')
    df = df.join(UltO)
    return df

#Donchian Channel
def DONCH(df, n):
    i = 0
    DC_l = []
    while i < n - 1:
        DC_l.append(0)
        i = i + 1
    i = 0
    while i + n - 1 < df.index[-1]:
        DC = max(df['High'].ix[i:i + n - 1]) - min(df['Low'].ix[i:i + n - 1])
        DC_l.append(DC)
        i = i + 1
    DonCh = pd.Series(DC_l, name = 'Donchian_' + str(n))
    DonCh = DonCh.shift(n - 1)
    df = df.join(DonCh)
    return df
