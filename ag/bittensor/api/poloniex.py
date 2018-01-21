#!/usr/bin/env python

"""
A Test poloniex trader.
@Ruckusist<ruckusist@alphagriffin.com>
"""

import os
import sys
import time
import datetime
import yaml
import humanize as Readable
import urllib.request
import urllib.parse
import json
import hmac
import hashlib

__author__ = "Eric Petersen @Ruckusist"
__copyright__ = "Copyright 2017, The Alpha Griffin Project"
__credits__ = ["Eric Petersen", "Shawn Wilson", "@alphagriffin"]
__license__ = "***"
__version__ = "0.0.2"
__maintainer__ = "Eric Petersen"
__email__ = "ruckusist@alphagriffin.com"
__status__ = "Beta"

def get_time():
    """Shortcut for datetime."""
    return datetime.datetime.now()


def get_time_diff(t1, t2):
    """Use Humanize for Sanity."""
    _time = Readable.naturaltime(t1 - t2)
    return _time


class timeStuff(object):
    """Conversion of time, for different cases."""

    def __init__(self):
        self.test = 1


    @staticmethod
    def unix2date(unixtime):
        return time.ctime(int(unixtime))


class Options(object):
    """OH OH DO a yaml file!."""

    def __init__(self, data_path=None):
        """OH OH DO a yaml file!."""
        if data_path is None:
            data_path = os.path.join(os.getcwd(), "config.yaml")
        config = self.load_options(data_path)
        for i in config:
            setattr(self, '{}'.format(i), '{}'.format(config[i]))

    @staticmethod
    def load_options(data_path):
        """Attach all keys to this class with their value."""
        try:
            with open(data_path, 'r') as config:
                new_config = yaml.load(config)
            return new_config
        except Exception as e:
            print("burn {}".format(e))


class Trader(object):
    """A controller for poloniex."""

    def __init__(self, options=None, database=None):
        """Setup the instrument."""
        self.options = options
        self.database = database
        self.starttime = get_time()

        # modules...

        # globals...
        self.state = 0

    def __call__(self):
        """Should be entry level here. Make it easy."""
        with urllib.request.urlopen('https://poloniex.com/public?command=returnTicker') as url:
            data = json.loads(url.read().decode('UTF-8'))
            for coin in sorted(data):
                print("{}".format(coin))
                for i in data[coin]:
                    x = data[coin][i]
                    print("\t{}: {}".format(i, x))
        return data

    def main(self):
        """Sanity Check."""
        if self.state is 0:
            print("Poloniex Api Tester. Testing...")
            return True

    @staticmethod
    def call_ticker():
        """Poloniex public ticker."""
        with urllib.request.urlopen(
            'https://poloniex.com/public?command=returnTicker'
                        ) as url:
            data = json.loads(url.read().decode('UTF-8'))
        return data

    @staticmethod
    def quote_history(coin="BTC_ETH", period=900, start=00, end=9999999999):
        """Poloniex public chart data with candlesticks."""
        # start = int(time.time())
        with urllib.request.urlopen(
            'https://poloniex.com/public?command=returnChartData&currencyPair={}&start={}&end={}&period={}'.format(
            coin, start, end, period
            )
                        ) as url:
            data = json.loads(url.read().decode('UTF-8'))
        return data

    def balances(self):
        """Print Personal Balances."""
        # print("Checking Balances")
        req = {}
        balances = dict()
        req['command'] = 'returnBalances'
        req['nonce'] = int(time.time() * 1000)
        post_data = urllib.parse.urlencode(req).encode('ASCII')

        sign = hmac.new(self.options.poloniex_secret.encode('ASCII'),
                        post_data,
                        hashlib.sha512).hexdigest()
        headers = {
            'Key': self.options.poloniex_api,
            'Sign': sign,
        }
        with urllib.request.urlopen(urllib.request.Request(
                'https://poloniex.com/tradingApi',
                post_data,
                headers)
                ) as url:
            data = json.loads(url.read().decode('UTF-8'))
            for i in data:
                j = data[i]
                balances[i] = j
        return balances

    def trade_history(self, coin='BTC_XEM'):
        """Specific trade history."""
        history = dict()
        req = {'currencyPair': coin}
        req['command'] = 'returnTradeHistory'
        req['nonce'] = int(time.time() * 1000)
        post_data = urllib.parse.urlencode(req).encode('ASCII')
        sign = hmac.new(self.options.poloniex_secret.encode('ASCII'),
                        post_data,
                        hashlib.sha512).hexdigest()
        headers = {
            'Key': self.options.poloniex_api,
            'Sign': sign,
        }
        with urllib.request.urlopen(urllib.request.Request(
            'https://poloniex.com/tradingApi',
            post_data,
            headers)) as url:
            data = json.loads(url.read().decode('UTF-8'))
        print_msg = "##############################\n"
        print_msg += "# Trade history for {} #\n".format(coin)
        for i in data:
            for j in i:
                print_msg += "==# {}: {}\n".format(j, i[j])
                history[j] = i[j]
        print_msg += "##############################\n"
        # print(print_msg)
        return history

    def buy(self, rate=0, amount=0, coin='BTC_ETH'):
        """Buy a coin on poloniex."""
        # return an 'coin': order_number dict
        _order_number = dict()
        req = {'currencyPair': coin,
               'rate': rate,
               'amount': amount}

        req['command'] = 'buy'
        req['nonce'] = int(time.time() * 1000)
        post_data = urllib.parse.urlencode(req).encode('ASCII')
        sign = hmac.new(self.options.poloniex_secret.encode('ASCII'),
                        post_data,
                        hashlib.sha512).hexdigest()
        headers = {
            'Key': self.options.poloniex_api,
            'Sign': sign,
        }
        with urllib.request.urlopen(urllib.request.Request(
            'https://poloniex.com/tradingApi',
            post_data,
            headers)
                                    ) as url:
            order_number = json.loads(url.read().decode('UTF-8'))
        print_msg = "##############################\n"
        print_msg += "# Buy order placed for {}\n".format(coin)
        print_msg += "==# rate: {}\n".format(rate)
        print_msg += "==# amount: {}\n".format(amount)
        total = float(amount) * float(rate)
        fee = total * .0025
        print_msg += "==# fee(0.25%): {0:.8f}\n".format(fee)
        print_msg += "==# total: {0:.8f}\n".format(
            fee + total)
        for i in order_number:
            if i == 'error':
                print("==# {}: {}".format(i, order_number[i]))
                print("!!# {} :: {} !! NOT PURCHASED.".format(amount, coin))
                return False
        # order completed successfully
        _order_number[coin] = order_number[i]
        print_msg += "==# {}".format(_order_number)
        print_msg += "##############################\n"
        print(print_msg)
        return _order_number

    def sell(self, amount, rate, coin):
        """Buy a coin on poloniex."""
        req = {'currencyPair': coin,
               'rate': rate,
               'amount': amount}

        req['command'] = 'sell'
        req['nonce'] = int(time.time() * 1000)
        post_data = urllib.parse.urlencode(req).encode('ASCII')
        sign = hmac.new(self.options.poloniex_secret.encode('ASCII'),
                        post_data,
                        hashlib.sha512).hexdigest()
        headers = {
            'Key': self.options.poloniex_api,
            'Sign': sign,
        }
        with urllib.request.urlopen(urllib.request.Request(
            'https://poloniex.com/tradingApi',
            post_data,
            headers)
                                    ) as url:
            order_number = json.loads(url.read().decode('UTF-8'))
        print_msg = "##############################\n"
        print_msg += "# Sell order placed for {}\n".format(coin)
        print_msg += "==# rate: {}\n".format(rate)
        print_msg += "==# amount: {}\n".format(amount)
        total = float(amount) * float(rate)
        fee = total * .0025
        print_msg += "==# fee(0.25%): {0:.8f}\n".format(fee)
        print_msg += "==# total: {0:.8f}\n".format(
            fee + total)
        for i in order_number:
            if i == 'error':
                print("==# {}: {}".format(i, order_number[i]))
                print("!!# {} :: {} !! NO Sell order Placed.".format(
                    amount, coin))
                return False
        print_msg = "##############################\n"
        print(print_msg)
        return print_msg

    def cancel(self, coin, order):
        """Buy a coin on poloniex."""
        req = {'currencyPair': coin,
               'orderNumber': order[coin]}

        req['command'] = 'cancelOrder'
        req['nonce'] = int(time.time() * 1000)
        post_data = urllib.parse.urlencode(req).encode('ASCII')
        sign = hmac.new(self.options.poloniex_secret.encode('ASCII'),
                        post_data,
                        hashlib.sha512).hexdigest()
        headers = {
            'Key': self.options.poloniex_api,
            'Sign': sign,
        }
        with urllib.request.urlopen(urllib.request.Request(
            'https://poloniex.com/tradingApi',
            post_data,
            headers)
                                    ) as url:
            order_cancel_bool = json.loads(url.read().decode('UTF-8'))
        ## END API CALL
        print_msg = "##############################\n"
        print_msg += "# Order Canceled for {}\n".format(coin)
        for i in order_number:
            if i == 'error':
                print("==# {}: {}".format(i, order_number[i]))
                print("!!# {} :: {} !! NO Sell order Placed.".format(
                    amount, coin))
                return False
        print_msg = "##############################\n"
        print(print_msg)
        return print_msg

    def withdraw(self, dest, amount=0.01, coin='BTC'):
        """Withdraw funds from poloniex."""
        pass

def main():
    """Launcher for the app."""
    options = Options()
    app = Trader(options)
    x = 0
    while x < 1:
        # app.call_ticker()
        test = app.balances()
        # app()
        # app.trade_history()
        # app.buy(amount=1, rate=.001, coin='BTC_ETH')
        # app.sell(amount=8, rate=.00000130, coin='BTC_DOGE')
        # order = app.buy()

        # if app.cancel(order):
        #     print("Order {}.. !! canceled".format(order))
        # time.sleep(.05)
        #
        # history = app.quote_history()
        print(test)
        x += 1

    if app.main():
        sys.exit('Alphagriffin.com | 2017')
    return True


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print("and thats okay too.")
        sys.exit(e)
