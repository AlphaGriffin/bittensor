#!/usr/bin/python3
"""
Bittensor.
by: AlphaGriffin



## TODO:: add btc_volume and coin_volume its current vol times last
"""
import os, time
from timeit import default_timer as timer
import csv
import ccxt

columns = [
    'timestamp', 'high', 'low',
    'last', 'change', 'baseVolume'
    ]

def write_files(ticker, loop_num):
    for pair in ticker:
        _pair = '{}_{}'.format(pair.split('/')[0], pair.split('/')[1])
        filename = '{}.csv'.format(_pair)
        filepath = os.path.join(os.getcwd(), 'data', 'files', filename)
        with open(filepath, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns)
            writer.writerow({
                'timestamp': ticker[pair]['timestamp'],
                'high': ticker[pair]['high'],
                'low': ticker[pair]['low'],
                'last': ticker[pair]['last'],
                'change': ticker[pair]['change'],
                'baseVolume': ticker[pair]['baseVolume'],
            })
    return True

exchange_name = 'bittrex'
timeout = 60
exchange = eval('ccxt.{}()'.format(exchange_name))
_file_num = 0
_loop_num = 0

while True:
    try:
        start = timer()
        _ticker = exchange.fetch_tickers()
        if write_files(_ticker, _loop_num):
            took = timer() - start
        if took > timeout:
            _file_num += 1
        print('took {:.2f}, sleeping for {:.2f}, gathered data: {}'.format(
                            took, timeout-took, _loop_num
                            )
        )
    except Exception as e:
        # make a break for keyboard int
        took = 2
        print('Couldnt Download, Skipping...')
        pass
    time.sleep(timeout-took)
    _loop_num += 1
