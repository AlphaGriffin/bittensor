#!/usr/bin/python3
"""
Bittensor. DataGrabber.
by: AlphaGriffin
"""

__author__ = "Eric Petersen @Ruckusist"
__copyright__ = "Copyright 2018, The Alpha Griffin Project"
__credits__ = ["Eric Petersen", "Shawn Wilson", "@alphagriffin"]
__license__ = "***"
__version__ = "0.1.0"
__maintainer__ = "Eric Petersen"
__email__ = "ruckusist@alphagriffin.com"
__status__ = "Beta"

# generic
import os, sys, time, datetime, collections, re, random, pickle
from timeit import default_timer as timer
from tqdm import tqdm, trange
import csv
import ccxt
import pandas

import ag.bittensor.utils.options as options
import ag.bittensor.ai.make_data as make_data

def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name, 'rb') as f:
        return pickle.load(f)

def load_file(name, columns=list):
    pass

if False:  # DEPRICATED
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

# # START UP # #
print('BITTENSOR DATA GRABBER.')
runtimer = timer()

## MAKE GLOBALS
CONFIG = options.Options('config/access_codes.yaml')
datasmith = make_data.MakeData(CONFIG)

# ESTABLISH VARIABLES
exchanges = list(CONFIG.exchanges.split(','))
datadir = 'exchanges'
cur_path = os.getcwd()
exchanges_path = os.path.join(cur_path, datadir)

# ESTABLISH FILESYSTEM
FILESYSTEM = {}
if not os.path.exists(exchanges_path):
    print('building {} directory'.format(exchanges_path))
    os.mkdir(exchanges_path)
FILESYSTEM['exchanges_path'] = exchanges_path

# start exchange folders
for exchange_name in exchanges:
    exchange = eval('ccxt.{}()'.format(exchange_name))
    eid = exchange.id
    exchange_path = os.path.join(exchanges_path, eid)
    if not os.path.exists(exchange_path):
        print('building {} directory'.format(exchange_path))
        os.mkdir(exchange_path)
    FILESYSTEM['{}_path'.format(eid)] = exchange_path

# start coins folder
coins_folder = 'coins_info'
coins_path = os.path.join(cur_path, coins_folder)
if not os.path.exists(coins_path):
    print('building {} directory'.format(coins_path))
    os.mkdir(coins_path)
FILESYSTEM[coins_folder] = coins_path

if False:
    for i in FILESYSTEM:
        print('{}: {}'.format(i, FILESYSTEM[i]))
print('Finished Establishing Filesystem.')

# GET COINS LIST
coins_dataset = {}
coins_dataset_filename = 'all_coins_data.pkl'
coins_dataset_path = os.path.join(FILESYSTEM['coins_info'], coins_dataset_filename)
if not os.path.exists(coins_dataset_path):
    print('Downloading Lists of Coins for all exchanges')
    for exchange_name in exchanges:
        exchange = eval('ccxt.{}()'.format(exchange_name))
        eid = exchange.id
        MARKETS = exchange.fetchMarkets()
        symbol_list = []
        for i in MARKETS:
            coin, base = i['symbol'].split('/')
            try:
                if coins_dataset[coin]: pass
            except:
                coins_dataset[coin] = {}

            try:
                if coins_dataset[coin]['exchanges']: pass
            except:
                 coins_dataset[coin]['exchanges'] = {}

            try:
                if coins_dataset[coin]['exchanges'][eid]: pass
            except:
                coins_dataset[coin]['exchanges'][eid] = {}

            try:
                if coins_dataset[coin]['exchanges'][eid]['pairs']: pass
            except:
                coins_dataset[coin]['exchanges'][eid]['pairs'] = []
            coins_dataset[coin]['exchanges'][eid]['pairs'].append(base)
    save_obj(coins_dataset, coins_dataset_path)
else:
    print('Loading Saved Lists of Coins for all exchanges')
    coins_dataset = load_obj(coins_dataset_path)

FILESYSTEM['all_coins_info'] = coins_dataset_path
if False:
    for c in sorted(coins_dataset):
        print('COIN:', c)
        for e in sorted(coins_dataset[c]['exchanges']):
            print('\tEXCHANGE:', e)
            print('\tTrade Pairs:')
            print('\t\t','\n\t\t'.join([p for p in coins_dataset[c]['exchanges'][e]['pairs']]))
print('Finished Establishing CoinsList.')
############## /coins list

# get historical data
FAILOUT = 0
for exchange_name in exchanges:
    exchange = eval('ccxt.{}()'.format(exchange_name))
    eid = exchange.id
    exchange_path = os.path.join(exchanges_path, eid)
    MARKETS = exchange.fetchMarkets()

    sizecounter = 0
    with tqdm(
        total=len(MARKETS),
        unit='Pair', unit_scale=False,
        ) as pbar:
        # for i in MARKETS:
        while sizecounter <= len(MARKETS)-1:
            try:
                i = MARKETS[sizecounter]
                pair = i['symbol']
                coin, base = pair.split('/')
                filename = '{}_{}_{}.csv'.format(eid, coin, base)
                pbar.set_postfix(file=filename[:-4], refresh=False)
                filepath = os.path.join(
                    FILESYSTEM['{}_path'.format(eid)], filename
                )
                if os.path.exists(filepath):
                    # tqdm.write('Already Collected {} | {}'.format(eid, pair))
                    sizecounter += 1
                    pbar.update(1)
                    continue
                # continue
                time_frame = '5m' if 'poloniex' in eid else '1m'
                historicial_data = exchange.fetch_ohlcv(pair, timeframe=time_frame, since=0, limit=time.time())
                write_time = timer()
                with open(filepath, 'a', newline='') as csvfile:
                    columns = ['timestamp','Open','High','Low','Close','Volume']
                    writer = csv.DictWriter(csvfile, fieldnames=columns)
                    for i in historicial_data:
                        writer.writerow({
                            'timestamp': i[0],
                            'Open': i[1],
                            'High': i[2],
                            'Low': i[3],
                            'Close': i[4],
                            'Volume': i[5]
                        })
                write_time = timer() - write_time
                sleeptime = 3 - write_time
                # tqdm.write('Took {:.2f} secs to write historical data for {} | {}'.format(
                #    write_time, eid, pair
                #))
                tqdm.write('Sleeping for {:.2f} secs'.format(sleeptime), end='\r')
                time.sleep(sleeptime)
                sizecounter += 1
                pbar.update(1)
            except:
                tqdm.write('[{}] Probably failed for a reason. Waiting 20 secs for retry. error {}/5'.format(
                    datetime.datetime.now(), FAILOUT
                ))
                time.sleep(20)
                if FAILOUT >= 4:
                    sizecounter += 1
                    pbar.update(1)
                    tqdm.write('Bailing on Pair: {}'.format(pair))
                    FAILOUT = 0
                else:
                    FAILOUT += 1
                pass
print('Finished Downloading all historical Data.')

print('Get Coin data!')
for i in coins_dataset:
    # print(i)
    pass
