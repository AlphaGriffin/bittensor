
# AlphaGriffin BitTensor
<p align="center">
![AlphaGriffin Logo](/images/logo.png)
</p>

Tensorflow vs. Bitcoin. Currently in development by 2 guys. if you would like
to help out with the project please donate Litecoin to the following address.
Litecoin is our preferred transactional coin for its low fees and segwit2x.

```
LTC: 33jckQAojmGrK2ZG9ZnA6z9hXAjK1mfAhR
```

## Getting Started

These instructions will get you a copy of the project up and running on your
local machine for development and testing purposes. See deployment for notes
on how to deploy the project on a live system.

### Prerequisites

Copy the dummy_codes to access_codes and add in your API keys.
Copy the dummy_secret to client_secret and add in your API keys.

```
cd config
cp dummy_codes.yaml access_codes.yaml
cp dummy_secret.json client_secret.json
```

Edit the new access_codes and client_secret file as necessary.

## Deployment

This Project won't get far without data. First edit the access_codes.yaml.
And add in all of the exchanges you plan to be working with. !! Important Note:
Bittrex full set takes up about .75GB of data.

Important Note: Not all exchange provide historical data to commoners.
```
# In access_codes.yaml
exchanges: 'poloniex,bittrex'
# then run the datagrabber
python bittensor_data.py
```

## Presentation
This is a Plotly Dash Site for data viewing.
```
python site.py
```

## Tensorflow
```
# train model
python bittensor_train.py
# backtest model
python bittensor_test.py
```

## _TODO_
most all tools are currently mid broken from the datagrabber upgrade. Everything
will need to be reevaluated one peice at a time.
Parts needing to be looked at:
* Paper Trader
    - Seperate Back Testing tools
    - Make it go fully online and start buying and selling to test transaction times
    - Make it work fully offline replicating what you would see if you were online
    - Keep really good track of trade history
    - Make the Paper Trader Growup to be a real trader some day.
    - PaperTrader Should otherwise function in Live/Fake Mode + strategy.
    - This is going to need its own file saving system.
    - Should be able to build this using the plotlyDash interface.
* BackTester and Paper Trader need to be 2 seperate things.
* Lots of TA tools left to build. The instructions are in the TALib file, just
    needs doing.
* Website:
    - Frontend for Paper Trader, Backtester
    - Navigation between Dash, Tensorboard, PaperTrader, Backtester
    - Better Graph And TA Chart building
    - Add in Coin Detail with Graph and TA charts like logo, company info,
        blockchain details
    - Pull the orderbook in from the web when a pair is selected
    - Full Resituation of the Main Presentation Page. Make it presentable.
* TensorFlow:
    - *better Data input Schema
    - reward mechinizm
    - Tensorboard
    - Build Price Prediction Estimator
    - Use Prediction in the Q-Learner
* DataSmith:
    - Needs need DataGrabber tools built into the namespace
    - Needs to save all the files to Google Drive.
    - Needs Other Data to grab. Twitter, Steam, Facebook, News, reddit, bitcointalk
        Company information, linkedin
* Blockchain Interface:
    - UNSTARTED
* Local Wallets:
    - UNSTARTED

## Update Log

V 0.0.1: bittensor_data.py is now working as expected with the datasmith and
the TA lib. This can be explored in the TA_Samples and DataSmith Notebooks.
