#!/usr/bin/python3
"""
Bittensor.
by: AlphaGriffin
"""

__author__ = "Eric Petersen @Ruckusist"
__copyright__ = "Copyright 2018, The Alpha Griffin Project"
__credits__ = ["Eric Petersen", "Shawn Wilson", "@alphagriffin"]
__license__ = "***"
__version__ = "0.0.1"
__maintainer__ = "Eric Petersen"
__email__ = "ruckusist@alphagriffin.com"
__status__ = "Beta"

#////////////////// | Imports | \\\\\\\\\\\\\\\#
# generic
import os, sys, time, datetime, collections

import pandas as pd
import numpy as np


class DataStruct(pd.DataFrame):
    """
    Source:
        http://blog.snapdragon.cc/2015/05/05/subclass-pandas-dataframe-to-save-custom-attributes/
    """

    def __init__(self, *args, **kw):
        super(DataStruct, self).__init__(*args, **kw)

    @property
    def _constructor(self):
        return DataStruct

    @property
    def _constructor_sliced(self):
        return pandas.Series

    def fuckTest(self):
        for i in self:
            print(i)
        return True



class DataHandler(object):
    """This Should handle the webside data collections; and other stuff too."""

    def __init__(self):
        self.data = DataStruct()

    def main(self):
        if self.data.fuckTest():
            print('HA IT WORKS!.')
            return True
        return True
