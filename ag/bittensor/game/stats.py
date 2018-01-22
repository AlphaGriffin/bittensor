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


class Stats(object):
    """
    A mechinizm that holds the games variables, positions, data,
    can parse the options for limitations, holds all golbal booleans.
    """

    def __init__(self, options):
        """Build out all the stats here."""
        # build Objects
        self.__options = options

        # Actual Stats.
