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

# Stats
from ag.bittensor.game.stats import Stats

class GameEngine(object):
    """
    The Game Engine is the mechinizm
    """

    def __init__(self, options):
        self.options = options

        # Hold the Game Stats Here.
        self.stats = Stats(options)

    def main(self):
        """Sanity Check."""
        if self.options:
            return True
        return False
