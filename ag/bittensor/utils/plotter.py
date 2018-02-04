#!/usr/bin/python3
"""
CryptoPotter.
"""

# Common Stuffs
import os, sys, time, datetime, collections

# Graphing Stuffs
from bokeh.io import push_notebook, show, output_notebook
from bokeh.layouts import row, layout
from bokeh.plotting import figure, output_file
from bokeh.models import ColumnDataSource, HoverTool, LinearAxis, Range1d, Toggle, BoxAnnotation, CustomJS
from bokeh.sampledata.autompg import autompg_clean as testset # sample data for testing
from bokeh.transform import factor_cmap

__author__ = "Eric Petersen @Ruckusist"
__copyright__ = "Copyright 2017, The Alpha Griffin Project"
__credits__ = ["Eric Petersen", "Shawn Wilson", "@alphagriffin"]
__license__ = "***"
__version__ = "0.0.2"
__maintainer__ = "Eric Petersen"
__email__ = "ruckusist@alphagriffin.com"
__status__ = "Beta"


class Plot(object):

    def __init__(self, options):
        self.options = options
        self.in_notebook = False

    def output_notebook_():
        self.in_notebook = True

    @staticmethod
    def Plot_me(dataframe, to_file=False):
        """
        Add up all the bokeh tutorials into one nice looking web page.
        """
        TOOLS = "crosshair,pan,wheel_zoom,box_zoom,reset,save,box_select"
        df = dataframe
        source = ColumnDataSource(df)

        p = figure(plot_width = 600,
                   plot_height = 350,
                   # title = "{} | {}".format(this_exchange.id, pair),
                   title = "AlphaGriffin CryptoPlotter",
                   # y range adjusted to fit on screen nicer
                   y_range = (df['close'].min()*.999, df['close'].max()*1.001),
                   # Axis Types
                   x_axis_type = "datetime",
                   #y_axis_type = "log",
                   toolbar_location = 'below',
                   tools = TOOLS
                  )

        # formating
        p.xgrid.grid_line_color = None  # 'navy'
        p.ygrid.grid_line_color = 'navy'
        p.ygrid.grid_line_alpha = .1
        p.xaxis.axis_label = "Coins Grouped By Exchange"
        p.xaxis.major_label_orientation = 1.2

        p.add_tools(
            HoverTool(
                tooltips=[
                    ("Close", "@close{%.8f}"),
                    ("Time", "@time_str{%F}"),
                    ("Volume", "@vol{0.00 a}"),
                ],
                formatters={
                    "Time": 'datetime',
                    "Close": 'printf'
                },
                mode='vline'
            )
        )

        # plots
        price_chart = p.vbar(
            x='time', top='close', color="navy", alpha=0.4, width=.2, source=source, legend='Price USD'
        )
        #p.square(x='time', y='close', color="firebrick", alpha=0.4, size=4, source=source)
        #p.line(x='time', y='close', line_width=2, color='navy', alpha=.8, source=source)

        # add volume plot over the top
        # FORMATTING
        p.extra_y_ranges = {"foo": Range1d(
            start=df['vol'].min(), end=df['vol'].max()
        )}
        p.add_layout(LinearAxis(y_range_name="foo"), 'left')

        # Volume plot
        vol_chart = p.line(
            x='time', y='vol', line_width=2, color='red', alpha=.8, source=source, y_range_name="foo", legend='Volume'
        )

        # Buttons for Charts
        hide_chart_code = """object.visible = toggle.active"""
        callback_1 = CustomJS.from_coffeescript(code=hide_chart_code, args={})
        button_1 = Toggle(label="Price Chart", button_type="success", callback=callback_1)
        callback_1.args = {'toggle': button_1, 'object': price_chart}

        callback_2 = CustomJS.from_coffeescript(code=hide_chart_code, args={})
        button_2 = Toggle(label="Volume Chart", button_type="success", callback=callback_2)
        callback_2.args = {'toggle': button_2, 'object': vol_chart}

        # show and or save to html
        dir_path = os.path.dirname(os.path.realpath(__file__))
        filename = "gf_{}_{}_{}.html".format(df.exchange, df.pair, datetime.datetime.now().strftime('%m_%d'))

        if to_file:
            output_file(os.path.join(os.getcwd(), 'html', 'graphs', filename))
        if not self.in_notebook:
            show(layout(
                    [p], [button_1, button_2]
                    # [button for button in buttons]
                    ))
            return filename
        else:
            output_notebook()
            return layout(
                [p], [button_1, button_2]
                # [button for button in buttons]
                )
