import simplejson
from flask import Flask, render_template, request, redirect
import requests
import simplejson as json
from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.resources import INLINE
import numpy as np
import pandas as pd
from bokeh.util.string import encode_utf8
import bokeh
# from settings import SRC_TEMPLATE_US_MAP
import sys
import os
from settings import DATA
from bokeh.embed import components
from bokeh.resources import INLINE, CDN
from bokeh.util.string import encode_utf8
import bokeh.sampledata.us_states
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    ColorMapper,
    LinearColorMapper,
    ColorBar,
    PrintfTickFormatter,
    BasicTicker
)
from bokeh.palettes import Viridis6 as palette
from bokeh.plotting import figure
from src import data_analysis as da

app = Flask(__name__)
app.vars = {}


@app.route('/')
def main():
    return redirect('/index')



@app.route('/index', methods=['GET', 'POST'])
def index():

    if request.method == 'GET':

        js_resources = INLINE.render_js()
        css_resources = INLINE.render_css()

        script, div = da.templateUsMapPercAcceptedLoan()

        return render_template(
            'us_map.html',
            plot_script=script,
            plot_div=div,
            js_resources=js_resources,
            css_resources=css_resources
        )

    # return encode_utf8(html)

    else:

        # grab the static resources
        js_resources = INLINE.render_js()
        css_resources = INLINE.render_css()

        # grap component
        # script, div = components(p1)

        return render_template(
            'us_map.html',
            # plot_script=script,
            # plot_div=div,
            js_resources=js_resources,
            css_resources=css_resources,
            # ticker_type=selected_ticker
        )
        # return encode_utf8(html)


@app.route('/info_<state>/')
def info_CA(state):

    js_resources = INLINE.render_js()
    css_resources = INLINE.render_css()

    script_corr, div_corr = da.templateRateCorrelation(state)

    return render_template(
        'info_state.html',
        plot_script_corr=script_corr,
        plot_div_corr=div_corr,
        js_resources=js_resources,
        css_resources=css_resources
    )


if __name__ == '__main__':
    app.run(port=33507)
    # app.run(host='0.0.0.0')
