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
    return redirect('/welcome')


@app.route('/welcome', methods=['GET', 'POST'])
def welcome():
    if request.method == 'GET':

        js_resources = INLINE.render_js()
        css_resources = INLINE.render_css()

        script, div = da.templateUsMapPercAcceptedLoan()

        script_time, div_time = da.templateAcceptedLoanPerRegion()

        script_roc_c002812, div_roc_c002812 = da.templateROC(c='c002812')
        script_roc_c00001, div_roc_00001 = da.templateROC(c='c00001')

        coeff_values = da.templateCoefRegression()

        return render_template(
            'welcome.html',
            js_resources=js_resources,
            css_resources=css_resources,
            plot_script=script,
            plot_div=div,
            plot_script_time=script_time,
            plot_div_time=div_time,
            plot_script_roc_1=script_roc_c00001,
            plot_div_roc_1=div_roc_00001,
            plot_script_roc_2=script_roc_c002812,
            plot_div_roc_2=div_roc_c002812,
            plot_script_m=coeff_values['month'][0],
            plot_div_m=coeff_values['month'][1],
            plot_script_e=coeff_values['empl'][0],
            plot_div_e=coeff_values['empl'][1],
            plot_script_s=coeff_values['state'][0],
            plot_div_s=coeff_values['state'][1],
            plot_script_o=coeff_values['other'][0],
            plot_div_o=coeff_values['other'][1],
        )

    else:

        # grab the static resources
        js_resources = INLINE.render_js()
        css_resources = INLINE.render_css()

        if 'map' in request.data:
            js_resources = INLINE.render_js()
            css_resources = INLINE.render_css()

            script, div = da.templateUsMapPercAcceptedLoan()

            return render_template(
                'index.html',
                plot_script=script,
                plot_div=div,
                js_resources=js_resources,
                css_resources=css_resources,
                # ticker_type=selected_ticker
            )
            # return encode_utf8(html)


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
def info_state(state):
    js_resources = INLINE.render_js()
    css_resources = INLINE.render_css()

    script_corr, div_corr = da.templateRateCorrelation(state)

    return render_template(
        'info_state.html',
        plot_script_corr=script_corr,
        plot_div_corr=div_corr,
        js_resources=js_resources,
        css_resources=css_resources,
        state=state
    )


@app.route('/perc_over_time')
def perc_over_time():
    js_resources = INLINE.render_js()
    css_resources = INLINE.render_css()

    script_time, div_time = da.templateAcceptedLoanPerRegion()

    return render_template(
        'perc_over_time.html',
        plot_script=script_time,
        plot_div=div_time,
        js_resources=js_resources,
        css_resources=css_resources
    )


if __name__ == '__main__':
    app.run(port=33507)
    # app.run(host='0.0.0.0')
