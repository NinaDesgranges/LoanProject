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
from bokeh.resources import INLINE
from bokeh.util.string import encode_utf8

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


# def downloadData():
#     import bokeh
#     bokeh.sampledata.download()


@app.route('/')
def main():
    return redirect('/index')


@app.route('/index', methods=['GET', 'POST'])
def index():
    # # get data
    # api_url = 'https://www.quandl.com/api/v3/datatables/WIKI/PRICES.json?api_key=Vs3VjGfhRTx_7Pbu2sZ_'
    # session = requests.Session()
    # session.mount('http://', requests.adapters.HTTPAdapter(max_retries=3))
    # raw_data = session.get(api_url).json()
    #
    # # transform the data
    # data_j = json.dumps(raw_data)
    # data = json.loads(data_j)
    #
    # # data informations
    # lendata = len(data['datatable']['data'])
    # ticker = [data['datatable']['data'][i][0] for i in range(0, lendata)]
    #
    # distinct_ticker = ', '.join(set(ticker))

    if request.method == 'GET':

        js_resources = INLINE.render_js()
        css_resources = INLINE.render_css()

        # script, div = da.templateUsMapPercAcceptedLoan()

        # html = render_template(
        #     'new_index.html',
        #     plot_script=script,
        #     plot_div=div,
        #     js_resources=js_resources,
        #     css_resources=css_resources
        # )







        new_ds = pd.read_csv(DATA + 'perc_acc_loan_per_state.csv', header=0)
        new_ds = new_ds.set_index('state')

        # Blues9.reverse()
        my_col = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#084594']
        cm = LinearColorMapper(palette=['#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#084594'],
                               low=min(new_ds.perc_acc_loan.values), high=max(new_ds.perc_acc_loan.values))

        states = json.load(open(DATA + 'boundaries.json', 'r'))
        # states = us_states.data.copy()
        # states = sta.copy()

        # f = open(DATA + "boundaries.json", "w")
        # f.write(str(states))
        # f.close()

        # del states["HI"]
        # del states["AK"]

        state_xs = [states[code]["lons"] for code in states]
        state_ys = [states[code]["lats"] for code in states]

        source = ColumnDataSource(data=dict(
            x=state_xs,
            y=state_ys,
            name=new_ds.index.get_values(),
            rate=new_ds['perc_acc_loan'],
        ))

        # output_file(TEMPLATE + "loan_perc_states_map.html", title="Loan Acceptance Rate")

        p = figure(title="Loan Acceptance Rate", toolbar_location="left",
                   plot_width=900, plot_height=573)

        p.patches('x', 'y', source=source,
                  fill_color={'field': 'rate', 'transform': cm},
                  fill_alpha=1, line_color="white", line_width=0.5)

        color_bar = ColorBar(color_mapper=cm,
                             orientation='vertical',
                             location=(0, 0))

        p.add_layout(color_bar, 'right')

        # grap component
        script, div = components(p)

        return render_template(
            'ciao.html',
            # plot_script=script,
            # plot_div=div,
            js_resources=js_resources,
            css_resources=css_resources
        )

        # return render_template(
        #     'new_index_map.html',
        #     plot_script=script,
        #     plot_div=div,
        #     js_resources=js_resources,
        #     css_resources=css_resources
        # )

        # return encode_utf8(html)
    #
    # return render_template('loan_perc_states_map.html')
    # return render_template('index.html',
    #                        distinct_ticker=distinct_ticker)
    else:

        # get data interted by user
        # app.vars['ticker_type'] = request.form['ticker_type']
        #
        # # select the data I need
        # opend = [data['datatable']['data'][i][2] for i in range(0, lendata)]
        # high = [data['datatable']['data'][i][3] for i in range(0, lendata)]
        # low = [data['datatable']['data'][i][4] for i in range(0, lendata)]
        # close = [data['datatable']['data'][i][5] for i in range(0, lendata)]
        # date = [data['datatable']['data'][i][1] for i in range(0, lendata)]
        #
        # # filter data
        # selected_ticker = app.vars['ticker_type']
        # print_date = np.array([date[i] for i in range(0, lendata) if ticker[i] == selected_ticker], dtype=np.datetime64)
        # print_close = [close[i] for i in range(0, lendata) if ticker[i] == selected_ticker]
        # print_opend = [opend[i] for i in range(0, lendata) if ticker[i] == selected_ticker]
        # print_high = [high[i] for i in range(0, lendata) if ticker[i] == selected_ticker]
        #
        # # plot data
        # p1 = figure(title='', x_axis_type="datetime")
        # p1.xaxis.axis_label = 'Date'
        # p1.yaxis.axis_label = 'Closing values'
        # p1.line(print_date, print_close)

        # grab the static resources
        js_resources = INLINE.render_js()
        css_resources = INLINE.render_css()

        # grap component
        # script, div = components(p1)

        return render_template(
            'new_index_map.html',
            # plot_script=script,
            # plot_div=div,
            js_resources=js_resources,
            css_resources=css_resources,
            # ticker_type=selected_ticker
        )
        # return encode_utf8(html)


if __name__ == '__main__':
    app.run(port=33507)
    # app.run(host='0.0.0.0')
