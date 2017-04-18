from flask import Flask, render_template, request, redirect
import requests
import simplejson as json
from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.resources import INLINE
import numpy as np
from bokeh.util.string import encode_utf8
import bokeh
# from settings import SRC_TEMPLATE_US_MAP
import sys
import os

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

    # get data
    api_url = 'https://www.quandl.com/api/v3/datatables/WIKI/PRICES.json?api_key=Vs3VjGfhRTx_7Pbu2sZ_'
    session = requests.Session()
    session.mount('http://', requests.adapters.HTTPAdapter(max_retries=3))
    raw_data = session.get(api_url).json()
    
    # transform the data
    data_j = json.dumps(raw_data)
    data = json.loads(data_j)

    #data informations
    lendata = len(data['datatable']['data'])
    ticker = [data['datatable']['data'][i][0] for i in range(0, lendata)]

    distinct_ticker = ', '.join(set(ticker))

    if request.method == 'GET':

        # downloadData()
        # grab the static resources
        js_resources = INLINE.render_js()
        css_resources = INLINE.render_css()

        script, div = da.templateUsMapPercAcceptedLoan()

        html = render_template(
            'new_index_map.html',
            plot_script=script,
            plot_div=div,
            js_resources=js_resources,
            css_resources=css_resources
        )
        return encode_utf8(html)

        # return render_template('loan_perc_states_map.html')
        # return render_template('index.html',
        #                        distinct_ticker=distinct_ticker)
    else:
        
        #get data interted by user
        app.vars['ticker_type'] = request.form['ticker_type']

        #select the data I need 
        opend = [data['datatable']['data'][i][2] for i in range(0, lendata)]
        high = [data['datatable']['data'][i][3] for i in range(0, lendata)]
        low = [data['datatable']['data'][i][4] for i in range(0, lendata)]
        close = [data['datatable']['data'][i][5] for i in range(0, lendata)]
        date = [data['datatable']['data'][i][1] for i in range(0, lendata)]

        #filter data
        selected_ticker = app.vars['ticker_type']
        print_date = np.array([date[i] for i in range(0, lendata) if ticker[i] == selected_ticker], dtype=np.datetime64)
        print_close = [close[i] for i in range(0, lendata) if ticker[i] == selected_ticker]
        print_opend = [opend[i] for i in range(0, lendata) if ticker[i] == selected_ticker]
        print_high = [high[i] for i in range(0, lendata) if ticker[i] == selected_ticker]

        #plot data
        p1 = figure(title='', x_axis_type="datetime")
        p1.xaxis.axis_label = 'Date'
        p1.yaxis.axis_label = 'Closing values'
        p1.line(print_date, print_close)

        # grab the static resources
        js_resources = INLINE.render_js()
        css_resources = INLINE.render_css()

        #grap component 
        script, div = components(p1)

        return render_template(
            'new_index_map.html',
            plot_script=script,
            plot_div=div,
            js_resources=js_resources,
            css_resources=css_resources,
            ticker_type=selected_ticker
        )
        # return encode_utf8(html)


if __name__ == '__main__':
    app.run(port=33507)
    #app.run(host='0.0.0.0')
