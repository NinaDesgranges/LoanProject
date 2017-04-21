from datetime import datetime
import pandas as pd
from settings import DATA, DATA_LOCAL
import json
import numpy as np
from bokeh.io import show

# from bokeh.sampledata import us_states
# from bokeh.sampledata.us_states import data as sta
from bokeh.sampledata import us_states
from bokeh.plotting import figure, show, output_file
from flask import Flask, render_template, request, redirect
from bokeh.embed import components
from bokeh.resources import INLINE
from bokeh.util.string import encode_utf8
from bokeh.models.callbacks import CustomJS
from bokeh.models.widgets import Select
from bokeh.layouts import row, column
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    ColorMapper,
    LinearColorMapper,
    ColorBar,
    FuncTickFormatter,
    FixedTicker,
    PrintfTickFormatter,
    BasicTicker,
    OpenURL,
    TapTool
)
from bokeh.palettes import Viridis6 as palette
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.plotting import Figure

from bokeh.models.widgets import Select, CheckboxGroup
from bokeh.models.layouts import HBox, VBox
import bokeh.io
from bokeh.models import CustomJS
import os.path

ACC_REF_HEADER = ['title', 'amnt', 'zip', 'state', 'emp_len', 'dti', 'date', 'loan']


def createNewDataset():
    '''
    Group all dataset in accepted.csv and refused.csv
    :return:
    '''
    path_ldata1 = DATA + "LoanStats3a.csv"
    ldata1 = pd.read_csv(path_ldata1, sep=',', header=0, low_memory=False)

    path_ldata2 = DATA + "LoanStats3b.csv"
    ldata2 = pd.read_csv(path_ldata2, sep=',', header=0, low_memory=False)

    path_ldata3 = DATA + "LoanStats3c.csv"
    ldata3 = pd.read_csv(path_ldata3, sep=',', header=0, low_memory=False)

    path_ldata4 = DATA + "LoanStats3d.csv"
    ldata4 = pd.read_csv(path_ldata4, sep=',', header=0, low_memory=False)

    ldata = pd.concat([ldata1, ldata2, ldata3, ldata4])

    pd.DataFrame(ldata).to_csv(DATA + 'accepted.csv', sep=",", index=False, quoting=True, quotechar='"')

    print ldata.shape
    print ldata.columns

    path_ldata2 = DATA + "RejectStatsA.csv"
    ldata2 = pd.read_csv(path_ldata2, sep=',', header=0, low_memory=False)

    path_ldata3 = DATA + "RejectStatsB.csv"
    ldata3 = pd.read_csv(path_ldata3, sep=',', header=0, low_memory=False)

    path_ldata4 = DATA + "RejectStatsD.csv"
    ldata4 = pd.read_csv(path_ldata4, sep=',', header=0, low_memory=False)

    ldata = pd.concat([ldata2, ldata3, ldata4])

    pd.DataFrame(ldata).to_csv(DATA + 'refused.csv', sep=",", index=False, quoting=True, quotechar='"')

    print ldata.shape
    print ldata.columns


def createDatasetForAcceptedVsRefused():
    '''
    Create a common dataset with the accepted and refused loan
    the selected columns are: ['title', 'amnt', 'zip', 'state', 'emp_len', 'dti', 'date', 'loan']
    - I filter DATE in order to have same time span
    - I modify type of data (date and dti) in order to omogenizzare the 2 datasets
    - select amont > 0
    - remove dti  1. >10^7; 2. = -1; 3. = 9999; 4. =99999 (remove 8% of data)
    - manually removeda row with no data but the 1 of the loan I added

    :return:
    '''

    # TODO: 2.fill NA values

    print 'Open refused dataset'
    refused = pd.read_csv(DATA + 'refused.csv', sep=',', low_memory=False)

    print 'Open accepted dataset'
    accepted = pd.read_csv(DATA + 'accepted.csv', sep=',', low_memory=False)

    print 'Create join dataset'
    small_acc = accepted[['purpose', 'loan_amnt', 'zip_code', 'addr_state', 'emp_length', 'dti', 'issue_d']]

    small_acc['loan'] = [1] * small_acc.shape[0]

    small_acc.columns = ACC_REF_HEADER

    small_ref = refused[
        ['Loan Title', 'Amount Requested', 'Zip Code', 'State', 'Employment Length', 'Debt-To-Income Ratio',
         'Application Date']]
    small_ref['Debt-To-Income Ratio'] = small_ref['Debt-To-Income Ratio'].map(lambda x: float(x.replace('%', '')))
    small_ref['Application Date'] = small_ref['Application Date'].map(
        lambda x: datetime.strptime(x, "%Y-%m-%d").strftime('%b-%Y') if x is not None else None)
    small_ref = small_ref[small_ref['Application Date'] != 'May-2007']
    small_ref['loan'] = [0] * small_ref.shape[0]

    small_ref.columns = ACC_REF_HEADER

    new_dataset = pd.concat([small_ref, small_acc])

    print 'Shape before removing values: ' + str(new_dataset.shape)
    new_dataset = new_dataset[new_dataset.amnt != 0]
    new_dataset = new_dataset[(new_dataset.dti < pow(10, 7)) & (new_dataset.dti != -1) & (new_dataset.dti != 9999) & (
        new_dataset.dti != 99999)]

    print 'Shape after removing values: ' + str(new_dataset.shape)

    print 'Save dataset in accepted_refused_ds.csv'
    pd.DataFrame(new_dataset).to_csv(DATA + 'accepted_refused_ds.csv', index=False, header=ACC_REF_HEADER)


def getInfo():
    ds = pd.read_csv(DATA + 'accepted_refused_ds.csv', header=0)
    print sum(ds.loan)
    print 'Percentage of accepted loan ' + str(float(sum(ds.loan)) / ds.shape[0] * 100) + '%'
    print ds.amnt.describe()
    print ds.dti.describe()
    print ds.emp_len.describe()


def templateUsMapPercAcceptedLoan():
    # ds = pd.read_csv(DATA + 'accepted_refused_ds.csv', header=0)
    #
    # aggregationState = ds[['state', 'loan']].groupby(['state'])
    # print 'There are ' + str(len(aggregationState)) + ' different states'
    # new_ds = aggregationState.agg(['count', 'sum'])
    # new_ds.reset_index(level=0, inplace=True)
    # new_ds.columns = ['state', 'requests', 'acc_loan']
    # new_ds['perc_acc_loan'] = new_ds.acc_loan / new_ds.requests
    # new_ds.to_csv(DATA + 'perc_acc_loan_per_state.csv', index=False)



    new_ds = pd.read_csv(DATA + 'perc_acc_loan_per_state.csv', header=0)

    new_ds = new_ds.set_index('state')

    boundaries = open(DATA + 'boundaries.json').read()
    states = json.loads(boundaries)

    state_xs = [states[code]["lons"] for code in states]
    state_ys = [states[code]["lats"] for code in states]
    rate = [new_ds.perc_acc_loan[code] * 100 for code in states]
    name = [states[code]["name"] + '-' + code for code in states]
    urls = ['http://theloan-app.herokuapp.com/info_' + str(code) for code in states]

    cm = LinearColorMapper(palette=['#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#084594'],
                           low=round(min(rate), 2),
                           high=round(max(rate), 2)
                           )

    source = ColumnDataSource(data=dict(
        x=state_xs,
        y=state_ys,
        name=name,
        rate=rate,
        urls=urls
    ))

    TOOLS = "pan,wheel_zoom,reset,hover,save,tap"

    p = figure(title="",
               toolbar_location="above",
               plot_width=800,
               plot_height=509,
               tools=TOOLS)

    p.patches('x', 'y', source=source,
              fill_color={'field': 'rate', 'transform': cm},
              fill_alpha=1, line_color="white", line_width=0.5)

    color_bar = ColorBar(color_mapper=cm,
                         orientation='vertical',
                         location=(0, 0),
                         # ticker=ticker,
                         # formatter=formatter
                         )

    p.add_layout(color_bar, 'right')

    hover = p.select_one(HoverTool)
    hover.point_policy = "follow_mouse"
    hover.tooltips = [
        ("State", "@name"),
        ("Loan Accepance Rate", "@rate%")
    ]

    taptool = p.select(type=TapTool)
    taptool.callback = OpenURL(url='@urls')

    # show(p)
    # grap component
    script, div = components(p)

    return script, div


def templateRateCorrelation(state):
    DEFAULT_X = ['Amount', 'Income', 'DebtToIncomeRatio']

    dati = pd.read_csv(DATA + 'accepted_less_col_small_' + state + '.csv', header=0)

    amnt = dati['amnt']
    income = dati['income']
    dti = dati['dti']
    rate = dati['rate']

    source = ColumnDataSource(
        data=dict(x=amnt, y=rate, Amount=amnt, Income=income, DebtToIncomeRatio=dti))

    codex = """
            var data = source.get('data');
            data['x'] = data[cb_obj.get('value')];//
            // var r = data[cb_obj.get('value')];
            // var {var} = data[cb_obj.get('value')];
            // //window.alert( "{var} " + cb_obj.get('value') + {var}  );
            // for (i = 0; i < r.length; i++) {{
            //     {var}[i] = r[i] ;
            //     data['{var}'][i] = r[i];
            // }}
            source.trigger('change');
        """

    callbackx = CustomJS(args=dict(source=source), code=codex)

    plot = Figure(title=None)

    # Make a line and connect to data source
    plot.circle(x="x", y="y", line_color="#F46D43", line_width=6, line_alpha=0.6, source=source)

    # yaxis_select = Select(title="Y axis:", value="Rate",
    #                       options=DEFAULT_X, callback=callbacky)

    xaxis_select = Select(title="X axis:", value="Amount",
                          options=DEFAULT_X, callback=callbackx)

    # Layout widgets next to the plot
    controls = VBox(xaxis_select)

    layout = HBox(controls, plot, width=800)

    script_corr, div_corr = components(layout)

    return script_corr, div_corr

    # code = """
    #         var data = source.get('data');
    #         var r = data[cb_obj.get('value')];
    #         var {var} = data[cb_obj.get('value')];
    #         //window.alert( "{var} " + cb_obj.get('value') + {var}  );
    #         for (i = 0; i < r.length; i++) {{
    #             {var}[i] = r[i] ;
    #             data['{var}'][i] = r[i];
    #         }}
    #         source.trigger('change');
    #     """


def templateAcceptedLoanPerRegion():
    LABELS = ["Central", "Mid - Atlantic", "NorthEast", "NorthWest", "South", "SouthEast", "SouthWest"]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#17becf']
    data = pd.read_csv(DATA + 'perc_acc_loan_per_region_date_compact.csv')

    central = data['Central']
    midatl = data['Mid-Atlantic']
    northe = data['Northeast']
    northw = data['Northwest']
    south = data['South']
    southe = data['Southeast']
    southw = data['Southwest']
    date = data['date']

    source = ColumnDataSource(data=dict(x=[datetime.strptime(d, '%b-%Y') for d in date.values],
                                        Central=central,
                                        MidAtlantic=midatl,
                                        Northeast=northe,
                                        Northwest=northw,
                                        South=south,
                                        Southeast=southe,
                                        Southwest=southw
                                        ))

    props = dict(line_width=4, line_alpha=0.8)
    p = Figure(x_axis_type="datetime", width=1200, height=380)
    p0 = p.line('x', 'Central', source=source, legend="Central", line_color=colors[0], **props)
    p1 = p.line('x', 'MidAtlantic', source=source, legend="Mid - Atlantic", line_color=colors[1], **props)
    p2 = p.line('x', 'Northeast', source=source, legend="NorthEast", line_color=colors[2], **props)
    p3 = p.line('x', 'Northwest', source=source, legend="NorthWest", line_color=colors[3], **props)
    p4 = p.line('x', 'South', source=source, legend="South", line_color=colors[4], **props)
    p5 = p.line('x', 'Southeast', source=source, legend="SouthEast", line_color=colors[5], **props)
    p6 = p.line('x', 'Southwest', source=source, legend="SouthWest", line_color=colors[6], **props)

    checkbox = CheckboxGroup(
        labels=LABELS,
        inline=True,
        active=[0, 1, 2, 3, 4, 5, 6],
        width=800)

    code = """
        //console.log(cb_obj.active);
        p0.visible = false;
        p1.visible = false;
        p2.visible = false;
        p3.visible = false;
        p4.visible = false;
        p5.visible = false;
        p6.visible = false;

        for (i in checkbox.active) {
            //console.log(cb_obj.active[i]);
            if (checkbox.active[i] == 0) {
                p0.visible = true;
            } else if (checkbox.active[i] == 1) {
                p1.visible = true;
            } else if (checkbox.active[i] == 2) {
                p2.visible = true;
            } else if (checkbox.active[i] == 3) {
                p3.visible = true;
            } else if (checkbox.active[i] == 4) {
                p4.visible = true;
            } else if (checkbox.active[i] == 5) {
                p5.visible = true;
            } else if (checkbox.active[i] == 6) {
                p6.visible = true;
            }
        }
    """

    checkbox.callback = CustomJS(args=dict(p0=p0, p1=p1, p2=p2, p3=p3, p4=p4, p5=p5, p6=p6, checkbox=checkbox),
                                 code=code)

    boundaries = open(DATA + 'boundaries.json').read()
    states = json.loads(boundaries)
    region_state = pd.read_csv(DATA + 'region-state.csv', header=0)
    region_state = region_state.set_index('state')

    state_xs = [states[code]["lons"] for code in states]
    state_ys = [states[code]["lats"] for code in states]
    name = states.keys()

    colors_state = []

    for i in name:
        if i != 'AK' and i != 'HI':
            reg = region_state.loc[i]['region']
            if reg == "Central":
                colors_state.append(colors[0])
            elif reg == "Mid-Atlantic":
                colors_state.append(colors[1])
            elif reg == "Northeast":
                colors_state.append(colors[2])
            elif reg == "Northwest":
                colors_state.append(colors[3])
            elif reg == "South":
                colors_state.append(colors[4])
            elif reg == "Southeast":
                colors_state.append(colors[5])
            elif reg == "Southwest":
                colors_state.append(colors[6])

    source = ColumnDataSource(data=dict(
        x=state_xs,
        y=state_ys,
        name=name,
        colors=colors_state,
    ))

    q = figure(title="",
               toolbar_location=None,
               plot_width=270,
               plot_height=150
               )
    q.xaxis.visible = False
    q.yaxis.visible = False
    q.xgrid.grid_line_color = None
    q.ygrid.grid_line_color = None
    q.min_border_left = False
    q.min_border_right = False
    q.min_border_top = False
    q.min_border_bottom = False

    q.patches('x', 'y', source=source,
              fill_color='colors',
              fill_alpha=0.9, line_color="white", line_width=0.1)

    layout = VBox(q, checkbox, p)

    # show(layout)

    script, div = components(layout)

    return script, div


def getRegionFromBoundaries():
    boundaries = open(DATA + 'boundaries.json').read()
    states = json.loads(boundaries)

    region = [states[code]["region"] for code in states]
    state = states.keys()

    df = pd.DataFrame({'region': region, "state": state})
    df.to_csv(DATA_LOCAL + "region-state.csv", index=False)


def countRateOverTime():
    # data = pd.read_csv(DATA_LOCAL + 'accepted_refused_ds.csv')
    # region = pd.read_csv(DATA_LOCAL + 'region-state.csv')
    # region = region.set_index('state')
    # print region.loc['DC'][0]
    #
    # reg = [region.loc[st][0] if (st != 'AK' and st != 'HI') else 'out' for st in data['state']]
    # print len(reg)
    # print len(data['loan'])
    #
    # small_ds = pd.DataFrame({'region': reg,
    #                          'loan': data['loan'],
    #                          'date': data['date']})
    #
    # small_ds.to_csv('region_loan.csv', index=False)



    # ds = pd.read_csv(DATA_LOCAL + 'region_loan.csv')
    # aggregation = ds[['region', 'loan', 'date']].groupby(['region', 'date'])
    # print 'There are ' + str(len(aggregation)) + ' different region-state'
    # new_ds = aggregation.agg(['count', 'sum'])
    # new_ds.reset_index(inplace=True)
    # print new_ds
    # print new_ds.columns
    # # new_ds.columns.droplevel()
    # # print new_ds
    # # print new_ds.columns
    # new_ds.columns = ['region', 'date', 'requests', 'acc_loan']
    # print new_ds
    # new_ds['perc_acc_loan'] = new_ds.acc_loan / new_ds.requests*100.0
    # del new_ds['requests']
    # del new_ds['acc_loan']
    # new_ds.to_csv(DATA_LOCAL + 'perc_acc_loan_per_region_date.csv', index=False)

    # ds = pd.read_csv(DATA_LOCAL + 'perc_acc_loan_per_region_date.csv')
    # dist_region = set(ds.region.values)
    # dist_date = set(ds.date.values)
    # ordered_date = sorted(dist_date, key=lambda x: datetime.strptime(x, '%b-%Y'))
    # col = ['date'] + list(dist_region)
    # my_ds = pd.DataFrame(columns=col)
    #
    # for od in ordered_date:
    #     new_row = [od]
    #     for dr in dist_region:
    #         perc = round(ds[(ds.date == od) & (ds.region == dr)]['perc_acc_loan'].values[0], 3)
    #         new_row.append(perc)
    #
    #     my_ds = pd.concat([my_ds, pd.DataFrame([new_row], columns=col)])
    #
    # my_ds.to_csv(DATA_LOCAL + 'perc_acc_loan_per_region_date_compact.csv', index=False)

    pass


# def createSmallDataset():
#
#     from sklearn.model_selection import train_test_split
#
#     data = pd.read_csv(DATA_LOCAL + 'accepted.csv', header=0)
#     print data.columns.values
#
#     small = data[['loan_amnt', 'annual_inc', 'dti', 'int_rate', 'addr_state']]
#     small.columns = ['amnt', 'income', 'dti', 'rate', 'state']
#     small['rate'] = small['rate'].map(lambda x: str(x).replace('%', '').strip())
#     small.to_csv(DATA_LOCAL + 'accepted_less_col.csv', index=False)
#
#     X_train, X_test = train_test_split(small, test_size=1.0 / 26, random_state=101)
#
#     dist_state = set(X_train.state.values)
#
#     for s in dist_state:
#         if s == 'IA' or s == 'ID':
#             my_set = X_train[X_train.state == s]
#             my_set.to_csv(DATA_LOCAL + 'accepted_less_col_small_' + s + '.csv', index=False)

def test():
    import numpy as np

    from bokeh.io import output_file, show
    from bokeh.layouts import row
    from bokeh.palettes import Viridis3
    from bokeh.plotting import figure
    from bokeh.models import CheckboxGroup, CustomJS

    output_file("line_on_off.html", title="line_on_off.py example")

    p = figure()
    props = dict(line_width=4, line_alpha=0.7)
    x = np.linspace(0, 4 * np.pi, 100)
    l0 = p.line(x, np.sin(x), color=Viridis3[0], legend="Line 0", **props)
    l1 = p.line(x, 4 * np.cos(x), color=Viridis3[1], legend="Line 1", **props)
    l2 = p.line(x, np.tan(x), color=Viridis3[2], legend="Line 2", **props)

    checkbox = CheckboxGroup(labels=["Line 0", "Line 1", "Line 2"],
                             active=[0, 1, 2], width=100)
    checkbox.callback = CustomJS(args=dict(l0=l0, l1=l1, l2=l2, checkbox=checkbox),
                                 lang="coffeescript", code="""
    l0.visible = 0 in checkbox.active;
    l1.visible = 1 in checkbox.active;
    l2.visible = 2 in checkbox.active;
    """)

    layout = row(checkbox, p)
    show(layout)
