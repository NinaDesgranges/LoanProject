from datetime import datetime
import pandas as pd
from settings import DATA
import numpy as np
from bokeh.io import show
from bokeh.palettes import Spectral6, viridis, Blues9
from bokeh.sampledata import us_states, us_counties, unemployment
from bokeh.plotting import figure, show, output_file

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
# import bokeh
# import bokeh.sampledata
from bokeh.sampledata.unemployment import data as unemployment
from bokeh.sampledata.us_counties import data as cc

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
    TODO: 1. clean dti = -1; 2. fill NA values

    :return:
    '''
    refused = pd.read_csv(DATA + 'refused.csv', sep=',', low_memory=False)
    print refused.columns.values

    accepted = pd.read_csv(DATA + 'accepted.csv', sep=',', low_memory=False)
    print accepted.columns.values

    small_acc = accepted[['purpose', 'loan_amnt', 'zip_code', 'addr_state', 'emp_length', 'dti', 'issue_d']]
    # small_acc['issue_d'] = small_acc['issue_d'].map(
    #     lambda x: datetime.strptime(x, "%b-%Y") if x is not None else None)
    small_acc['loan'] = [1] * small_acc.shape[0]

    small_acc.columns = ACC_REF_HEADER

    small_ref = refused[
        ['Loan Title', 'Amount Requested', 'Zip Code', 'State', 'Employment Length', 'Debt-To-Income Ratio',
         'Application Date']]
    small_ref['Debt-To-Income Ratio'] = small_ref['Debt-To-Income Ratio'].map(lambda x: x.replace('%', ''))
    small_ref['Application Date'] = small_ref['Application Date'].map(
        lambda x: datetime.strptime(x, "%Y-%m-%d").strftime('%b-%Y') if x is not None else None)
    small_ref = small_ref[small_ref['Application Date'] != 'May-2007']
    small_ref['loan'] = [0] * small_ref.shape[0]

    small_ref.columns = ACC_REF_HEADER

    new_dataset = pd.concat([small_ref, small_acc])
    new_dataset = new_dataset[new_dataset.amnt != 0]

    pd.DataFrame(new_dataset).to_csv(DATA + 'accepted_refused_ds.csv', index=False, header=ACC_REF_HEADER)


def getInfo():
    ds = pd.read_csv(DATA + 'accepted_refused_ds.csv', header=0)
    print sum(ds.loan)
    print 'Percentage of accepted loan ' + str(float(sum(ds.loan)) / ds.shape[0] * 100) + '%'
    print ds.amnt.describe()
    print ds.dti.describe()
    print ds.emp_len.describe()


def plotWithUSPercentage():
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

    cm = LinearColorMapper(palette=Blues9, low=min(new_ds.perc_acc_loan.values), high=max(new_ds.perc_acc_loan.values))
    # , low=min(new_ds.perc_acc_loan.values), high=max(new_ds.perc_acc_loan.values)

    # viridis[10]

    # bokeh.sampledata.download()

    mus_states = us_states.data.copy()

    del mus_states["HI"]
    del mus_states["AK"]

    state_xs = [mus_states[code]["lons"] for code in mus_states]
    state_ys = [mus_states[code]["lats"] for code in mus_states]

    source = ColumnDataSource(data=dict(
        x=state_xs,
        y=state_ys,
        name=new_ds.index.get_values(),
        rate=new_ds['perc_acc_loan'],
    ))

    output_file("choropleth.html", title="choropleth.py example")

    p = figure(title="Loan acceptance rate", toolbar_location="left",
               plot_width=1100, plot_height=700)

    p.patches('x', 'y', source=source,
              fill_color={'field': 'rate', 'transform': cm},
              fill_alpha=0.7, line_color="white", line_width=0.5)

    # p.grid.grid_line_color = None
    # p.axis.axis_line_color = None
    # p.axis.major_tick_line_color = None
    # p.axis.major_label_text_font_size = "5pt"
    # p.axis.major_label_standoff = 0
    # p.xaxis.major_label_orientation = 180 / 3

    # color_bar = ColorBar(color_mapper=cm, major_label_text_font_size="20pt",
    #                      ticker=BasicTicker(desired_num_ticks=len(Blues9)),
    #                      formatter=PrintfTickFormatter(format="%d%"),
    #                      label_standoff=60, border_line_color=None, location=(0, 0))

    color_bar = ColorBar(color_mapper=cm,
                         orientation='vertical',
                         location=(0, 0))
    p.add_layout(color_bar, 'right')


    show(p)