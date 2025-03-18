from dash import Dash, dcc, html, dash_table, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import dash
from dash import callback_context

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.JOURNAL, dbc.icons.FONT_AWESOME],
)

#  make dataframe from  spreadsheet:
history_index = -1
df = pd.read_csv("assets/historic.csv")
# print(df.head())
history_df = pd.DataFrame(columns=["start_year", "planning_time", "cash_allocation", "stock_allocation",
                                   "bond_allocation", "start_balance", "end_balance", "cagr"])

print("Columns in dff:", history_df.columns)


def record_history(stocks, cash, start_bal, planning_time, start_yr, dff):
    global history_df, history_index

    # Calculate the end balance and CAGR
    end_balance = dff["Total"].iloc[-1] if not dff.empty and "Total" in dff.columns else 0

    cagr_value = cagr(dff["Total"]) if not dff.empty and "Total" in dff.columns else "0.0%"

    # Calculate bond allocation
    bond_allocation = 100 - cash - stocks

    # Create a new entry for the history log
    new_entry = {
        "start_year": start_yr,
        "planning_time": planning_time,
        "cash_allocation": cash,
        "stock_allocation": stocks,
        "bond_allocation": bond_allocation,
        "start_balance": start_bal,
        "end_balance": end_balance,
        "cagr": cagr_value,
    }

    # Convert the new entry to a DataFrame
    new_entry_df = pd.DataFrame([new_entry])

    # Use pd.concat to add the new entry to the existing history_df
    history_df = pd.concat([history_df, new_entry_df], ignore_index=True)

    # Update the history index to point to the most recent setting
    history_index = len(history_df) - 1


MAX_YR = df.Year.max()
MIN_YR = df.Year.min()
START_YR = 2007

# since data is as of year-end, need to add start year
# The surrounding bracket creates a list so that the row would be perfectly created
row = pd.DataFrame({"Year": [MIN_YR - 1]})
df = (
    pd.concat([df, row], ignore_index=True)
    .sort_values("Year", ignore_index=True)
    .fillna(0)
)

COLORS = {
    "cash": "#3cb521",
    "bonds": "#fd7e14",
    "stocks": "#446e9b",
    "inflation": "#cd0200",
    "background": "whitesmoke",
}

"""
==========================================================================
Markdown Text
"""

datasource_text = dcc.Markdown(
    """
    [Data source:](http://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/histretSP.html)
    Historical Returns on Stocks, Bonds and Bills from NYU Stern School of
    Business
    """
)

asset_allocation_text = dcc.Markdown(
    """
> **Asset allocation** is one of the main factors that drive portfolio risk and returns.   Play with the app and see for yourself!

> Change the allocation to cash, bonds and stocks on the sliders and see how your portfolio performs over time in the graph.
  Try entering different time periods and dollar amounts too.
"""
)

learn_text = dcc.Markdown(
    """
    Past performance certainly does not determine future results, but you can still
    learn a lot by reviewing how various asset classes have performed over time.

    Use the sliders to change the asset allocation (how much you invest in cash vs
    bonds vs stock) and see how this affects your returns.

    Note that the results shown in "My Portfolio" assumes rebalancing was done at
    the beginning of every year.  Also, this information is based on the S&P 500 index
    as a proxy for "stocks", the 10 year US Treasury Bond for "bonds" and the 3 month
    US Treasury Bill for "cash."  Your results of course,  would be different based
    on your actual holdings.

    This is intended to help you determine your investment philosophy and understand
    what sort of risks and returns you might see for each asset category.

    The  data is from [Aswath Damodaran](http://people.stern.nyu.edu/adamodar/New_Home_Page/home.htm)
    who teaches  corporate finance and valuation at the Stern School of Business
    at New York University.

    Check out his excellent on-line course in
    [Investment Philosophies.](http://people.stern.nyu.edu/adamodar/New_Home_Page/webcastinvphil.htm)
    """
)

cagr_text = dcc.Markdown(
    """
    (CAGR) is the compound annual growth rate.  It measures the rate of return for an investment over a period of time, 
    such as 5 or 10 years. The CAGR is also called a "smoothed" rate of return because it measures the growth of
     an investment as if it had grown at a steady rate on an annually compounded basis.
    """
)

footer = html.Div(
    dcc.Markdown(
        """
         This information is intended solely as general information for educational
        and entertainment purposes only and is not a substitute for professional advice and
        services from qualified financial services providers familiar with your financial
        situation.    
        """
    ),
    className="p-2 mt-5 bg-primary text-white small",
)

"""
==========================================================================
Tables
"""

total_returns_table = dash_table.DataTable(
    id="total_returns",
    columns=[{"id": "Year", "name": "Year", "type": "text"}]
            + [
                {"id": col, "name": col, "type": "numeric", "format": {"specifier": "$,.0f"}}
                for col in ["Cash", "Bonds", "Stocks", "Total"]
            ],
    page_size=15,
    style_table={"overflowX": "scroll"},
)

annual_returns_pct_table = dash_table.DataTable(
    id="annual_returns_pct",
    columns=(
            [{"id": "Year", "name": "Year", "type": "text"}]
            + [
                {"id": col, "name": col, "type": "numeric", "format": {"specifier": ".1%"}}
                for col in df.columns[1:]
            ]
    ),
    data=df.to_dict("records"),
    sort_action="native",
    page_size=15,
    style_table={"overflowX": "scroll"},
)


def make_summary_table(dff):
    """Make html table to show cagr and  best and worst periods"""

    table_class = "h5 text-body text-nowrap"
    cash = html.Span(
        [html.I(className="fa fa-money-bill-alt"), " Cash"], className=table_class
    )
    bonds = html.Span(
        [html.I(className="fa fa-handshake"), " Bonds"], className=table_class
    )
    stocks = html.Span(
        [html.I(className="fa fa-industry"), " Stocks"], className=table_class
    )
    inflation = html.Span(
        [html.I(className="fa fa-ambulance"), " Inflation"], className=table_class
    )

    start_yr = dff["Year"].iat[0]
    end_yr = dff["Year"].iat[-1]

    df_table = pd.DataFrame(
        {
            "": [cash, bonds, stocks, inflation],
            f"Rate of Return (CAGR) from {start_yr} to {end_yr}": [
                cagr(dff["all_cash"]),
                cagr(dff["all_bonds"]),
                cagr(dff["all_stocks"]),
                cagr(dff["inflation_only"]),
            ],
            f"Worst 1 Year Return": [
                worst(dff, "3-mon T.Bill"),
                worst(dff, "10yr T.Bond"),
                worst(dff, "S&P 500"),
                "",
            ],
        }
    )
    return dbc.Table.from_dataframe(df_table, bordered=True, hover=True)


"""
==========================================================================
Figures
"""


def make_bar(slider_input, title):
    fig = go.Figure(
        data=[
            go.Bar(
                x=["Cash", "Bonds", "Stocks"],
                y=slider_input,
                textposition="inside",
                marker=dict(color=[COLORS["cash"], COLORS["bonds"], COLORS["stocks"]]),

            )
        ]
    )
    fig.update_layout(
        title_text=title,
        title_x=0.5,
        margin=dict(b=25, t=75, l=35, r=25),
        height=325,
        paper_bgcolor=COLORS["background"],
        xaxis_title='Asset Type',
        yaxis_title='Percentage',
        showlegend=False,
    )
    return fig


def make_line_chart(dff):
    start = dff.loc[1, "Year"]
    yrs = dff["Year"].size - 1
    dtick = 1 if yrs < 16 else 2 if yrs in range(16, 30) else 5

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dff["Year"],
            y=dff["all_cash"],
            name="All Cash",
            marker_color=COLORS["cash"],
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dff["Year"],
            y=dff["all_bonds"],
            name="All Bonds (10yr T.Bonds)",
            marker_color=COLORS["bonds"],
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dff["Year"],
            y=dff["all_stocks"],
            name="All Stocks (S&P500)",
            marker_color=COLORS["stocks"],
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dff["Year"],
            y=dff["Total"],
            name="My Portfolio",
            marker_color="black",
            line=dict(width=6, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dff["Year"],
            y=dff["inflation_only"],
            name="Inflation",
            visible=True,
            marker_color=COLORS["inflation"],
        )
    )
    fig.update_layout(
        title=f"Returns for {yrs} years starting {start}",
        template="none",
        showlegend=True,
        legend=dict(x=0.01, y=0.99),
        height=400,
        margin=dict(l=40, r=10, t=60, b=55),
        yaxis=dict(tickprefix="$", fixedrange=True),
        xaxis=dict(title="Year Ended", fixedrange=True, dtick=dtick),
    )
    return fig


"""
==========================================================================
Make Tabs
"""

# =======Play tab components
previous_button = dbc.Button(
    'Previous Setting', id='previous-setting-btn', n_clicks=0, disabled=True, color='primary'
)

asset_allocation_card = dbc.Card(asset_allocation_text, className="mt-2")

slider_card = dbc.Card(
    [
        html.H4("First set cash allocation %:", className="card-title"),
        dcc.Slider(
            id="cash",
            marks={i: f"{i}%" for i in range(0, 101, 10)},
            min=0,
            max=100,
            step=5,
            value=10,
            included=False,
        ),
        html.H4(
            "Then set stock allocation % ",
            className="card-title mt-3",
        ),
        dcc.Slider(
            id="stock_bond",
            marks={i: f"{i}%" for i in range(0, 91, 10)},
            min=0,
            max=90,
            step=5,
            value=50,
            included=False,
        ),
        html.H4("After set bond stock allocation %", className="card-title mt-3"),
        dcc.Slider(
            id='bond',
            marks={i: f'{i}%' for i in range(0, 91, 10)},
            min=0,
            max=90,
            step=5,
            value=50,
            included=False,
        ),
        previous_button
    ],
    body=True,
    className="mt-4",
)

time_period_data = [
    {
        "label": f"2007-2008: Great Financial Crisis to {MAX_YR}",
        "start_yr": 2007,
        "planning_time": MAX_YR - START_YR + 1,
    },
    {
        "label": "1999-2010: The decade including 2000 Dotcom Bubble peak",
        "start_yr": 1999,
        "planning_time": 10,
    },
    {
        "label": "1969-1979:  The 1970s Energy Crisis",
        "start_yr": 1970,
        "planning_time": 10,
    },
    {
        "label": "1929-1948:  The 20 years following the start of the Great Depression",
        "start_yr": 1929,
        "planning_time": 20,
    },
    {
        "label": f"{MIN_YR}-{MAX_YR}",
        "start_yr": "1928",
        "planning_time": MAX_YR - MIN_YR + 1,
    },
]

time_period_card = dbc.Card(
    [
        html.H4(
            "Or select a time period:",
            className="card-title",
        ),
        dbc.RadioItems(
            id="time_period",
            options=[
                {"label": period["label"], "value": i}
                for i, period in enumerate(time_period_data)
            ],
            value=0,
            labelClassName="mb-2",
        ),
    ],
    body=True,
    className="mt-4",
)

# ======= InputGroup components

start_amount = dbc.InputGroup(
    [
        dbc.InputGroupText("Start Amount $"),
        dbc.Input(
            id="starting_amount",
            placeholder="Min $10",
            type="number",
            min=10,
            value=10000,
        ),
    ],
    className="mb-3",
)
start_year = dbc.InputGroup(
    [
        dbc.InputGroupText("Start Year"),
        dbc.Input(
            id="start_yr",
            placeholder=f"min {MIN_YR}   max {MAX_YR}",
            type="number",
            min=MIN_YR,
            max=MAX_YR,
            value=START_YR,
        ),
    ],
    className="mb-3",
)
number_of_years = dbc.InputGroup(
    [
        dbc.InputGroupText("Number of Years:"),
        dbc.Input(
            id="planning_time",
            placeholder="# yrs",
            type="number",
            min=1,
            value=MAX_YR - START_YR + 1,
        ),
    ],
    className="mb-3",
)
end_amount = dbc.InputGroup(
    [
        dbc.InputGroupText("Ending Amount"),
        dbc.Input(id="ending_amount", disabled=True, className="text-black"),
    ],
    className="mb-3",
)
rate_of_return = dbc.InputGroup(
    [
        dbc.InputGroupText(
            "Rate of Return(CAGR)",
            id="tooltip_target",
            className="text-decoration-underline",
        ),
        dbc.Input(id="cagr", disabled=True, className="text-black"),
        dbc.Tooltip(cagr_text, target="tooltip_target"),
    ],
    className="mb-3",
)

input_groups = html.Div(
    [start_amount, start_year, number_of_years, end_amount, rate_of_return],
    className="mt-4 p-4",
)

# =====  Results Tab components

results_card = dbc.Card(
    [
        dbc.CardHeader("My Portfolio Returns - Rebalanced Annually"),
        html.Div(total_returns_table),
    ],
    className="mt-4",
)

# ==== History Tab Component

history_card = dbc.Card(
    [
        dbc.CardHeader("This is your history with this app"),
        dash_table.DataTable(
            id='history-table',
            columns=[
                {"name": "Start Year", "id": "start_year", "type": "numeric"},
                {"name": "Years", "id": "planning_time", "type": "numeric"},
                {"name": "Cash %", "id": "cash_allocation", "type": "numeric", "format": {"specifier": ".1f"}},
                {"name": "Stock %", "id": "stock_allocation", "type": "numeric", "format": {"specifier": ".1f"}},
                {"name": "Bond %", "id": "bond_allocation", "type": "numeric", "format": {"specifier": ".1f"}},
                {"name": "Start Balance", "id": "start_balance", "type": "numeric", "format": {"specifier": "$,.0f"}},
                {"name": "End Balance", "id": "end_balance", "type": "numeric", "format": {"specifier": "$,.0f"}},
                {"name": "CAGR", "id": "cagr", "type": "text"}
                ,
            ],
            data=[],
            sort_action='native',
            style_table={"height": "400px", "overflowY": "auto"},
            style_cell={
                'textAlign': 'center',
                'padding': '8px'
            },

        ),
        dbc.Button(
            "Apply Selected Settings",
            id="apply-history-btn",
            color="primary",
            className="mt-3"
        ),

    ],
    className="mt-4",
)

data_source_card = dbc.Card(
    [
        dbc.CardHeader("Source Data: Annual Total Returns"),
        html.Div(annual_returns_pct_table),
    ],
    className="mt-4",
)

# ========= Learn Tab  Components
learn_card = dbc.Card(
    [
        dbc.CardHeader("An Introduction to Asset Allocation"),
        dbc.CardBody(learn_text),
    ],
    className="mt-4",
)

# ========= Build tabs
tabs = dbc.Tabs(
    [
        dbc.Tab(learn_card, tab_id="tab1", label="Learn"),
        dbc.Tab(
            [asset_allocation_text, slider_card, input_groups, time_period_card],
            tab_id="tab-2",
            label="Play",
            className="pb-4",
        ),
        dbc.Tab([results_card, data_source_card], tab_id="tab-3", label="Results"),
        dbc.Tab(history_card, tab_id='tab-4', label='History')
    ],
    id="tabs",
    active_tab="tab-2",
    className="mt-2",
)

"""
==========================================================================
Helper functions to calculate investment results, cagr and worst periods
"""
history_data = []


def history_save_data(start_amount, start_year, number_of_years, cash, stocks, bond):
    # These variables have a global scope, meaning they can be accessed and
    # modified by any function within the same module.
    global history_data
    history_data.append(
        {
            'start_amount': start_amount,
            'start_year': start_year,
            'number_of_years': number_of_years,
            'cash': cash,
            'stocks': stocks,
            'bond': bond,
        }
    )


def backtest(stocks, cash, start_bal, nper, start_yr):
    """calculates the investment returns for user selected asset allocation,
    rebalanced annually and returns a dataframe
    """

    end_yr = start_yr + nper - 1
    cash_allocation = cash / 100
    stocks_allocation = stocks / 100
    bonds_allocation = (100 - stocks - cash) / 100

    # Select time period - since data is for year_end, include year prior
    # for start ie year[0]
    dff = df[(df.Year >= start_yr - 1) & (df.Year <= end_yr)].set_index(
        "Year", drop=False
    )
    dff["Year"] = dff["Year"].astype(int)

    # add columns for My Portfolio returns
    dff["Cash"] = cash_allocation * start_bal
    dff["Bonds"] = bonds_allocation * start_bal
    dff["Stocks"] = stocks_allocation * start_bal
    dff["Total"] = start_bal
    dff["Rebalance"] = True

    # calculate My Portfolio returns
    for yr in dff.Year + 1:
        if yr <= end_yr:
            # Rebalance at the beginning of the period by reallocating
            # last period's total ending balance
            if dff.loc[yr, "Rebalance"]:
                dff.loc[yr, "Cash"] = dff.loc[yr - 1, "Total"] * cash_allocation
                dff.loc[yr, "Stocks"] = dff.loc[yr - 1, "Total"] * stocks_allocation
                dff.loc[yr, "Bonds"] = dff.loc[yr - 1, "Total"] * bonds_allocation

            # calculate this period's  returns
            dff.loc[yr, "Cash"] = dff.loc[yr, "Cash"] * (
                    1 + dff.loc[yr, "3-mon T.Bill"]
            )
            dff.loc[yr, "Stocks"] = dff.loc[yr, "Stocks"] * (1 + dff.loc[yr, "S&P 500"])
            dff.loc[yr, "Bonds"] = dff.loc[yr, "Bonds"] * (
                    1 + dff.loc[yr, "10yr T.Bond"]
            )
            dff.loc[yr, "Total"] = dff.loc[yr, ["Cash", "Bonds", "Stocks"]].sum()
            dff.loc[yr, 'Total'] = float(dff.loc[yr, 'Total'])

    dff = dff.reset_index(drop=True)
    columns = ["Cash", "Stocks", "Bonds", "Total"]
    dff[columns] = dff[columns].round(0).astype('int64')

    # create columns for when portfolio is all cash, all bonds or  all stocks,
    #   include inflation too
    #
    # create new df that starts in yr 1 rather than yr 0
    dff1 = (dff[(dff.Year >= start_yr) & (dff.Year <= end_yr)]).copy()
    #
    # calculate the returns in new df:
    columns = ["all_cash", "all_bonds", "all_stocks", "inflation_only"]
    annual_returns = ["3-mon T.Bill", "10yr T.Bond", "S&P 500", "Inflation"]
    for col, return_pct in zip(columns, annual_returns):
        dff1[col] = round(start_bal * (1 + (1 + dff1[return_pct]).cumprod() - 1), 0)
    #
    # select columns in the new df to merge with original
    dff1 = dff1[["Year"] + columns]
    dff = dff.merge(dff1, how="left")
    # fill in the starting balance for year[0]
    dff.loc[0, columns] = start_bal
    return dff


def cagr(dff):
    """calculate Compound Annual Growth Rate for a series and returns a formated string"""

    start_bal = dff.iat[0]
    end_bal = dff.iat[-1]
    planning_time = len(dff) - 1
    cagr_result = ((end_bal / start_bal) ** (1 / planning_time)) - 1
    return f"{cagr_result:.1%}"


def worst(dff, asset):
    """calculate worst returns for asset in selected period returns formated string"""

    worst_yr_loss = min(dff[asset])
    worst_yr = dff.loc[dff[asset] == worst_yr_loss, "Year"].iloc[0]
    return f"{worst_yr_loss:.1%} in {worst_yr}"


"""
===========================================================================
Main Layout
"""

app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                html.H2(
                    "Nice Teta Hirwa -- CS150 -- Professor Mike Ryu",
                    className="bg-primary p-2 mb-2 text-center",
                )
            )
        ),
        dbc.Row(
            dbc.Col(
                html.H2(
                    "Asset Allocation Visualizer",
                    className="text-center bg-primary text-white p-2",
                ),
            )
        ),
        dbc.Row(
            [
                dbc.Col(tabs, width=12, lg=5, className="mt-4 border"),
                dbc.Col(
                    [
                        dcc.Graph(id="allocation_bar_chart", className="mb-2"),
                        dcc.Graph(id="returns_chart", className="pb-4"),
                        html.Hr(),
                        html.Div(id="summary_table"),
                        html.H6(datasource_text, className="my-2"),
                    ],
                    width=12,
                    lg=7,
                    className="pt-4",
                ),
            ],
            className="ms-1",
        ),
        dbc.Row(dbc.Col(footer)),
    ],
    fluid=True,
)

"""
==========================================================================
Callbacks
"""


@app.callback(
    Output('history-table', 'data'),
    [Input('tabs', 'active_tab')]
)
def update_history_tab(active_tab):
    global history_df

    # Get current context to identify trigger
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

    # Update the history table only when the 'History' tab is active
    if active_tab == 'tab-4':
        return history_df.to_dict('records')

    return []


# Add callback for applying selected history settings
@app.callback(
    [Output("start_yr", "value", allow_duplicate=True),
     Output("planning_time", "value", allow_duplicate=True),
     Output("cash", "value", allow_duplicate=True),
     Output("stock_bond", "value", allow_duplicate=True),
     Output("bond", "value", allow_duplicate=True),
     Output("starting_amount", "value", allow_duplicate=True),
     Output("tabs", "active_tab")],
    [Input("apply-history-btn", "n_clicks"),
     Input("history-table", "selected_rows")],
    [State("history-table", "data")],
    prevent_initial_call=True
)
def apply_history_settings(n_clicks, selected_rows, table_data):
    if n_clicks is None or not selected_rows or not table_data:
        raise dash.exceptions.PreventUpdate

    # Get the selected row data
    selected_data = table_data[selected_rows[0]]

    # Return the settings to apply and switch to the Play tab
    return (
        selected_data["start_year"],
        selected_data["planning_time"],
        selected_data["cash_allocation"],
        selected_data["stock_allocation"],
        selected_data["bond_allocation"],
        selected_data["start_balance"],
        "tab-2"  # Switch to Play tab
    )


@app.callback(
    [
        Output("allocation_bar_chart", "figure"),
        Output("stock_bond", "max"),
        Output("stock_bond", "marks"),
        Output("stock_bond", "value"),
        Output("planning_time", "value"),
        Output("start_yr", "value"),
        Output("total_returns", "data"),
        Output("returns_chart", "figure"),
        Output("summary_table", "children"),
        Output("ending_amount", "value"),
        Output("cagr", "value"),
        Output("previous-setting-btn", "disabled"),
        Output("cash", "value"),
        Output("bond", "value"),
    ],
    [
        Input("stock_bond", "value"),
        Input("cash", "value"),
        Input("bond", "value"),
        Input("starting_amount", "value"),
        Input("planning_time", "value"),
        Input("start_yr", "value"),
        Input('previous-setting-btn', 'n_clicks'),
        Input('time_period', 'value')
    ],
)
def update_dashboard(
        stocks, cash, bond, start_bal, planning_time, start_yr, prev_n_clicks, time_period
):
    global history_df, history_index

    # Get current context to identify trigger
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

    # Handle previous settings button click
    if trigger_id == 'previous-setting-btn' and prev_n_clicks > 0:
        # There's at least one history entry
        if len(history_df) >= 1:
            # If we're at the current settings (end of history), start by going back one
            if history_index >= len(history_df) or history_index == -1:
                history_index = len(history_df) - 1

            # Get the settings at the current history index
            prev_setting = history_df.iloc[history_index]
            stocks = prev_setting['stock_allocation']
            cash = prev_setting['cash_allocation']
            bond = prev_setting['bond_allocation']
            start_bal = prev_setting['start_balance']
            start_yr = prev_setting['start_year']
            planning_time = prev_setting['planning_time']

            # Move back one more position for the next click
            history_index -= 1

            # Button should be disabled if we reach the oldest entry
            disabled = history_index < 0
        else:
            disabled = True
    elif trigger_id == 'time_period':
        # Handle time period selection
        period_data = time_period_data[time_period]
        start_yr = period_data["start_yr"] if isinstance(period_data["start_yr"], int) else int(period_data["start_yr"])
        planning_time = period_data["planning_time"]
        # Don't update the allocation as it's not part of the time period selection
        disabled = len(history_df) == 0


    else:

        # Only record history for interactions that change values

        if trigger_id is not None and trigger_id not in ['previous-setting-btn', 'time_period']:
            # Create investment returns dataframe for the current settings

            dff = backtest(stocks, cash, start_bal, planning_time, start_yr)

            # Record the current settings in history

            record_history(stocks, cash, start_bal, planning_time, start_yr, dff)

            # Reset history index to point to current settings

            history_index = len(history_df) - 1

        # Button is enabled as soon as we have at least one entry

        disabled = len(history_df) == 0

        # Set defaults for invalid inputs
    start_bal = 10 if start_bal is None else start_bal
    planning_time = 1 if planning_time is None else planning_time
    start_yr = MIN_YR if start_yr is None else int(start_yr)

    # Calculate valid planning time start year
    max_time = MAX_YR + 1 - start_yr
    planning_time = min(max_time, planning_time)
    if start_yr + planning_time > MAX_YR:
        start_yr = MAX_YR - planning_time + 1

    # Create investment returns dataframe
    dff = backtest(stocks, cash, start_bal, planning_time, start_yr)

    # Create data for DataTable
    data = dff.to_dict("records")

    # Create the line chart
    fig = make_line_chart(dff)

    summary_table = make_summary_table(dff)

    # Format ending balance
    ending_amount = f"${dff['Total'].iloc[-1]:0,.0f}"

    # Calculate CAGR
    ending_cagr = cagr(dff["Total"])

    # Update the bar chart and other sliders
    bonds = 100 - stocks - cash
    slider_input = [cash, bonds, stocks]

    if stocks >= 70:
        investment_style = "Aggressive"
    elif stocks <= 30:
        investment_style = "Conservative"
    else:
        investment_style = "Moderate"

        # Bar chart figure
    bar_chart = make_bar(slider_input, investment_style + " Asset Allocation")

    # Update the stock slider
    max_slider = 100 - int(cash)
    stocks = min(max_slider, stocks)

    if max_slider > 50:
        marks_slider = {i: f"{i}%" for i in range(0, max_slider + 1, 10)}
    elif max_slider <= 15:
        marks_slider = {i: f"{i}%" for i in range(0, max_slider + 1, 1)}
    else:
        marks_slider = {i: f"{i}%" for i in range(0, max_slider + 1, 5)}

    return (
        bar_chart,
        max_slider,
        marks_slider,
        stocks,
        planning_time,
        start_yr,
        data,
        fig,
        summary_table,
        ending_amount,
        ending_cagr,
        disabled,
        cash,
        bond
    )


if __name__ == "__main__":
    app.run(debug=True)
