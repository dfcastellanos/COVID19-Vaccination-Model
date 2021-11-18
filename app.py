import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from datetime import datetime as dt


from model import run_model_sampling, sample_param_combinations


# pylint: disable=E0102

pio.templates.default = "plotly_white"


def get_country_data():

    df = pd.read_csv(
        "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/vaccinations/vaccinations.csv"
    )
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["location", "date"])

    columns_to_keep = [
        "location",
        "date",
        "people_fully_vaccinated_per_hundred",
        "daily_vaccinations_per_million",
    ]

    df = df.loc[:, columns_to_keep]

    avail_countries = df["location"].unique()
    country_data = pd.pivot_table(
        df,
        columns="location",
        values=[
            "people_fully_vaccinated_per_hundred",
            "daily_vaccinations_per_million",
        ],
        index="date",
    )

    return avail_countries, country_data


avail_countries, country_data = get_country_data()


tab_style = {"padding-top": 7}

tab_selected_style = {"padding-top": 7}

tabs_styles = {"height": "44px"}


def description_card():

    return html.Div(
        id="description-card",
        children=[
            html.Div(
                id="intro",
                # children="Here you can tune the population views on vaccines, the vaccines production rates, and the Monte Carlo sampling parameters.",
            ),
            html.Br(),
        ],
    )


def generate_population_controls():

    return html.Div(
        id="population-controls",
        children=[
            # html.P("Population views on vaccines"),
            html.Div(id="output-p-yes-value"),
            dcc.RangeSlider(
                id="slider-p-yes",
                min=0.0,
                max=100,
                value=[60, 70],
                allowCross=False,
                marks={"0": "0%", "100": "100%"},
                step=1,
                tooltip={"placement": "bottom", "always_visible": False},
            ),
            html.Div(id="output-p-hard-no-value"),
            dcc.RangeSlider(
                id="slider-p-hard-no",
                min=0.0,
                max=100,
                value=[5, 15],
                marks={"0": "0%", "100": "100%"},
                step=1,
                tooltip={"placement": "bottom", "always_visible": False},
            ),
            html.Div(id="pop-controls-error-msg", style={"color": "red"}),
            html.Div(id="output-p-soft-no-value"),
            html.Br(),
            html.Div(id="output-pressure-value"),
            dcc.RangeSlider(
                id="slider-pressure",
                min=0.0,
                max=10,
                value=[2, 5],
                marks={"0": "0%", "10": "10%"},
                step=0.5,
                tooltip={"placement": "bottom", "always_visible": False},
            ),
        ],
    )


def generate_country_and_date_controls():

    return html.Div(
        id="contry-date-controls",
        children=[
            html.Br(),
            html.P("Date range"),
            dcc.DatePickerRange(
                id="date-picker-select",
                start_date=dt(2020, 12, 30),
                end_date=dt.today(),
                display_format="MMM Do, YY",
                initial_visible_month=dt.today(),
            ),
            html.Br(),
            html.Br(),
            html.P("Add country"),
            dcc.Dropdown(
                id="country-select",
                options=[{"label": i, "value": i} for i in avail_countries],
                value=[],
                multi=True,
            ),
        ],
    )


def generate_plots_section():

    return html.Div(
        id="plot_header",
        children=[
            dcc.Graph(
                id="plot_grid",
                style={
                    # "width": "130vh",
                    # "height": "50vh",
                    # "display": "inline-block",
                    # "overflow": "hidden",
                    # "position": "absolute",
                    # "top": "50%",
                    # "left": "50%",
                    # "transform": "translate(-50%, -50%)"
                },
            ),
        ],
    )


def generate_model_explanation():

    s1 = """                        
        The goal of the model is to capture the **main characteristics** of the **evolution** of an ongoing **vaccination campaign** on a specific population. To this end, the population is described as a sample of discrete **random variables** whose values change according to some **evolution rules**. The model is sampled using the **Monte Carlo method**, i.e., generating random numbers, which are used to simulate the evolution of the random variables over time.
        """
    s2 = """                        
        The **population** is segmented into **three groups**, depending on their views on vaccines:
        
        -   **Pro-vaccines**: they take the vaccine as soon as they have the chance
        -   **Anti-vaccines**: they will never take a vaccine
        -   **Agnostic**: they will initially hesitate, but given enough social pressure, they will take it
        """
    s3 = """ 
        The **evolution of the vaccination** campaign is simulated by applying the following **rules** iteratively where **one iteration** corresponds to **one day**:
        
        1.  Every **non-vaccinated person** in the pro-vaccines group for whom a vaccine is available **becomes vaccinated**. A vaccine becomes available with a probability given by the number of vaccines in stock divided by the population size. That probability is multiplied by 2/7 to account for vaccinations occurring only two days a week, giving an effective per-day probability.
        2.  Every **agnostic person** might **become pro-vaccines** with a probability equal to the number of vaccinated people divided by the population size. This probability is multiplied by a factor, denoted as pressure, which allows for tuning the strength of this effect. This mechanism is a proxy for **social pressure**, i.e., the higher the fraction of vaccinated people is, the higher the influence on non-vaccinated ones to do the same.
        3.  The **stock of vaccines decreases** according to the number of people vaccinated during the day. Care is taken that, each day, no more vaccines than the available stock can be applied.
        """
    s4 = """                        
        The **stock of vaccines is increased** once a week. We distinguish two stages:
        
        1.  Initially, the number of vaccines added to the stock each week **grows exponentially**, representing a fast production growth to meet the demand.
        2.  When a specific **maximum delivery capacity** is reached, that amount does not grow anymore. Every subsequent week, that amount of vaccines are added to the existing stock.
        """
    return html.Div(
        id="text-explanation",
        children=[
            html.Br(),
            dcc.Markdown(s1),
            html.Br(),
            dcc.Markdown(s2),
            html.Br(),
            dcc.Markdown(s3),
            html.Br(),
            dcc.Markdown(s4),
        ],
    )


def generate_model_help():
    return


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

server = app.server

app.layout = html.Div(
    id="app-container",
    children=[
        # Banner
        html.Div(
            id="banner",
            className="banner",
            children=[
                html.H3(
                    "Monte Carlo model of COVID-19 vaccination",
                    style={"color": "#2c8cff"},
                ),
            ],
        ),
        # Left column
        html.Div(
            id="left-column",
            className="three columns",
            children=[
                description_card(),
                dcc.Tabs(
                    children=[
                        dcc.Tab(
                            label="Population",
                            children=[html.Br(), generate_population_controls()],
                            style=tab_style,
                            selected_style=tab_selected_style,
                        ),
                        dcc.Tab(
                            label="Vaccines",
                            children=[],
                            style=tab_style,
                            selected_style=tab_selected_style,
                        ),
                        dcc.Tab(
                            label="Sampling",
                            children=[],
                            style=tab_style,
                            selected_style=tab_selected_style,
                        ),
                    ],
                    style=tabs_styles,
                ),
                generate_country_and_date_controls(),
            ],
        ),
        # Right column
        html.Div(
            id="right-column",
            className="eight columns",
            children=[
                dcc.Tabs(
                    children=[
                        dcc.Tab(
                            label="Results",
                            children=[
                                generate_plots_section(),
                                dcc.Loading(
                                    id="ls-loading-2",
                                    children=[
                                        html.Div([html.Div(id="ls-loading-output-2")])
                                    ],
                                    type="circle",
                                ),
                            ],
                            style=tab_style,
                            selected_style=tab_selected_style,
                        ),
                        dcc.Tab(
                            label="Model explanation",
                            children=[generate_model_explanation()],
                            style=tab_style,
                            selected_style=tab_selected_style,
                        ),
                        dcc.Tab(
                            label="Help",
                            children=[generate_model_help()],
                            style=tab_style,
                            selected_style=tab_selected_style,
                        ),
                    ],
                    style=tabs_styles,
                ),
            ],
        ),
    ],
)


def add_line(fig, x, y, color, name=None, row=1, col=1, fill="none", width=2):

    # plot line

    data = dict(
        x=x,
        y=y,
        mode="lines",
        fill=fill,
        line_shape="spline",
        showlegend=False,
        line=dict(color=color, width=width),
    )

    if name is not None:
        data["legendgroup"] = name
        data["name"] = name

    fig.add_trace(
        go.Scatter(data),
        row=row,
        col=col,
    )

    # write annotation
    if name is not None:

        fig.add_annotation(
            xref="paper",
            x=x[-1],
            y=y[-1],
            xanchor="left",
            yanchor="middle",
            text=name,
            font=dict(family="Arial", size=14, color=color),
            showarrow=False,
            row=row,
            col=col,
        )

    return fig


@app.callback(
    Output("plot_grid", "figure"),
    Output("ls-loading-output-2", "children"),
    # population message: error and agnostic percentage
    Output("pop-controls-error-msg", "children"),
    Output("pop-controls-error-msg", "style"),
    # population parameters
    Input("slider-p-yes", "value"),
    Input("slider-p-hard-no", "value"),
    Input("slider-pressure", "value"),
    # countries and dates
    Input("date-picker-select", "start_date"),
    Input("date-picker-select", "end_date"),
    Input("country-select", "value"),
)
def update_figures(
    p_yes_bounds,
    p_hard_no_bounds,
    pressure_bounds,
    start_date,
    end_date,
    selected_countries,
):

    #
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "People vaccinated per hundred",
            "Daily vaccinations per million",
        ),
    )

    # ---- sample the model with the selected parameters ----

    start_date = dt.strptime(start_date.split("T")[0], "%Y-%m-%d")
    end_date = dt.strptime(end_date.split("T")[0], "%Y-%m-%d")

    n_rep = 100
    N = 2000

    # sliders use values 0-100

    params_combinations, p_soft_no_values = sample_param_combinations(
        np.array(p_yes_bounds) / 100,
        np.array(p_hard_no_bounds) / 100,
        np.array(pressure_bounds) / 100,
        n_rep,
    )

    if params_combinations is None:
        return fig, None, "The pertentages above are too high", {"color": "red"}

    # p_yes_values = np.random.uniform(p_yes_bounds[0], p_yes_bounds[1], size=n_rep) / 100

    # p_hard_no_values = (
    #     np.random.uniform(p_hard_no_bounds[0], p_hard_no_bounds[1], size=n_rep) / 100
    # )
    # pressure_values = (
    #     np.random.uniform(pressure_bounds[0], pressure_bounds[1], size=n_rep) / 100
    # )

    max_delivery = 0.05
    a = 0.0007
    F = lambda t: min(a * np.exp(0.2 * t / 7), max_delivery) * N

    # np.random.seed(1234567)

    data = run_model_sampling(params_combinations, start_date, end_date, F, N)

    # ---- plot model results ----

    colors = px.colors.qualitative.Safe

    fig = add_line(fig, data["pv_dates"], data["pv_mean"], "royalblue", "Model", 1, 1)
    fig = add_line(
        fig, data["pv_dates"], data["pv_upper"], colors[0], None, 1, 1, width=0
    )
    fig = add_line(
        fig,
        data["pv_dates"],
        data["pv_lower"],
        colors[0],
        None,
        1,
        1,
        width=0,
        fill="tonexty",
    )

    fig = add_line(fig, data["dv_dates"], data["dv_mean"], "royalblue", "Model", 1, 2)
    fig = add_line(
        fig, data["dv_dates"], data["dv_upper"], colors[0], None, 1, 2, width=0
    )

    data["dv_lower"][data["dv_lower"] < 0] = 0.0
    fig = add_line(
        fig,
        data["dv_dates"],
        data["dv_lower"],
        colors[0],
        None,
        1,
        2,
        width=0,
        fill="tonexty",
    )

    # ----- add curves with data from the selected countries ----

    df = country_data["people_fully_vaccinated_per_hundred"]
    for i, country in enumerate(selected_countries):
        g = df[country].dropna()
        fig = add_line(fig, g.index, g, colors[i + 1], country, 1, 1)

    df = country_data["daily_vaccinations_per_million"]
    for i, country in enumerate(selected_countries):
        g = df[country].dropna()
        fig = add_line(fig, g.index, g, colors[i + 1], country, 1, 2)

    fig.update_yaxes(range=[0, 100], row=1, col=1)
    # fig.update_layout(height=400, width=1100)

    p_soft_no_values = 100 * np.array(p_soft_no_values)
    a = max(np.mean(p_soft_no_values) - np.std(p_soft_no_values), 0)
    b = np.mean(p_soft_no_values) + np.std(p_soft_no_values)
    a_str = "{0:.0f}".format(a)
    b_str = "{0:.0f}".format(b)
    if abs(a - b) < 1:
        msg_agnostics_pct = "Agnosticts: " + a_str + "%"
    else:
        msg_agnostics_pct = "Agnosticts: " + a_str + " - " + b_str + "%"

    return fig, None, msg_agnostics_pct, dict()


@app.callback(
    Output(component_id="output-p-yes-value", component_property="children"),
    Input(component_id="slider-p-yes", component_property="value"),
)
def update_output_div(values):
    return f"Pro-vaccines: {values[0]} - {values[1]}%"


@app.callback(
    Output(component_id="output-p-hard-no-value", component_property="children"),
    Input(component_id="slider-p-hard-no", component_property="value"),
)
def update_output_div(values):
    return f"Anti-vaccines: {values[0]} - {values[1]}%"


# @app.callback(
#     Output(component_id="output-p-soft-no-value", component_property="children"),
#     Input("slider-p-yes", "value"),
#     Input("slider-p-hard-no", "value"),
# )
# def update_output_div(p_yes, p_hard_no):
#     v = 100 - (p_yes + p_hard_no)
#     return r"Agnosticts: {0:.0f}%".format(v)


@app.callback(
    Output(component_id="output-pressure-value", component_property="children"),
    Input(component_id="slider-pressure", component_property="value"),
)
def update_output_div(values):
    return f"Pressure on the agnostics: {values[0]} - {values[1]}%"


# Run the server
if __name__ == "__main__":
    app.run_server(debug=True)
    # app.run_server(host='0.0.0.0', port=80)
