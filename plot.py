import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def get_country_data():

    df = pd.read_csv(
        "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/vaccinations/vaccinations.csv"
    )
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["location", "date"])

    columns_to_keep = [
        "location",
        "date",
        "people_vaccinated_per_hundred",
        "daily_vaccinations_per_million",
    ]

    df = df.loc[:, columns_to_keep]

    avail_countries = df["location"].unique()
    country_data = pd.pivot_table(
        df,
        columns="location",
        values=[
            "people_vaccinated_per_hundred",
            "daily_vaccinations_per_million",
        ],
        index="date",
    )

    return avail_countries, country_data


colors = px.colors.qualitative.Safe

to_plot = [
    "people_vaccinated_per_hundred",
    "daily_vaccinations_per_million",
    "cum_number_vac_received_per_hundred",
    "vaccines_in_stock_per_hundred",
]


def add_line(
    fig, x, y, color, name=None, row=1, col=1, fill="none", width=2, annotation=False
):

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
    if annotation:

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


def plot_model_results(model_results, CI):

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "People vaccinated per hundred",
            "Daily vaccinations per million",
            "Vaccines received per hundred",
            "Vaccines in stock per hundred",
        ),
        horizontal_spacing=0.08,
        vertical_spacing=0.15,
    )

    for n, k in enumerate(to_plot):

        i, j = np.unravel_index(n, [2, 2])
        i += 1
        j += 1

        # ---- plot model results ----
        # the first automatic call will have no stored model_results and it will be None

        if model_results is not None:
            df = model_results[k]
            fig = add_line(
                fig,
                df["dates"],
                df["mean"],
                "royalblue",
                "Model",
                i,
                j,
                annotation=True,
            )
            fig = add_line(
                fig,
                df["dates"],
                df["upper"],
                colors[0],
                f"Model CI={CI}%",
                i,
                j,
                width=0,
                annotation=False,
            )
            fig = add_line(
                fig,
                df["dates"],
                df["lower"],
                colors[0],
                f"Model CI={CI}%",
                i,
                j,
                width=0,
                fill="tonexty",
                annotation=False,
            )

    # fig.update_yaxes(range=[0, 100], row=1, col=1)
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=50))  # height=400, width=1100)

    return fig


def plot_country_data(fig, selected_countries, country_data, initialized=False):

    for n, k in enumerate(to_plot):

        i, j = np.unravel_index(n, [2, 2])
        i += 1
        j += 1

        # ----- add curves with data from the selected countries ----
        if k in country_data.columns:
            df = country_data[k]
            for ncolor, country in enumerate(selected_countries):
                g = df[country].dropna()
                fig = add_line(
                    fig,
                    g.index,
                    g,
                    colors[ncolor + 1],
                    country,
                    i,
                    j,
                    width=1,
                    annotation=True,
                )
        else:
            # Some of the results that we obtain from the model do not have equivalent real world data.
            # This causes some plots not to show up initially, until the model has ben run at least once.
            # If model results are not yet available, we place a 'no data' annotation in those plots.
            # That will make Plotly draw the axes so the user will be aware of them from the begining.
            if initialized:

                fig = add_line(
                    fig,
                    [0],
                    [0],
                    colors[0],
                    "No data to show",
                    i,
                    j,
                    width=1,
                    annotation=True,
                )

    return
