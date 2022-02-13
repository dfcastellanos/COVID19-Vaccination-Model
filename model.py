"""    
License
-------
Copyright (C) 2021  - David Fernández Castellanos

You can use this software, redistribute it, and/or modify it under the 
terms of the Creative Commons Attribution 4.0 International Public License.


Explanation
---------
This module contains the statistical model of the COVID-19 vaccination campaign
described in assets/model_explanation.html. Moreover, it also includes functions
to sample the model's parameter space.
"""

import numpy as np
import pandas as pd
import time
import datetime
import functools
from collections import defaultdict
import argparse
from argparse import RawTextHelpFormatter

from plot import plot_model_results


def run_single_realization(
    p_pro, p_anti, pressure, tau, nv_0, nv_max, max_day_number, N
):

    """
    Run a single realization of the statistical model of vaccination campaigns.
    This single run corresponds to simulating the evolution of the vaccination campaign
    as a function of time. See the assets/model_explanation.html for details on the model.

    Parameters
    ----------
    p_pro : float
        The probability that a certain person belongs to the pro-vaccines group
    p_anti :  float
        The probability that a specific person belongs to the anti-vaccines group
    pressure : float
        Strenght of the social pressure effect
    tau : float
        Duplication time of the weekly arriving vaccines
    nv_0 : float
        Initial stock of vaccines, measured as a fraction over the population size
    nv_max : floa
        Maximum weekly delivery capacity, measured as a fraction over the population size
    max_day_number : int
        Number of days that are going to be simulated
    N : int
        The population size

    Returns
    -------
    Dictionary (key:string, value:list)
        Dictionary with different data collected as a function of the day number
    """

    assert p_pro + p_anti <= 1.0
    p_agnostics = 1 - (p_pro + p_anti)

    n_pro = int(p_pro * N)
    n_agnostics = int(p_agnostics * N)

    F = lambda t: min(nv_0 * np.exp(np.log(2) * t / (tau * 7)), nv_max) * N

    day_number = 0
    vaccines_stock = 0
    cum_number_vac_received = 0
    n_vaccinated = 0
    n_waiting = n_pro - n_vaccinated

    people_vaccinated_per_hundred = list()
    daily_vaccinations_per_million = list()
    cum_number_vac_received_per_hundred = list()
    vaccines_in_stock_per_hundred = list()

    while day_number < max_day_number:

        # ------ add arriving vaccines to the stock ------
        if day_number % 7 == 0.0:
            nv_arriving = int(F(day_number))
        else:
            nv_arriving = 0

        assert nv_arriving >= 0
        vaccines_stock += nv_arriving
        cum_number_vac_received += nv_arriving

        # ------  apply vaccines ------

        # prob. of having it available does not take into account only people waitting, but the whole population
        # for example, if the population is big, the vaccines will be more spread and less likely to reach anyone
        # however is we use the people waiting, we assume the vaccines are being distributed specifically among
        # them. Moreover, this is the prob. of having it available a specific day. Since we work in cycles of
        # 7 days and only ~2 days a week is possible to have it, we should multiply it by ~2/7 to get an effective
        # prob. per day;
        proc_vac_available = (2.0 / 7.0) * vaccines_stock / N

        delta_n_vacc = np.random.poisson(n_waiting * proc_vac_available)
        # don't apply more vaccines than available
        delta_n_vacc = min(delta_n_vacc, vaccines_stock)
        # don't apply more vaccines than people waiting for it
        delta_n_vacc = min(delta_n_vacc, n_waiting)
        n_vaccinated += delta_n_vacc
        n_waiting -= delta_n_vacc
        vaccines_stock -= delta_n_vacc
        fract_pop_vaccinated = n_vaccinated / N

        # ------ convert agnostics ------
        prob_change_mind = fract_pop_vaccinated * pressure
        delta_n_agnos = np.random.poisson(n_agnostics * prob_change_mind)
        # don't convert more agnostics than agnostics available
        delta_n_agnos = min(delta_n_agnos, n_agnostics)
        n_agnostics -= delta_n_agnos
        n_waiting += delta_n_agnos

        day_number += 1

        people_vaccinated_per_hundred.append(fract_pop_vaccinated * 100)
        daily_vaccinations_per_million.append(delta_n_vacc * 1e6 / N)
        cum_number_vac_received_per_hundred.append(cum_number_vac_received * 100 / N)
        vaccines_in_stock_per_hundred.append(vaccines_stock * 100 / N)

    data = {
        "people_vaccinated_per_hundred": people_vaccinated_per_hundred,
        "daily_vaccinations_per_million": daily_vaccinations_per_million,
        "cum_number_vac_received_per_hundred": cum_number_vac_received_per_hundred,
        "vaccines_in_stock_per_hundred": vaccines_in_stock_per_hundred,
    }

    return data


@functools.lru_cache(maxsize=10)
def run_sampling(params, start_date, end_date, CI, N, max_running_time=None):

    """
    Sample the model's parameter space. For that, the model is run for
    each input combination of parameters.

    Parameters
    ----------
    params : tuple of tuples
        Each of the tuples contain a combination of model parameters
        (p_pro, p_anti, pressure, tau, nv_0, nv_max, max_day_number).
        See run_single_realization for details.
    start_date : datetime.datetime
        Starting date
    end_date : datetime.datetime
        The last date at which the model run stops
    CI : float
        Value of the quantile used for establishing the confidence intervals
    N : int
        The population size

    Returns
    -------
    Dictionary of dictionaries
        Each dictionary key corresponds to the different quantities returned by run_single_realization.
        Each of the values is another dictionary of lists that contains the mean of the quantity, its upper
        and lower confidence intervals, and the dates associated with each list index.
    """

    starting_time = time.time()

    dates = pd.date_range(start_date, end_date, freq="1d")
    max_days = len(dates)

    data = defaultdict(list)
    number_finished_samples = 0
    for p_pro, p_anti, pressure, tau, nv_0, nv_max in params:
        data_ = run_single_realization(
            p_pro, p_anti, pressure, tau, nv_0, nv_max, max_days, N
        )

        # merge a dict into a dict of lists
        for k, v in data_.items():
            data[k].append(v)

        number_finished_samples += 1

        elapsed_time = time.time() - starting_time
        if max_running_time is not None and elapsed_time > max_running_time:
            break

    # we work with numpy arrays since Dash Store cannot handle DataFarmes

    data = {k: {"dates": dates, "samples": np.vstack(v)} for k, v in data.items()}

    # Note: the average is over a time window, but samples are not mixed here
    for k in ["daily_vaccinations_per_million"]:
        v = data[k]["samples"]
        df = pd.DataFrame(np.vstack(v).T, index=dates)
        # The model simulates the dynamics of the application of a single dosis, but actually
        # (most) those who got a first dosis, will get a second one ~30 days later. Since such second
        # doses are included in the daily_vaccinations_per_million from the real-world data,
        # we must ialso ncluded them in the model results. For that, we shift the original applied
        # doses by 30 days and concatenate the DataFrames.
        # The fact that all the second doses are appended after all the first ones
        # doesn't matter since afterward we will reindex to compute a moving average
        shifted_df = pd.DataFrame(
            np.vstack(v).T, index=dates + datetime.timedelta(days=30)
        )
        df = df.add(shifted_df, fill_value=0.0)
        # compute averages over windows of 7 days, as in the real-world data
        df = df.reindex(pd.date_range(start=start_date, end=end_date, freq="7d"))
        # do not call df.index.values, because that transforms Timestamps to numpy.datetime, and plotly seems to prefer Timestamps
        data[k]["dates"] = df.index
        data[k]["samples"] = df.values.T

    # get confidence intervals for each date, computed accros samples
    data_CI = defaultdict(dict)
    for k in data.keys():
        samples = data[k]["samples"]
        quantiles = np.quantile(samples, [(1 - CI)/2., (1 + CI)/2.], axis=0)
        data_CI[k]["upper"] = quantiles[1]
        data_CI[k]["lower"] = quantiles[0]
        data_CI[k]["mean"] = samples.mean(axis=0)
        data_CI[k]["dates"] = data[k]["dates"]

    data_CI["number_finished_samples"] = number_finished_samples

    return data_CI


def sample_param_combinations(
    p_pro_bounds,
    p_anti_bounds,
    pressure_bounds,
    tau_bounds,
    nv_0_bounds,
    nv_max_bounds,
    n_rep,
):

    """
    Create a sample of parameter combinations. Each parameter
    combination is created by drawing values from uniform distributions
    with bounds defined by the function's arguments.

    Parameters
    ----------
    p_pro_bounds : 2D-tuple of floats
        Lower and upper bound for the probability that a certain person belongs to the pro-vaccines group
    p_anti_bounds : 2D-tuple of floats
        Lower and upper bound for the probability that a specific person belongs to the anti-vaccines group
    pressure_bounds : 2D-tuple of floats
        Lower and upper bound for the strength of the social pressure effect
    tau_bounds : 2D-tuple of floats
        Lower and upper bound for the duplication time of the weekly arriving vaccines
    nv_0_bounds : 2D-tuple of floats
        Lower and upper bound for the initial stock of vaccines measured as a fraction over the population size
    nv_max_bounds : 2D-tuple of floats
        Lower and upper bound for the maximum weekly delivery capacity measured as a fraction over the population size
    n_rep : int
        Number of parameter combination, i.e., number of random parameter samples drawn

    Returns
    -------
    Tuple of tuples
        Each of the tuples contain a combination of model parameters
        (p_pro, p_anti, pressure, tau, nv_0, nv_max, max_day_number).
    Tuple
        The probability that a person belongs to the agnostics group
    """

    params_combinations = list()
    p_soft_no_values = list()
    n = 0
    while len(params_combinations) < n_rep:

        p_pro = np.random.uniform(p_pro_bounds[0], p_pro_bounds[1])
        p_anti = np.random.uniform(p_anti_bounds[0], p_anti_bounds[1])
        # use rejection sampling to ensure that p_anti + p_pro < 1
        if p_pro + p_anti > 1.0:
            # rejection
            n += 1
            if n > n_rep * 10:
                # if the ammount of rejections is not too high, it means
                # that given upper and lower bounds of p_anti and p_pro are
                # mutually incompatible. Thus, we abort the parameter sampling
                return None, None
            else:
                continue
        else:
            pressure = np.random.uniform(pressure_bounds[0], pressure_bounds[1])
            tau = np.random.uniform(tau_bounds[0], tau_bounds[1])
            nv_0 = np.random.uniform(nv_0_bounds[0], nv_0_bounds[1])
            nv_max = np.random.uniform(nv_max_bounds[0], nv_max_bounds[1])

            # work with tuples so that we can later use @functools.lru_cache, since it need
            # hashable types
            params_combinations.append(
                tuple([p_pro, p_anti, pressure, tau, nv_0, nv_max])
            )
            p_soft_no_values.append(1 - (p_pro + p_anti))

    return tuple(params_combinations), tuple(p_soft_no_values)


def run_model(
    # populatio parameters
    p_pro_bounds,
    p_anti_bounds,
    pressure_bounds,
    # vaccinations parameters
    tau_bounds,
    nv_0_bounds,
    nv_max_bounds,
    # samping
    CI,
    n_rep,
    N,
    date_range,
    max_running_time=None,
):

    # default output messages
    msg_agnostics_pct = "Agnosticts: "
    msg_error = ""

    # some sliders use values 0-100
    params_combinations, p_soft_no_values = sample_param_combinations(
        np.array(p_pro_bounds) / 100,
        np.array(p_anti_bounds) / 100,
        np.array(pressure_bounds),
        np.array(tau_bounds),
        np.array(nv_0_bounds) / 100,
        np.array(nv_max_bounds) / 100,
        n_rep,
    )

    if params_combinations is not None:
        # evaluate the agnostics population from the pro and anti vaccines samples
        p_soft_no_values = 100 * np.array(p_soft_no_values)
        a = max(np.mean(p_soft_no_values) - np.std(p_soft_no_values), 0)
        b = np.mean(p_soft_no_values) + np.std(p_soft_no_values)
        a_str = "{0:.0f}".format(a)
        b_str = "{0:.0f}".format(b)
        # if the uncertainty interval is smaller than 1%, report one value instead of the interval
        if abs(a - b) < 1:
            msg_agnostics_pct += a_str + "%"
        else:
            msg_agnostics_pct += a_str + " - " + b_str + "%"
    else:
        msg_error = "ERROR: The pertentages of pro- and anti-vaccines are simultaneously too high. Please reduce them."
        return None, msg_error, msg_agnostics_pct

    model_results = run_sampling(
        params_combinations,
        date_range["start_date"],
        date_range["end_date"],
        CI / 100,
        N,
        max_running_time,
    )

    if max_running_time is not None:
        number_finished_samples = model_results["number_finished_samples"]
        if number_finished_samples < len(params_combinations):
            msg_error = f"ERROR: Maximum computation time of {max_running_time}s exceeded. Only {number_finished_samples} of the desired {len(params_combinations)} Monte Carlo runs were performed."

    return model_results, msg_error, msg_agnostics_pct


class SplitArgsStr(argparse.Action):
    def __call__(self, parser, namespace, values_str, option_string=None):
        values = values_str.split(",")
        # If ',' is not in the string, the input corresponds to a single value.
        # Create list of two values with it.
        if len(values) == 1:
            values += values
        setattr(namespace, self.dest, values)


class SplitArgsFloat(argparse.Action):
    def __call__(self, parser, namespace, values_str, option_string=None):
        values = [float(x) for x in values_str.split(",")]
        # If ',' is not in the string, the input corresponds to a single value.
        # Create list of two values with it.
        if len(values) == 1:
            values += values
        setattr(namespace, self.dest, values)


def main():

    description = """  
    This program performs a Monte Carlo sampling of a statistical model of the
    COVID-19 vaccination campaign (you can find a detailed explanation of
    the model in assets/model_explanation.html). 
    
    In each Monte Carlo run, the value of each parameter is drawn from a uniform
    probability distribution. The bounds of each distribution are defined in the
    command line call as comma-separated strings for each parameter. If instead
    of a comma-separated string, a single value is given, that parameter will 
    assume in every Monte Carlo run exactly that specific value.
    
    When the sampling is complete, the results are automatically rendered as an
    interactive plot in your default internet browser.
    
    Example call: 
        'python model.py --pro=30,40 --anti=17,40 --pressure=0.02,0.025 --dupl_time=3,4 --init_stock=0.2,0.24 --max_delivery=10,10 --date_range=2020-12-30,2021-12-1'
   
    Author: David Fernández Castellanos.

    Related links:
    - The author's website: https://www.davidfcastellanos.com
    - The source code: https://github.com/kastellane/COVID19-Vaccination-Model
    - An interactive web app version: https://covid19-vaccination-app.davidfcastellanos.com
    - An associated blog post: https://www.davidfcastellanos.com/covid-19-vaccination-model    
    """

    parser = argparse.ArgumentParser(
        description=description, formatter_class=RawTextHelpFormatter
    )

    parser.add_argument(
        "--pro",
        type=str,
        help="comma-separated upper and lower bounds for the probability that a certain person belongs to the pro-vaccines group",
        default="30,40",
        action=SplitArgsFloat,
        required=True,
    )

    parser.add_argument(
        "--anti",
        type=str,
        help="comma-separated upper and lower bounds for the probability that a specific person belongs to the anti-vaccines group",
        default="30,40",
        action=SplitArgsFloat,
        required=True,
    )

    parser.add_argument(
        "--pressure",
        type=str,
        help="comma-separated upper and lower bounds for the strenght of the social pressure effect",
        default="0.02,0.025",
        action=SplitArgsFloat,
        required=True,
    )

    parser.add_argument(
        "--dupl_time",
        type=str,
        help="comma-separated upper and lower bounds for the duplication time of the weekly arriving vaccines",
        default="3,4",
        action=SplitArgsFloat,
        required=True,
    )

    parser.add_argument(
        "--init_stock",
        type=str,
        help="comma-separated upper and lower bounds for the initial stock of vaccines, measured as a percentege of the population size",
        default="0.2,0.2",
        action=SplitArgsFloat,
        required=True,
    )

    parser.add_argument(
        "--max_delivery",
        type=str,
        help="comma-separated upper and lower bounds for the maximum weekly delivery capacity, measured as a percentage over the population size",
        default="10,10",
        action=SplitArgsFloat,
        required=True,
    )

    parser.add_argument(
        "--mc_samples",
        type=int,
        help="number of Monte Carlo samples (optional)",
        default="100",
    )

    parser.add_argument(
        "--date_range",
        type=str,
        help="comma-separated starting and ending dates (optional)",
        default="2020-12-30,2021-12-1",
        action=SplitArgsStr,
        required=True,
    )

    parser.add_argument(
        "--CI",
        type=float,
        help="value of the quantile used for establishing the confidence intervals",
        default="0.95",
    )

    args = vars(parser.parse_args())

    # populatio parameters
    p_pro_bounds = args["pro"]
    p_anti_bounds = args["anti"]
    pressure_bounds = args["pressure"]
    # vaccinations parameters
    tau_bounds = args["dupl_time"]
    nv_0_bounds = args["init_stock"]
    nv_max_bounds = args["max_delivery"]
    # samping
    n_rep = args["mc_samples"]
    N = 50000
    start_date = args["date_range"][0]
    end_date = args["date_range"][1]
    CI = args["CI"]

    date_range = dict(start_date=start_date, end_date=end_date)

    model_results, msg_error, msg_agnostics_pct = run_model(
        # populatio parameters
        p_pro_bounds,
        p_anti_bounds,
        pressure_bounds,
        # vaccinations parameters
        tau_bounds,
        nv_0_bounds,
        nv_max_bounds,
        # samping
        CI,
        n_rep,
        N,
        date_range,
    )

    if msg_error != "":
        print(msg_error)
    else:
        fig = plot_model_results(model_results, CI)
        # plot_country_data(fig, selected_countries, country_data)
        fig.show(renderer="browser")

    return


if __name__ == "__main__":
    main()
