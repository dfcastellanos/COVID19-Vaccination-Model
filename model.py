import numpy as np
import pandas as pd
import time
import functools
from collections import defaultdict


def run_single_realization(
    p_yes, p_no_hard, pressure, tau, nv_0, nv_max, max_day_number, N
):

    assert p_yes + p_no_hard <= 1.0

    p_no_soft = 1 - (p_yes + p_no_hard)

    C = np.zeros(N)
    n_yes = int(p_yes * N)
    n_no_soft = int(p_no_soft * N)
    n_no_hard = int(p_no_hard * N)
    C[:n_yes] = 2
    C[n_yes : n_yes + n_no_soft + 1] = 1
    C[n_yes + n_no_soft + 1 : (n_yes + n_no_soft + 1) + n_no_hard + 1] = 0

    F = lambda t: min(nv_0 * np.exp(np.log(2) * t / (tau * 7)), nv_max) * N

    vaccines_stock = 0
    cum_number_vac_received = 0
    day_number = 0
    vaccine_already = np.repeat(False, N)

    people_fully_vaccinated_per_hundred = list()
    daily_vaccinations_per_million = list()
    cum_number_vac_received_per_hundred = list()
    vaccines_in_stock_per_hundred = list()

    while day_number < max_day_number:

        if day_number % 7 == 0.0:
            nv_arriving = int(F(day_number))
        else:
            nv_arriving = 0

        assert nv_arriving >= 0
        vaccines_stock += nv_arriving
        cum_number_vac_received += nv_arriving

        # apply vaccine to yes people
        w_yes = C == 2
        w_waiting = np.logical_and(np.logical_not(vaccine_already), w_yes)

        # no all that want it have it available, only  a fraction of them gets it
        N_waiting = np.sum(w_waiting)

        # prob. of having it available does not take into account only people waitting, but the whole population
        # for example, if the population is big, the vaccines will be more spread and less likely to reach anyone
        # however is we use the people waiting, we assume the vaccines are being distributed specifically among
        # them. Moreover, this is the prob. of having it available a specific day. Since we work in cycles of
        # 7 days and only ~2 days a week is possible to have it, we should multiply it by ~2/7 to get an effective
        # prob. per day;
        proc_vac_available = (2.0 / 7.0) * vaccines_stock / N
        daily_vaccinations = np.zeros(N)
        daily_vaccinations[w_waiting] = (
            np.random.uniform(size=N_waiting) < proc_vac_available
        )

        # no more vaccines than available can be applied
        x = vaccines_stock - np.sum(daily_vaccinations)
        if x < 0:
            w_today = daily_vaccinations == True
            corrected_vaccination_today = daily_vaccinations[w_today]
            corrected_vaccination_today[: int(abs(x))] = False
            daily_vaccinations[w_today] = corrected_vaccination_today

        x = vaccines_stock - np.sum(daily_vaccinations)
        assert x >= 0

        vaccine_already = vaccine_already + daily_vaccinations
        vaccines_stock -= np.sum(daily_vaccinations)
        fract_pop_vaccinated = np.sum(vaccine_already) / N

        # change soft not for a yes
        w_no_soft = C == 1
        n_no_soft = np.sum(w_no_soft)
        z = np.random.uniform(size=n_no_soft) < fract_pop_vaccinated * pressure
        C[w_no_soft] = np.where(z, 2, 1)

        day_number += 1

        people_fully_vaccinated_per_hundred.append(fract_pop_vaccinated * 100)
        daily_vaccinations_per_million.append(np.sum(daily_vaccinations) * 1e6 / N)
        cum_number_vac_received_per_hundred.append(cum_number_vac_received * 100 / N)
        vaccines_in_stock_per_hundred.append(vaccines_stock * 100 / N)

    data = {
        "people_fully_vaccinated_per_hundred": people_fully_vaccinated_per_hundred,
        "daily_vaccinations_per_million": daily_vaccinations_per_million,
        "cum_number_vac_received_per_hundred": cum_number_vac_received_per_hundred,
        "vaccines_in_stock_per_hundred": vaccines_in_stock_per_hundred,
    }

    return data


@functools.lru_cache(maxsize=10)
def run_model_sampling(params_sets, start_date, end_date, CI, N):

    starting_time = time.time()

    dates = pd.date_range(start_date, end_date, freq="1d")
    max_days = len(dates)

    data = defaultdict(list)
    number_finished_samples = 0
    for p_yes, p_hard_no, pressure, tau, nv_0, nv_max in params_sets:
        data_ = run_single_realization(
            p_yes, p_hard_no, pressure, tau, nv_0, nv_max, max_days, N
        )

        # merge a dict into a dict of lists
        for k, v in data_.items():
            data[k].append(v)

        number_finished_samples += 1

        elapsed_time = time.time() - starting_time
        if elapsed_time > 30:
            break

    # we work with numpy arrays since Dash Store cannot handle DataFarmes

    data = {k: {"dates": dates, "samples": np.vstack(v)} for k, v in data.items()}

    # Note: the average is over a time window, but samples are not mixed here
    for k in ["daily_vaccinations_per_million"]:
        v = data[k]["samples"]
        df = pd.DataFrame(np.vstack(v).T, index=dates).reindex(
            pd.date_range(start=start_date, end=end_date, freq="7d")
        )
        # do not call df.index.values, because that transforms Timestamps to numpy.datetime, and plotly seems to prefer Timestamps
        data[k]["dates"] = df.index
        data[k]["samples"] = df.values.T

    # multiply by 2 to take into account that in the real world data, for most
    # of the countries (at least for EU) a full vaccination counts as
    # 2 units
    # dv *= 2
    # NOTE: the graphs show that without this factor, the match with the reference
    # data is better

    # get confidence intervals for each date, computed accros samples
    data_CI = defaultdict(dict)
    for k in data.keys():
        samples = data[k]["samples"]
        quantiles = np.quantile(samples, [1 - CI, CI], axis=0)
        data_CI[k]["upper"] = quantiles[1]
        data_CI[k]["lower"] = quantiles[0]
        data_CI[k]["mean"] = samples.mean(axis=0)
        data_CI[k]["dates"] = data[k]["dates"]

    data_CI["number_finished_samples"] = number_finished_samples

    return data_CI


def sample_param_combinations(
    p_yes_bounds,
    p_hard_no_bounds,
    pressure_bounds,
    tau_bounds,
    nv_0_bounds,
    nv_max_bounds,
    n_rep,
):

    params_combinations = list()
    p_soft_no_values = list()
    n = 0
    while len(params_combinations) < n_rep:
        if n > n_rep * 10:
            return None, None
        else:
            p_yes = np.random.uniform(p_yes_bounds[0], p_yes_bounds[1])
            p_hard_no = np.random.uniform(p_hard_no_bounds[0], p_hard_no_bounds[1])
            if p_yes + p_hard_no > 1.0:
                n += 1
                continue
            pressure = np.random.uniform(pressure_bounds[0], pressure_bounds[1])
            tau = np.random.uniform(tau_bounds[0], tau_bounds[1])
            nv_0 = np.random.uniform(nv_0_bounds[0], nv_0_bounds[1])
            nv_max = np.random.uniform(nv_max_bounds[0], nv_max_bounds[1])

            # work with tuples so that we can later use @functools.lru_cache, since it need
            # hashable types
            params_combinations.append(
                tuple([p_yes, p_hard_no, pressure, tau, nv_0, nv_max])
            )
            p_soft_no_values.append(1 - (p_yes + p_hard_no))

    return tuple(params_combinations), tuple(p_soft_no_values)
