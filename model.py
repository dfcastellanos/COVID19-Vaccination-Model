import numpy as np
import pandas as pd
import scipy.stats as st
import time


def run_single_realization(
    p_yes, p_no_hard, pressure, tau, nv_0, nv_max, max_day_number, N
):

    # tau = 4
    # nv_0 = 0.001
    # nv_max = 5

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

    nv_available = 0
    cum_number_vac_received = 0
    day_number = 0
    vaccine_already = np.repeat(False, N)

    people_fully_vaccinated_per_hundred = list()
    daily_vaccinations_per_million = list()

    while day_number < max_day_number:

        # upated vaccines
        if day_number % 7 == 0.0:
            nv_arriving = int(F(day_number))
            assert nv_arriving >= 0
            nv_available += nv_arriving
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
        proc_vac_available = (2.0 / 7.0) * nv_available / N
        daily_vaccinations = np.zeros(N)
        daily_vaccinations[w_waiting] = (
            np.random.uniform(size=N_waiting) < proc_vac_available
        )

        # no more vaccines than available can be applied
        x = nv_available - np.sum(daily_vaccinations)
        if x < 0:
            w_today = daily_vaccinations == True
            corrected_vaccination_today = daily_vaccinations[w_today]
            corrected_vaccination_today[: int(abs(x))] = False
            daily_vaccinations[w_today] = corrected_vaccination_today

        x = nv_available - np.sum(daily_vaccinations)
        assert x >= 0

        vaccine_already = vaccine_already + daily_vaccinations
        nv_available -= np.sum(daily_vaccinations)
        fract_pop_vaccinated = np.sum(vaccine_already) / N

        # change soft not for a yes
        w_no_soft = C == 1
        n_no_soft = np.sum(w_no_soft)
        z = np.random.uniform(size=n_no_soft) < fract_pop_vaccinated * pressure
        C[w_no_soft] = np.where(z, 2, 1)

        day_number += 1

        people_fully_vaccinated_per_hundred.append(fract_pop_vaccinated * 100)
        daily_vaccinations_per_million.append(np.sum(daily_vaccinations) * 1e6 / N)

    return people_fully_vaccinated_per_hundred, daily_vaccinations_per_million


def run_model_sampling(params_sets, start_date, end_date, CI, N):

    starting_time = time.time()

    dates = pd.date_range(start_date, end_date, freq="1d")
    max_days = len(dates)

    pv = list()
    dv = list()
    for p_yes, p_hard_no, pressure, tau, nv_0, nv_max in params_sets:
        pv_, dv_ = run_single_realization(
            p_yes, p_hard_no, pressure, tau, nv_0, nv_max, max_days, N
        )
        elapsed_time = time.time() - starting_time
        if elapsed_time > 30:
            return None

        pv.append(pv_)
        dv.append(dv_)
    pv = np.vstack(pv)
    dv = np.vstack(dv)

    # build a DataFrame and perform a weekly average of the daily doses. Then, get the data back into a numpy array.
    # Note: the average is over a time window, but simulation repetitions are not mixed here
    filtered_dv_df = pd.DataFrame(dv.T, index=dates).reindex(
        pd.date_range(start=start_date, end=end_date, freq="7d")
    )
    dv = filtered_dv_df.values.T
    dates_dv = filtered_dv_df.index

    # multiply by 2 to take into account that in the real world data, for most
    # of the countries (at least for EU) a full vaccination counts as
    # 2 units
    # dv *= 2
    # NOTE: the graphs show that without this factor, the match with the reference
    # data is better

    # get confidence intervals for each date, computed accros repetitions
    # CI of 1.0 produces same result as 0, let's allow only till 0.99 instead
    CI = min(CI, 0.99)
    fun = lambda x: st.t.interval(CI, len(x) - 1, loc=np.mean(x), scale=np.std(x))
    pv_CI = np.vstack(list(map(fun, pv.T))).T
    dv_CI = np.vstack(list(map(fun, dv.T))).T

    pv_mean = pv.mean(axis=0)
    dv_mean = dv.mean(axis=0)

    data = {
        "pv_dates": dates,
        "pv_mean": pv_mean,
        "pv_upper": pv_CI[1],
        "pv_lower": pv_CI[0],
        "dv_dates": dates_dv,
        "dv_mean": dv_mean,
        "dv_upper": dv_CI[1],
        "dv_lower": dv_CI[0],
    }

    return data


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

            params_combinations.append([p_yes, p_hard_no, pressure, tau, nv_0, nv_max])
            p_soft_no_values.append(1 - (p_yes + p_hard_no))

    return params_combinations, p_soft_no_values
