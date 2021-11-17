import numpy as np
import pandas as pd
import scipy.stats as st


def run_single_realization(p_yes, p_no_hard, pressure, max_day_number, F, N):

    assert p_yes + p_no_hard < 1.0

    p_no_soft = 1 - (p_yes + p_no_hard)

    C = np.zeros(N)
    n_yes = int(p_yes * N)
    n_no_soft = int(p_no_soft * N)
    n_no_hard = int(p_no_hard * N)
    C[:n_yes] = 2
    C[n_yes : n_yes + n_no_soft + 1] = 1
    C[n_yes + n_no_soft + 1 : (n_yes + n_no_soft + 1) + n_no_hard + 1] = 0

    nv_available = 0
    cum_number_vac_received = 0
    day_number = 0
    vaccine_already = np.repeat(False, N)

    people_fully_vaccinated_per_hundred = list()
    daily_vaccinations_per_million = list()

    while day_number < max_day_number:

        # upated vaccines
        if day_number % 7 == 0.0:
            # approx. number of pfizer for spain
            # max_delivery = 0.05
            # a = 0.0007
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


def run_model_sampling(
    p_yes_values, p_hard_no_values, pressure_values, start_date, end_date, F, N
):

    assert len(p_yes_values) == len(p_hard_no_values)
    assert len(p_hard_no_values) == len(pressure_values)

    dates = pd.date_range(start_date, end_date, freq="1d")
    max_days = len(dates)

    pv = list()
    dv = list()
    for p_yes, p_hard_no, pressure in zip(
        p_yes_values, p_hard_no_values, pressure_values
    ):
        pv_, dv_ = run_single_realization(p_yes, p_hard_no, pressure, max_days, F, N)
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
    dv *= 2

    # get confidence intervals for each date, computed accros repetitions
    fun = lambda x: st.t.interval(0.95, len(x) - 1, loc=np.mean(x), scale=np.std(x))
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
