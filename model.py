import numpy as np
import pandas as pd
from datetime import timedelta


def run(
    p_yes, p_no_hard, pressure, cum_number_vac_purchased, start_date, end_date, F, N
):

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
    vaccine_already = np.repeat(False, N)
    date = start_date
    day_number = 0
    data = pd.DataFrame()

    while date <= end_date:

        # upated vaccines
        if day_number % 7 == 0.0 and cum_number_vac_received < cum_number_vac_purchased:
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

        date += timedelta(days=1)
        day_number += 1

        data_today = pd.DataFrame(
            {
                "date": date,
                "people_fully_vaccinated_per_hundred": fract_pop_vaccinated * 100,
                "vaccines_avail_per_hundred": nv_available * 100 / N,
                "ratio_vaccines_received_over_purchased": cum_number_vac_received
                / cum_number_vac_purchased,
                "daily_vaccinations_per_million": np.sum(daily_vaccinations) * 1e6 / N,
            },
            index=[0],
        )
        data = pd.concat([data, data_today])

    data = data.set_index("date")

    return data
