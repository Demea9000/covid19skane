
from calculate_R import *
import numpy as np
import matplotlib.pyplot as plt
import datetime
plt.style.use('seaborn-dark')

pop = 1.362e6


def get_sameday(incidence, days):

    revinc = list(reversed(incidence))[::days]

    return revinc

def get_distribution_per_weekday(a):
    dists = []
    days = []
    for i in range(len(a)-7, len(a)):
        d = get_sameday(a[:i], 7)
        d = get_distribution(a)
        dists.append(d)
        days.append(i)

    return [dists,days]

def match_distribution(dists, days, day):

    for i in range(len(days)):
        dis = dists[i]
        da = days[i]
        if abs(day-da)%7==0:
            return dists[i]
            break

def get_timeaxis(start_year, start_month, start_day, days_forward):

    out = []
    start = datetime.datetime(start_year, start_month, start_day)
    out.append(start.strftime('%Y-%m-%d'))
    for day in range(1, days_forward):
        start += datetime.timedelta(days=1)
        out.append(start.strftime('%Y-%m-%d'))

    return out

def I_t(Rt, I, w, t):
    I0 = 0
    for s in range(1, t):
        I0 += I[t-s]*w[s]
    return Rt*I0

def lambda_t(R, I, w, t):
    L0 = 0
    for s in range(1, t):
        L0 += R[s]*I[s]*w[t-s]
    return L0

def get_total_infected(Incidence):
    tot = np.sum(Incidence)
    return tot



def get_data():
    with open("R_calc.txt", 'r') as file:
        R_future = file.readlines()
        R_future = [float(i) for i in R_future]

    with open("Incidence.txt", 'r') as file:
        I_future = file.readlines()
        I_future = [float(i) for i in I_future]

    with open("weights.txt", 'r') as file:
        w_future = file.readlines()
        w_future = [float(i) for i in w_future]

    return [R_future, I_future, w_future]



def moving_average(data_set, periods=7):
    weights = np.ones(periods) / periods
    ma = np.convolve(data_set, weights, mode='valid')

    ma = np.pad(ma, pad_width=(periods, 0), mode= 'constant', constant_values=0)

    return ma


def get_distribution(a):

    dist = np.array(a)
    mu = np.mean(dist)
    std = np.std(dist)

    return (mu, std)

def simulate_old(lookback=31):

    R_future, I_future, w_future = get_data()
    mu, std = get_distribution(np.diff(I_future))
    muR, stdR = get_distribution(np.diff(R_future))

    for days in range(len(w_future), len(w_future)+lookback):
        tot_coeff = get_total_infected(I_future)/pop
        It = lambda_t(R_future, I_future, w_future, days)
        random_element = np.random.normal(mu, std)
        if It+random_element >= 0:
            It += 0.6*random_element
        I_future.append(It)
        w_future.append(0)
        R_new = R_post(I_future, w_future, Lambda, days)*(1-np.tanh(tot_coeff)) + 0.6*np.random.normal(muR, stdR)
        R_future.append(R_new)


    return [I_future, R_future]



def simulate(start, stop, lookback=31, R_tolerance=0.2):

    R_future, I_future, w_future = get_data()
    R_real, I_real, w_real = get_data()

    R_future = R_future[start:stop-lookback]
    I_future = I_future[start:stop-lookback]
    w_future = w_future[start:stop-lookback]
    R_real = R_real[start:stop]
    I_real=I_real[start:stop]
    w_real=w_real[start:stop]

    # R_future = list(moving_average(R_future, 7))
    # I_future = list(moving_average(I_future, 7))
    # w_future = list(moving_average(w_future, 7))

    mu, std = get_distribution(np.diff(I_future))
    muR, stdR = get_distribution(np.diff(R_real[30:]))

    #R_future[-1] = np.random.normal(muR, stdR)
    for days in range(len(w_future), len(w_future)+lookback):
        tot_coeff = get_total_infected(I_future)/pop
        if tot_coeff>=1:
            break
        It = lambda_t(R_future, I_future, w_future, days)
        random_element = np.random.normal(mu, std)
        if It+random_element >= 0:
            It += .6*random_element
        if It < 0:
            It = 0
        I_future.append(int(It))
        w_future.append(0)
        R_new = R_post(I_future, w_future, Lambda, days)*(1-np.tanh(tot_coeff))
        R_new += .1*np.random.normal(muR, stdR)
        if abs(R_new-R_future[-1]) < R_tolerance:
            R_future.append(R_new)
        else:
            R_future.append(R_future[-1])


    return [I_future, R_future, I_real, R_real]







if __name__ == "__main__":


    I_lst = []
    R_lst = []
    I_interval = []
    np.random.seed(2)
    R, I, w = get_data()
    days_ahead = 280
    lookback = 162
    test_point = len(I)+days_ahead
    lookback += days_ahead
    Nsims = 10
    for i in range(Nsims):
        If, Rf, Ir, Rr = simulate(0, test_point, lookback=lookback)

        I_lst.append(If)
        R_lst.append(Rf)
        I_interval.append(If)


    print("{}% tested positive of Skåne population by end of simulation".format(round(get_total_infected(If)/pop, 3)))
    If = np.array(I_lst)
    Istd = If.std(axis=0)
    If = If.mean(axis=0)

    Rf = np.array(R_lst)
    Rstd = Rf.std(axis=0)
    Rf = Rf.mean(axis=0)

    I_interval = np.array(I_interval)
    I_interval_sum = [np.sum(i) for i in I_interval]
    If_max = If + Istd
    If_min = If - Istd

    Rf_max = Rf+Rstd
    Rf_min = Rf-Rstd

    If_max[:-lookback] = np.nan
    If_min[If_min<0]=0
    If_min[:-lookback] = np.nan
    If[:-lookback] = np.nan


    I[:] = moving_average(I, 6)
    x = np.linspace(1, len(If_min)+1, len(If_min))

    timeaxis = get_timeaxis(2020, 3, 2, len(If))

    print(timeaxis[-lookback])

    plt.plot(timeaxis, If, label='Average incidence of {} simulations'.format(Nsims))
    plt.fill_between(x, If_max, If_min, color='blue', alpha=0.2, label="Standard deviation of {} simulations".format(Nsims))
    plt.plot(timeaxis[:len(I)],I, color='crimson', label='Actual Incidence, 7-day moving average', alpha=.6)
    plt.plot(timeaxis,If_max, color="tab:blue", alpha=0)
    plt.plot(timeaxis,If_min, color="tab:blue", alpha=0)
    plt.title("Simulation of future new positive cases at Skåne. \n Simulation starting from {}".format(timeaxis[-lookback]))
    plt.xticks(np.arange(0, len(If), step=50), rotation=20)
    plt.legend()
    plt.grid(True)
    plt.show()

    Rf_max[:-lookback] = np.nan
    Rf_min[:-lookback] = np.nan
    Rf[:-lookback] = np.nan

    x = np.linspace(0, len(Rf_min), len(Rf_min))
    timeaxis = get_timeaxis(2020, 3, 9, len(Rf))
    Rf[:-lookback] = np.nan
    plt.plot(timeaxis,Rf, label='Simulated')
    plt.plot(timeaxis[:len(Rr)],Rr, color='crimson', label='Real R', alpha=.6)
    plt.plot(timeaxis,Rf_max, color="tab:blue", alpha=0)
    plt.plot(timeaxis,Rf_min, color="tab:blue", alpha=0)
    plt.plot(np.array(Rf)*0+1, 'k,')
    plt.fill_between(x, Rf_max, Rf_min, color='blue', alpha=0.2)
    plt.xticks(np.arange(0, len(Rf), step=50), rotation=20)
    plt.title("Simulation of R-number. \n Simulation starting from {}".format(timeaxis[-lookback]))
    plt.legend()
    plt.grid(True)
    plt.show()

    # today = datetime.date.today()
    # list2file(If, "Log/I_mean_{}.txt".format(today))
