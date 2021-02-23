
from calculate_R import *
import numpy as np
import matplotlib.pyplot as plt
import datetime

pop = 1.362e6

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

def simulate(days_ahead=31):

    R_future, I_future, w_future = get_data()
    mu, std = get_distribution(np.diff(I_future))
    muR, stdR = get_distribution(np.diff(R_future))

    for days in range(len(w_future), len(w_future)+days_ahead):
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



def compare_backward(start, stop, days_ahead=31, tolerance=0.2):

    R_future, I_future, w_future = get_data()
    R_real, I_real, w_real = get_data()

    R_future = R_future[start:stop-days_ahead]
    I_future = I_future[start:stop-days_ahead]
    w_future = w_future[start:stop-days_ahead]
    R_real = R_real[start:stop]
    I_real=I_real[start:stop]
    w_real=w_real[start:stop]

    # R_future = list(moving_average(R_future, 7))
    # I_future = list(moving_average(I_future, 7))
    # w_future = list(moving_average(w_future, 7))

    mu, std = get_distribution(np.diff(I_future))
    muR, stdR = get_distribution(np.diff(R_real[60:]))

    #R_future[-1] = np.random.normal(muR, stdR)
    for days in range(len(w_future), len(w_future)+days_ahead):
        tot_coeff = get_total_infected(I_future)/pop
        if tot_coeff>=1:
            break
        It = lambda_t(R_future, I_future, w_future, days)
        random_element = np.random.normal(mu, std)
        if It+random_element >= 0:
            It += .7*random_element
        if It < 0:
            It = 0
        I_future.append(int(It))
        w_future.append(0)
        R_new = R_post(I_future, w_future, Lambda, days)*(1-np.tanh(tot_coeff)) + .7*np.random.normal(muR, stdR)
        if abs(R_new-R_future[-1]) < tolerance:
            R_future.append(R_new)
        else:
            R_future.append(R_future[-1])


    return [I_future, R_future, I_real, R_real]




if __name__ == "__main__":


    I_lst = []
    R_lst = []
    I_interval = []
    #np.random.seed(10982756)
    _, I, _ = get_data()
    test_point = len(I)+62
    days_ahead = 31+62
    Nsims = 10
    for i in range(Nsims):
        If, Rf, Ir, Rr = compare_backward(0, test_point, days_ahead=days_ahead)

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

    If_max[:-days_ahead] = np.nan
    If_min[If_min<0]=0
    If_min[:-days_ahead] = np.nan
    If[:-days_ahead] = np.nan


    I = moving_average(I, 7)
    x = np.linspace(0, len(If_min), len(If_min))

    timeaxis = get_timeaxis(2020, 3, 2, len(If))

    plt.plot(timeaxis, If, label='Average incidence of {} simulations'.format(Nsims))
    plt.fill_between(x, If_max, If_min, color='blue', alpha=0.2, label="Standard deviation of {} simulations".format(Nsims))
    plt.plot(timeaxis[:len(I)],I[:], color='crimson', label='Actual 7-day moving average Incidence', alpha=.7)
    plt.plot(timeaxis,If_max, color="tab:blue", alpha=0)
    plt.plot(timeaxis,If_min, color="tab:blue", alpha=0)
    plt.title("Test how prediction behaves historically")
    plt.xticks(np.arange(0, len(If), step=93), rotation=20)
    plt.legend()
    plt.grid()
    plt.show()

    x = np.linspace(0, len(Rf_min), len(Rf_min))
    timeaxis = get_timeaxis(2020, 3, 9, len(Rf))
    plt.plot(timeaxis,Rf, label='Simulated')
    plt.plot(timeaxis[:len(Rr)],Rr, color='crimson', label='Real R')
    plt.plot(timeaxis,Rf_max, color="tab:blue", alpha=0)
    plt.plot(timeaxis,Rf_min, color="tab:blue", alpha=0)
    plt.plot(np.array(Rr)*0+1, 'k,')
    plt.fill_between(x, Rf_max, Rf_min, color='blue', alpha=0.2)
    plt.xticks(np.arange(0, len(Rf), step=93), rotation=20)
    plt.legend()
    plt.show()

    # Isim = []
    # Rsim = []
    #
    # for i in range(Nsims):
    #     If, Rf = simulate(days_ahead=days_ahead)
    #
    #     Isim.append(If)
    #     Rsim.append(Rf)
    #
    # If = np.array(Isim)
    # Istd = If.std(axis=0)
    # If = If.mean(axis=0)
    # Rf = np.array(Rsim)
    # Rstd = Rf.std(axis=0)
    # Rf = Rf.mean(axis=0)
    #
    # If_max = If + Istd
    # If_min = If - Istd
    #
    # Rf_max = Rf+Rstd
    # Rf_min = Rf-Rstd
    #
    # x = np.linspace(0, len(If_min), len(If_min))
    # plt.plot(If[:])
    # plt.plot(If_max, color="tab:blue", alpha=0)
    # plt.plot(If_min, color="tab:blue", alpha=0)
    # plt.fill_between(x, If_max, If_min, color='blue', alpha=0.2, label="Spread of model")
    # plt.title('Simulated incidence')
    # plt.show()
    #
    # x = np.linspace(0, len(Rf_min), len(Rf_min))
    # plt.plot(Rf[:])
    # plt.plot(Rf[:]*0+1, 'k,')
    # plt.title('Simulated R')
    # plt.plot(Rf_max, color="tab:blue", alpha=0)
    # plt.plot(Rf_min, color="tab:blue", alpha=0)
    # plt.fill_between(x, Rf_max, Rf_min, color='blue', alpha=0.2)
    # plt.show()
    #
    # print(int(If[-1]), int(If_max[-1]+If[-1]),
    # int(If[-1] + If_min[-1]), Rf[-1], Rf_max[-1], Rf_min[-1])
