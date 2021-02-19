
from calculate_R import *
import numpy as np
import matplotlib.pyplot as plt

pop = 1.362e6

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



def moving_average(data_set, periods=3):
    weights = np.ones(periods) / periods
    return np.convolve(data_set, weights, mode='valid')


def get_distribution(Incidence, from_=0):

    dist = np.array(Incidence[from_:])
    I_diff = np.diff(dist)
    mu = np.mean(I_diff)
    std = np.std(I_diff)

    return (mu, std)

def simulate(days_ahead=200):

    R_future, I_future, w_future = get_data()
    mu, std = get_distribution(I_future)

    for days in range(len(w_future), len(w_future)+days_ahead):
        tot_coeff = get_total_infected(I_future)/pop
        It = lambda_t(R_future, I_future, w_future, days)
        I_future.append(It)
        w_future.append(0)
        R_new = R_post(I_future, w_future, Lambda, days)*(1-np.tanh(tot_coeff))
        R_future.append(R_new)


    return [I_future, R_future]



def compare_backward(start, stop, days_ahead=21):

    R_future, I_future, w_future = get_data()
    R_real, I_real, w_real = get_data()

    R_future = R_future[start:stop-days_ahead]
    I_future = I_future[start:stop-days_ahead]
    w_future = w_future[start:stop-days_ahead]
    R_real = R_real[start:stop]
    I_real=I_real[start:stop]
    w_real=w_real[start:stop]

    mu, std = get_distribution(I_future)



    for days in range(len(w_future), len(w_future)+days_ahead):
        tot_coeff = get_total_infected(I_future)/pop
        It = lambda_t(R_future, I_future, w_future, days)
        random_element = np.random.normal(mu, std)
        if It+random_element >= 0:
            It += 0.4*random_element
        I_future.append(It)
        w_future.append(0)
        R_new = R_post(I_future, w_future, Lambda, days)*(1-np.tanh(tot_coeff))
        R_future.append(R_new)


    return [I_future, R_future, I_real, R_real]




if __name__ == "__main__":

    days_ahead = 93
    I_lst = []
    R_lst = []
    I_interval = []

    test_point = 350
    for i in range(100):
        If, Rf, Ir, Rr = compare_backward(0, test_point, days_ahead=days_ahead)

        I_lst.append(If)
        R_lst.append(Rf)
        I_interval.append(If)

    If = np.array(I_lst)
    If = If.mean(axis=0)

    Rf = np.array(R_lst)
    Rf = Rf.mean(axis=0)

    I_interval = np.array(I_interval)
    I_interval_sum = [np.sum(i) for i in I_interval]
    I_maxindex = I_interval_sum.index(max(I_interval_sum))
    I_minindex = I_interval_sum.index(min(I_interval_sum))
    If_max = I_interval[I_maxindex]
    If_min = I_interval[I_minindex]

    If_max[:-days_ahead] = np.nan
    If_min[:-days_ahead] = np.nan
    If[:-days_ahead] = np.nan


    Ir[-days_ahead+1:] = moving_average(Ir, 7)[-days_ahead:]
    x = np.linspace(0, len(If_min), len(If_min))
    plt.plot(If, label='Simulated')
    plt.plot(Ir, color='crimson', label='Real Incidence')
    plt.plot(If_max)
    plt.plot(If_min)
    plt.title("Test how prediction behaves historically")
    plt.legend()
    #plt.fill_between(x, If_max, If_min, color='blue', alpha=0.5)
    plt.show()

    plt.plot(Rf, label='Simulated')
    plt.plot(Rr, color='crimson', label='Real R')
    plt.legend()
    plt.show()

    I, R = simulate(days_ahead=days_ahead)
    plt.plot(I[300:])
    plt.title('Simulated incidence')
    plt.show()
