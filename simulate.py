
from calculate_R import *
import numpy as np
import matplotlib.pyplot as plt
pop = 1.362e6

# R_init = R_calc[-1]
# I_future = list(I_orig)
# R_future = R_calc
# w_future = list(weights)


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


def get_distribution(Incidence, from_):

    dist = np.array(Incidence[from_:])
    I_diff = np.diff(dist)
    mu = np.mean(I_diff)
    std = np.std(I_diff)

    return (mu, std)

def simulate(days_ahead=31, decay=True, decay_coeff=1,
            random_element_coeff=1,
            diff_distribution_from=32):

    R_future, I_future, w_future = get_data()
    R_init = R_future[-1]
    mu, std = get_distribution(I_future, diff_distribution_from)


    for days in range(len(w_future), len(w_future)+days_ahead):
        tot_coeff = get_total_infected(I_future)/pop
        It = I_t(R_init, I_future, w_future, days)
        random_element = np.random.normal(mu, std)
        if It+random_element >= 0:
            It += random_element_coeff*random_element

        I_future.append(It)
        w_future.append(0)
        R_new = R_post(I_future, w_future, Lambda, days)
        R_future.append(R_init)
        if decay:
            R_init = decay_coeff*R_new*(1 - np.tanh(tot_coeff*4))
            decay_coeff/=100/10
        else:
            R_init = R_new*(1 - np.tanh(tot_coeff*4))


    return [I_future, R_future]




if __name__ == "__main__":
    I_lst = []
    R_lst = []
    days_to_simulate = 300
    with open("Incidence.txt", 'r') as file:
        I_or = file.readlines()
        I_or = [float(i) for i in I_or]
    for i in range(10):
        I, R = simulate(days_ahead=days_to_simulate, decay=False,
                    decay_coeff = 1,
                    random_element_coeff=1,
                    diff_distribution_from=32)

        I = np.array(I)
        I = moving_average(I, 7)
        I_lst.append(I)
        R_lst.append(R)

    I = np.array(I_lst)
    R = np.array(R_lst)
    I = I.mean(axis=0)
    R = R.mean(axis=0)
    perc = int(get_total_infected(I))/pop*100
    print("Om {} dagar har {} provats för covid 19 \n vilket motsvarar {}% av Skånes befolkning".format(days_to_simulate, int(get_total_infected(I)),

                                                                                               round(perc, 2)))
    fig, ax1 = plt.subplots()
    ax1.plot(I)
    ax2 = ax1.twinx()
    ax2.plot(R, color='crimson')
    ax2.plot(R*0+1, 'k,')
    plt.show()
