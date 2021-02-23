import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

R_data = pd.read_excel('Rtalet_sim.xlsx')
weights = R_data['si_distr']
w = list(weights)
cases = pd.read_excel('totalt-antal-konstaterade-covid-fall.xlsx')

I_orig = cases['Nya konstaterade personer']
tau=7

def list2file(list_=list, filename=str):
    with open(filename, 'w') as file:
        for listitem in list_:
            file.write('{}\n'.format(listitem))

def Lambda(I, w, t):
    L = 0
    for s in range(1, t):
        L += I[t-s]*w[s]
    return L

def R_post(I, w, L, t, a=1, b=5, tau=7):
    I0 = 0
    L0 = 0
    for s in range(t-tau+1, t+1):
        I0 += I[s]
        L0 += L(I, w, s)

    numerator = a + I0
    denominator = 1/b + L0

    return numerator/denominator

def calculate_R(write=False):
    R_calc = []
    for days in range(tau, len(I_orig)):
        R_calc.append(R_post(I_orig, w, Lambda, days, tau=tau))
        w.append(0)
    if write:
        list2file(R_calc, "R_calc.txt")
        list2file(I_orig, "Incidence.txt")
        list2file(weights, "weights.txt")

def plot_R():
    fig, ax1 = plt.subplots()

    ax1.plot(R_calc, label="Egen uträkning")
    ax1.plot(R, label="Original Rkod", color='crimson')

    ax1.legend(loc=1)
    plt.show()

if __name__ == "__main__":
    calculate_R(write=True)
