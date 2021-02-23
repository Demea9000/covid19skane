import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt
import datetime

x = np.linspace (0, 7, 100)
y1 = stats.gamma.pdf(x, a=4.8, scale=2.3)



def get_timeaxis(start_year, start_month, start_day, days_forward):

    out = []
    start = datetime.datetime(start_year, start_month, start_day)
    out.append(start.strftime('%Y-%m-%d'))
    for day in range(1, days_forward):
        start += datetime.timedelta(days=1)
        out.append(start.strftime('%Y-%m-%d'))

    return out


X = [[1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10 , 11, 12]]


Xs = [np.sum(i) for i in X]

print(Xs)

print(Xs.index(max(Xs)))


a = np.array([1, 2, 3, 4, 5])


b = np.pad(a, pad_width=(3, 0), mode= 'constant', constant_values=0)
print(b)

start = datetime.datetime(2020, 3, 2)

next = start + datetime.timedelta(days=1)


print(start.strftime('%Y-%m-%d'))
print(next)

dates = get_timeaxis(2020, 3, 2, 300)

print(dates[-2])

print(np.arange(0, 1000, 8))
