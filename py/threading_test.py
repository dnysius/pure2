# -*- coding: utf-8 -*-
import threading
import numpy as np
from time import perf_counter_ns


L = 10000
lend2 = 40
limit = L*lend2
xx, yy = np.meshgrid(np.arange(lend2), np.arange(L))
xx = xx.reshape((xx.size, 1))
yy = yy.reshape((yy.size, 1))
ind_arr = np.hstack((xx, yy))
post = np.zeros((lend2, L))


def main(i):
    for j in range(100):
        for k in range(100):
            a = np.exp(j+k)
#    ind = ind_arr[i]
#    post[ind[0], ind[1]] = ind[0] + ind[1]
#    return post


if __name__ == '__main__':
    start_time = perf_counter_ns()*1e-9
    jobs = []
    print("Append")
#    for i in range(L):
    for i in range(limit):
        jobs.append(threading.Thread(target=main, args=(i,)))
    print("Starting")
    for job in jobs:
        job.start()
    print("Joining")
    for job in jobs:
        job.join()
    print("Stitching")
    duration = perf_counter_ns()*1e-9-start_time
    print(duration)
