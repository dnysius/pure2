# -*- coding: utf-8 -*-
from numba import vectorize


@vectorize(['float64(float64, float64, float64)'], target='parallel')
def normalize(p, pmin, pmax):
    if p < 0:
        return p / abs(pmin)
    elif p > 0:
        return p / abs(pmax)
    else:
        return 0
    
