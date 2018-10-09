"""Analise results of the OEM estimation of a Van der Pol oscillator."""


import os
import re

import numpy as np


def get_popt(datadir):
    popt = []
    pattern =  r'popt-\d+.txt$'
    for f in os.listdir(datadir):
        if re.match(pattern, f):
            popt.append(np.loadtxt(os.path.join(datadir, f)))
    return np.array(popt)
