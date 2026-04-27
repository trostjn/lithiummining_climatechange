#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  5 22:40:00 2025

@author: jennatrost
"""

################ packages

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## lithium

lce = 5.323
li2o = 2.153
lhe = 3.448
spodumeneconc_kingsmountain = .072 # %, Li2O, https://www.albemarle.com/cl/en/what-we-offer/featured-products/spodumene#:~:text=Albemarle's%20technical%20grade%20spodumene%20concentrate,has%20a%20low%20impurity%20profile.

## mass

ton = 1
Mton = 1e-6