#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 17:44:16 2025

@author: jennatrost
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.stats import weibull_min
import math

samples = 10000


#%%

## IMPORTANT! UPDATE YOUR FILE PATH HERE

file_path = ''

#%%

## we'll start with the easier one: given capacities

given_water = pd.read_csv(f'/{file_path}/Uncertainty Analysis/Given_Water/proposed_mines_givenwater.csv')


## creating a new dataframe that holds each HUC's mean and std

given_water_stats = pd.DataFrame({
    'Proposed Site Name': given_water['Proposed Site Name'],
    'mean': given_water['wc_given (m3/yr)'],
    'min': given_water['wc_min (m3/yr)'],
    'max': given_water['wc_max (m3/yr)'],
    'std': 0    # Blank column, because we're going to calculate it
})

given_water_stats['std'] = (given_water_stats['mean'] - given_water_stats['min'])/3 ## assume min and max are 3 standard deviations from the mean

given_water_mine_list = given_water['Proposed Site Name'].astype(str).tolist()

#%%

rungivenwater = True

if rungivenwater == True:
    
    givenwater_uncertainty_values = pd.DataFrame({ ## creating a blank df to store all of our values!
        'sample number': range(1, samples+1)
        })

    ## appending mine names as column headers

    for i in given_water_mine_list:
        givenwater_uncertainty_values[given_water_mine_list] = pd.NA # storing blanks initially in the columns
    
    for i in range(0,len(given_water_mine_list)): #iterating over columns
    
        mine_name = given_water_mine_list[i]
    
        mine_index = given_water_stats[given_water_stats['Proposed Site Name'] == mine_name].index[0]
        
        mine_mean = given_water_stats.loc[mine_index, 'mean']
        
        mine_std = given_water_stats.loc[mine_index, 'std']
        
        givenwater_uncertainty_values[given_water_mine_list[i]] = abs(np.random.normal(mine_mean,mine_std, samples))
        
#%%

## okay, now let's pull out the percentiles and store them in the bigger given_capacities df

## same formula as above        

def calc_percs_water(perc_df, unc_values, minelist):
    
    perc_df['wc_p10 (m3/yr)'] = np.nan ## sets float dtype and helps avoid error later
    perc_df['wc_p50 (m3/yr)'] = np.nan ## sets float dtype and helps avoid error later
    perc_df['wc_p90 (m3/yr)'] = np.nan ## sets float dtype and helps avoid error later

    for i in range(0,len(minelist)): #iterating over columns
        
        mine_name = minelist[i]
        
        #print(mine_name)
        
        if mine_name in unc_values.columns:
        
            mine_index = perc_df[perc_df['Proposed Site Name'] == mine_name].index[0]
            
            perc_df.loc[mine_index,'wc_p10 (m3/yr)'] = float(np.percentile(unc_values[mine_name], 10))
            perc_df.loc[mine_index,'wc_p50 (m3/yr)'] = float(np.percentile(unc_values[mine_name], 50))
            perc_df.loc[mine_index,'wc_p90 (m3/yr)'] = float(np.percentile(unc_values[mine_name], 90))
            
        else:
            
            continue
        
## running formulas for given capacity and proposed mine capacity csvs

calc_percs_water(given_water, givenwater_uncertainty_values , given_water_mine_list)

## saving percentiles!

given_water = given_water.loc[:, ~given_water.columns.str.contains('^Unnamed|^$', na=False)]

given_water.to_csv(f'/{file_path}/Uncertainty Analysis/Given_Water/proposed_mines_givenwater.csv')






        















        