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

## importing CGCM A2 and B1 files

cgcm_a2 = pd.read_csv(f'/{file_path}/Uncertainty Analysis/CGCM3_A2.csv')
cgcm_b1 = pd.read_csv(f'/{file_path}/Uncertainty Analysis/CGCM3_B1.csv')

#%%

## loading huc8 values for proposed mines

proposed_mines_huc8 = pd.read_csv(f'/{file_path}/Uncertainty Analysis/proposed_mines_huc8.csv')

## pulling out unique huc8s

unique_huc8_ids = proposed_mines_huc8['HUC8 ID'].unique() # not a dataframe
unique_huc8_ids_df = pd.DataFrame({'HUC8 ID': unique_huc8_ids}) # now a dataframe

#%%

## filtering CGCM files to only contain unique_huc8_ids

cgcm_a2_filtered = cgcm_a2[cgcm_a2['CELL'].isin(unique_huc8_ids)]
cgcm_b1_filtered = cgcm_b1[cgcm_b1['CELL'].isin(unique_huc8_ids)]

#%%

## creating a new dataframe that holds each HUC's mean and std

huc8_stats_a2 = pd.DataFrame({
    'HUC8 ID': unique_huc8_ids_df['HUC8 ID'],
    'mean': 0,  # Blank column
    'std': 0    # Blank column
})

huc8_stats_b1 = pd.DataFrame({
    'HUC8 ID': unique_huc8_ids_df['HUC8 ID'],
    'mean': 0,  # Blank column
    'std': 0    # Blank column
})

## zeroes for now, will update with below

#%%

## writing functions to pull mean and stds

def calc_stats(climatemodel,hucs):
    
    for i in range(0,len(hucs['HUC8 ID'])):
        
        huc = hucs['HUC8 ID'][i]
        
        climatemodel_reduced = climatemodel[climatemodel['CELL'] == huc]
        
        hucmean = float(climatemodel_reduced['TOTSUP_MGD'].mean())
        
        hucs.loc[i, 'mean'] = float(hucmean) ## useful for normaland triangular
        hucs.loc[i, 'std'] = float(climatemodel_reduced['TOTSUP_MGD'].std()) ## useful for normal 
        
        hucs.loc[i, 'max']  = float(climatemodel_reduced['TOTSUP_MGD'].max()) ## useful for triangular 
        hucs.loc[i, 'min'] = float(climatemodel_reduced['TOTSUP_MGD'].min()) ## useful for triangular
        
        shape, loc, scale = weibull_min.fit(climatemodel_reduced['TOTSUP_MGD'], floc=0)  ## finding shape of weibull

        hucs.loc[i, 'a'] = float(shape) ## useful for weibull
        hucs.loc[i, 'k'] = float(scale) ## useful for weibull
        
#%%
## calculating stats

calc_stats(cgcm_a2_filtered, huc8_stats_a2)
calc_stats(cgcm_b1_filtered, huc8_stats_b1)

#%%

## creating monte carlo

huc8_list = unique_huc8_ids_df['HUC8 ID'].astype(str).tolist()  # string list of HUCs... these will be the column names in our iterated dfs

#%%

## a2

cgcm_a2_uncertainty_values_normal = pd.read_csv(f'/{file_path}/Uncertainty Analysis/cgcm_a2_uncertainty_values_normal.csv')
cgcm_a2_uncertainty_values_weibull = pd.read_csv(f'/{file_path}/Uncertainty Analysis/cgcm_a2_uncertainty_values_weibull.csv')
cgcm_a2_uncertainty_values_triangular = pd.read_csv(f'/{file_path}/Uncertainty Analysis/cgcm_a2_uncertainty_values_triangular.csv')

for i in huc8_list:
    cgcm_a2_uncertainty_values_normal[huc8_list] = pd.NA # storing blanks initially in the columns
    cgcm_a2_uncertainty_values_weibull[huc8_list] = pd.NA # storing blanks initially in the columns
    cgcm_a2_uncertainty_values_triangular[huc8_list] = pd.NA # storing blanks initially in the columns
#%%

runa2 = True

## running monte carlo

if runa2 == True:
    
    cgcm_a2_uncertainty_values_normal['sample number'] = range(1,samples+1) # since we're not starting at zero, only need to do this once
    cgcm_a2_uncertainty_values_weibull['sample number'] = range(1,samples+1) # since we're not starting at zero, only need to do this once
    cgcm_a2_uncertainty_values_triangular['sample number'] = range(1,samples+1) # since we're not starting at zero, only need to do this once
    
    for i in range(0,len(huc8_list)): #iterating over columns
        
        huc8_float = float(huc8_list[i])
    
        huc_index = huc8_stats_a2[huc8_stats_a2['HUC8 ID'] == huc8_float].index[0]
        
        huc_mean = huc8_stats_a2.loc[huc_index, 'mean']
        
        huc_std = huc8_stats_a2.loc[huc_index, 'std']
        
        huc_max = huc8_stats_a2.loc[huc_index, 'max']
        
        huc_min = huc8_stats_a2.loc[huc_index, 'min']
        
        huc_a = huc8_stats_a2.loc[huc_index, 'a']
        
        huc_k = huc8_stats_a2.loc[huc_index, 'k']
        
        ## REMEMBER THE VALUES ARE TOTAL WATER SUPPLY IN MGAL/DAY

        ## normal

        cgcm_a2_uncertainty_values_normal[huc8_list[i]] = abs(np.random.normal(huc_mean,huc_std, samples)) # absolute, because doesn't make sense to have negative water supply
        
        
        ## weibull
        
        cgcm_a2_uncertainty_values_weibull[huc8_list[i]] = (np.random.weibull(huc_a, samples)) * huc_k
        
    
        ## triangular
        
        cgcm_a2_uncertainty_values_triangular[huc8_list[i]] = (np.random.triangular(huc_min, huc_mean, huc_max, samples))
    
        
    ## saving into csvs
    
    cgcm_a2_uncertainty_values_normal.to_csv(f'/{file_path}/Uncertainty Analysis/cgcm_a2_uncertainty_values_normal.csv')
    cgcm_a2_uncertainty_values_weibull.to_csv(f'/{file_path}/Uncertainty Analysis/cgcm_a2_uncertainty_values_weibull.csv')    
    cgcm_a2_uncertainty_values_triangular.to_csv(f'/{file_path}/Uncertainty Analysis/cgcm_a2_uncertainty_values_triangular.csv')
#%%

## b1

cgcm_b1_uncertainty_values_normal = pd.read_csv(f'/{file_path}/Uncertainty Analysis/cgcm_b1_uncertainty_values_normal.csv')
cgcm_b1_uncertainty_values_weibull = pd.read_csv(f'/{file_path}/Uncertainty Analysis/cgcm_b1_uncertainty_values_weibull.csv')
cgcm_b1_uncertainty_values_triangular = pd.read_csv(f'/{file_path}/Uncertainty Analysis/cgcm_b1_uncertainty_values_triangular.csv')

for i in huc8_list:
    cgcm_b1_uncertainty_values_normal[huc8_list] = pd.NA # storing blanks initially in the columns
    cgcm_b1_uncertainty_values_weibull[huc8_list] = pd.NA # storing blanks initially in the columns
    cgcm_b1_uncertainty_values_triangular[huc8_list] = pd.NA # storing blanks initially in the columns

#%%

runb1 = True

## running monte carlo

if runb1 == True:
    
    cgcm_b1_uncertainty_values_normal['sample number'] = range(1,samples+1) # since we're not starting at zero, only need to do this once
    cgcm_b1_uncertainty_values_weibull['sample number'] = range(1,samples+1) # since we're not starting at zero, only need to do this once
    cgcm_b1_uncertainty_values_triangular['sample number'] = range(1,samples+1) # since we're not starting at zero, only need to do this once
    
    for i in range(0,len(huc8_list)): #iterating over columns
        
        huc8_float = float(huc8_list[i])
    
        huc_index = huc8_stats_b1[huc8_stats_b1['HUC8 ID'] == huc8_float].index[0]
        
        huc_mean = huc8_stats_b1.loc[huc_index, 'mean']
        
        huc_std = huc8_stats_b1.loc[huc_index, 'std']
        
        huc_max = huc8_stats_b1.loc[huc_index, 'max']
        
        huc_min = huc8_stats_b1.loc[huc_index, 'min']
        
        huc_a = huc8_stats_b1.loc[huc_index, 'a']
        
        huc_k = huc8_stats_b1.loc[huc_index, 'k']
        
        ## REMEMBER THE VALUES ARE TOTAL WATER SUPPLY IN MGAL/DAY
        

        ## normal

        cgcm_b1_uncertainty_values_normal[huc8_list[i]] = abs(np.random.normal(huc_mean,huc_std, samples)) # absolute, because doesn't make sense to have negative water supply 
        
        
    
        ## weibull
        
        cgcm_b1_uncertainty_values_weibull[huc8_list[i]] = (np.random.weibull(huc_a, samples)) * huc_k
        
    
        
        ## triangular
        
        cgcm_b1_uncertainty_values_triangular[huc8_list[i]] = (np.random.triangular(huc_min, huc_mean, huc_max, samples))
        
    
    
    ## saving into csvs
    
    cgcm_b1_uncertainty_values_normal.to_csv(f'/{file_path}/Uncertainty Analysis/cgcm_b1_uncertainty_values_normal.csv')
    cgcm_b1_uncertainty_values_weibull.to_csv(f'/{file_path}/Uncertainty Analysis/cgcm_b1_uncertainty_values_weibull.csv')    
    cgcm_b1_uncertainty_values_triangular.to_csv(f'/{file_path}/Uncertainty Analysis/cgcm_b1_uncertainty_values_triangular.csv')
    
#%%

## pulling out 10, 50, and 90 percentiles  
   
perc_cgcm_a2_normal = pd.DataFrame({'HUC8 ID': unique_huc8_ids_df['HUC8 ID']})
perc_cgcm_b1_normal = pd.DataFrame({'HUC8 ID': unique_huc8_ids_df['HUC8 ID']})

perc_cgcm_a2_weibull = pd.DataFrame({'HUC8 ID': unique_huc8_ids_df['HUC8 ID']})
perc_cgcm_b1_weibull = pd.DataFrame({'HUC8 ID': unique_huc8_ids_df['HUC8 ID']})

perc_cgcm_a2_triangular = pd.DataFrame({'HUC8 ID': unique_huc8_ids_df['HUC8 ID']})
perc_cgcm_b1_triangular = pd.DataFrame({'HUC8 ID': unique_huc8_ids_df['HUC8 ID']})

#%%
def calc_percs(perc_df, unc_values, huclist):
    
    perc_df['p10'] = np.nan ## sets float dtype and helps avoid error later
    perc_df['p50'] = np.nan ## sets float dtype and helps avoid error later
    perc_df['p90'] = np.nan ## sets float dtype and helps avoid error later

    for i in range(0,len(huclist)): #iterating over columns
        
        huc8_float = int(huclist[i])
        
        huc_index = perc_df[perc_df['HUC8 ID'] == huc8_float].index[0]
        
        perc_df.loc[i,'p10'] = float(np.percentile(unc_values[huclist[i]], 10))
        perc_df.loc[i,'p50'] = float(np.percentile(unc_values[huclist[i]], 50))
        perc_df.loc[i,'p90'] = float(np.percentile(unc_values[huclist[i]], 90))
        
 #%%   
calc_percs(perc_cgcm_a2_normal , cgcm_a2_uncertainty_values_normal, huc8_list)
calc_percs(perc_cgcm_b1_normal , cgcm_b1_uncertainty_values_normal, huc8_list)

calc_percs(perc_cgcm_a2_weibull , cgcm_a2_uncertainty_values_weibull, huc8_list)
calc_percs(perc_cgcm_b1_weibull , cgcm_b1_uncertainty_values_weibull, huc8_list)

calc_percs(perc_cgcm_a2_triangular , cgcm_a2_uncertainty_values_triangular, huc8_list)
calc_percs(perc_cgcm_b1_triangular , cgcm_b1_uncertainty_values_triangular, huc8_list)

## saving percs into csvs

perc_cgcm_a2_normal.to_csv(f'/{file_path}/Uncertainty Analysis/perc_cgcm_a2_normal.csv')
perc_cgcm_b1_normal.to_csv(f'/{file_path}/Uncertainty Analysis/perc_cgcm_b1_normal.csv')

perc_cgcm_a2_weibull.to_csv(f'/{file_path}/Uncertainty Analysis/perc_cgcm_a2_weibull.csv')
perc_cgcm_b1_weibull.to_csv(f'/{file_path}/Uncertainty Analysis/perc_cgcm_b1_weibull.csv')

perc_cgcm_a2_triangular.to_csv(f'/{file_path}/Uncertainty Analysis/perc_cgcm_a2_triangular.csv')
perc_cgcm_b1_triangular.to_csv(f'/{file_path}/Uncertainty Analysis/perc_cgcm_b1_triangular.csv')


#%%

## new uncertainty section! time to model uncertainty in estimating capacities

proposed_mines_capacities = pd.read_csv(f'/{file_path}/Uncertainty Analysis/proposed_mines_capacities.csv')

proposed_mines_list = proposed_mines_capacities['Proposed Site Name'].astype(str).tolist()

#%%

## we'll start with the easier one: given capacities

given_capacities = pd.read_csv(f'/{file_path}/Uncertainty Analysis/proposed_mines_givencapacities.csv')


## creating a new dataframe that holds each HUC's mean and std

given_capacities_stats = pd.DataFrame({
    'Proposed Site Name': given_capacities['Proposed Site Name'],
    'mean': given_capacities['ac_given (tons/yr)'],
    'min': given_capacities['ac_min (tons/yr)'],
    'max': given_capacities['ac_max (tons/yr)'],
    'std': 0    # Blank column, because we're going to calculate it
})

given_capacities_stats['std'] = (given_capacities_stats['mean'] - given_capacities_stats['min'])/3 ## assume min and max are 3 standard deviations from the mean

given_capacities_mine_list = given_capacities['Proposed Site Name'].astype(str).tolist()

#%%

rungivencapacities = True

if rungivencapacities == True:
    
    givencapacity_uncertainty_values = pd.DataFrame({ ## creating a blank df to store all of our values!
        'sample number': range(1, samples+1)
        })

    ## appending mine names as column headers

    for i in given_capacities_mine_list:
        givencapacity_uncertainty_values[given_capacities_mine_list] = pd.NA # storing blanks initially in the columns
    
    for i in range(0,len(given_capacities_mine_list)): #iterating over columns
    
        mine_name = given_capacities_mine_list[i]
    
        mine_index = given_capacities_stats[given_capacities_stats['Proposed Site Name'] == mine_name].index[0]
        
        mine_mean = given_capacities_stats.loc[mine_index, 'mean']
        
        mine_std = given_capacities_stats.loc[mine_index, 'std']
        
        givencapacity_uncertainty_values[given_capacities_mine_list[i]] = abs(np.random.normal(mine_mean,mine_std, samples))
        


## okay, now let's pull out the percentiles and store them in the bigger given_capacities df

## same formula as above        

def calc_percs_caps(perc_df, unc_values, minelist):
    
    perc_df['ac_p10 (tons/yr)'] = np.nan ## sets float dtype and helps avoid error later
    perc_df['ac_p50 (tons/yr)'] = np.nan ## sets float dtype and helps avoid error later
    perc_df['ac_p90 (tons/yr)'] = np.nan ## sets float dtype and helps avoid error later

    for i in range(0,len(minelist)): #iterating over columns
        
        mine_name = minelist[i]
        
        #print(mine_name)
        
        if mine_name in unc_values.columns:
        
            mine_index = perc_df[perc_df['Proposed Site Name'] == mine_name].index[0]
            
            perc_df.loc[mine_index,'ac_p10 (tons/yr)'] = float(np.percentile(unc_values[mine_name], 10))
            perc_df.loc[mine_index,'ac_p50 (tons/yr)'] = float(np.percentile(unc_values[mine_name], 50))
            perc_df.loc[mine_index,'ac_p90 (tons/yr)'] = float(np.percentile(unc_values[mine_name], 90))
            
        else:
            
            continue
        
## running formulas for given capacity and proposed mine capacity csvs

calc_percs_caps(given_capacities, givencapacity_uncertainty_values, given_capacities_mine_list)

## saving percentiles!

given_capacities = given_capacities.loc[:, ~given_capacities.columns.str.contains('^Unnamed|^$', na=False)]

given_capacities.to_csv(f'/{file_path}/Uncertainty Analysis/proposed_mines_givencapacities.csv')


#%%

## now time for calculated capacities

calculated_capacities = pd.read_csv(f'/{file_path}/Uncertainty Analysis/proposed_mines_calculatedcapacities.csv')

calculated_mines_list = calculated_capacities['Proposed Site Name'].astype(str).tolist()

#%%

## creating a new class that will each regression variable's mean, standard error, and samples

class regression_stats:
    def __init__(self, intercept, slope, intercept_SE, slope_SE):
        self.intercept = intercept
        self.slope = slope
        self.intercept_SE = intercept_SE
        self.slope_SE = slope_SE
        

r2_stats = regression_stats(1.687, 0.923, 0.981, 0.073) # resource 2 regression
capacity_stats = regression_stats(4.439, 0.374, 0.642, 0.042) # capacity

## creating blank df for regression monte carlo

calculatedcapacity_uncertainty_values = pd.DataFrame({ ## creating a blank df to store all of our values!
    'sample number': range(1, samples+1)
    })

#%%

runregression = False

if runregression == True:
    
    calculatedcapacity_uncertainty_values['r2 intercept'] = np.random.normal(r2_stats.intercept,r2_stats.intercept_SE, samples)
    calculatedcapacity_uncertainty_values['r2 slope'] = np.random.normal(r2_stats.slope,r2_stats.slope_SE, samples)
    
    calculatedcapacity_uncertainty_values['capacity intercept'] = np.random.normal(capacity_stats.intercept,capacity_stats.intercept_SE, samples)
    calculatedcapacity_uncertainty_values['capacity slope'] = np.random.normal(capacity_stats.slope,capacity_stats.slope_SE, samples)

## adding mine columns

for mine in calculated_mines_list:
    calculatedcapacity_uncertainty_values[mine] = np.nan


# ## let's estimate capacity!

# for mine in calculated_mines_list:
    
#     mine_index = calculated_capacities[calculated_capacities['Proposed Site Name'] == mine].index[0]
    
#     r1 = calculated_capacities.loc[mine_index, 'resource_1']*1e6 # tons LCE
    
#     r2_estimate = np.exp(calculatedcapacity_uncertainty_values['r2 intercept'] + calculatedcapacity_uncertainty_values['r2 slope']*np.log(r1)) # tons LCE
    
#     calculatedcapacity_uncertainty_values[mine] = np.exp(calculatedcapacity_uncertainty_values['capacity intercept'] + calculatedcapacity_uncertainty_values['capacity slope']*np.log(r2_estimate))


## let's estimate capacity (hopefully with less error propagation!) jk still propogating

for mine in calculated_mines_list:
    
    mine_index = calculated_capacities[calculated_capacities['Proposed Site Name'] == mine].index[0]
    
    r1 = calculated_capacities.loc[mine_index, 'resource_1']*1e6 # tons LCE
    
    a = calculatedcapacity_uncertainty_values['capacity intercept']
    
    b = calculatedcapacity_uncertainty_values['capacity slope']
    
    c = calculatedcapacity_uncertainty_values['r2 intercept']
    
    d = calculatedcapacity_uncertainty_values['r2 slope']
    
    calculatedcapacity_uncertainty_values[mine] = np.exp(calculatedcapacity_uncertainty_values['capacity intercept'] + (calculatedcapacity_uncertainty_values['capacity slope']*calculatedcapacity_uncertainty_values['r2 intercept'])+(calculatedcapacity_uncertainty_values['capacity slope']*calculatedcapacity_uncertainty_values['r2 slope']*np.log(r1)))


#%%
## pulling capacities

calc_percs_caps(calculated_capacities, calculatedcapacity_uncertainty_values, calculated_mines_list)

## saving percentiles

calculated_capacities = calculated_capacities.loc[:, ~calculated_capacities.columns.str.contains('^Unnamed|^$', na=False)]

calculated_capacities.to_csv(f'/{file_path}/Uncertainty Analysis/proposed_mines_calculatedcapacities.csv')


#%%

for mine in proposed_mines_list:
    
    mine_name = mine
    
    
    if mine_name in given_capacities.values: ## given capacity
        
        mine_index = given_capacities[given_capacities['Proposed Site Name'] == mine_name].index[0]
        
        proposed_index = proposed_mines_capacities[proposed_mines_capacities['Proposed Site Name'] == mine_name].index[0]
        
        proposed_mines_capacities.loc[proposed_index, 'ac_p10 (tons/yr)'] = given_capacities.loc[mine_index, 'ac_p10 (tons/yr)']
        proposed_mines_capacities.loc[proposed_index, 'ac_p50 (tons/yr)'] = given_capacities.loc[mine_index, 'ac_p50 (tons/yr)']
        proposed_mines_capacities.loc[proposed_index, 'ac_p90 (tons/yr)'] = given_capacities.loc[mine_index, 'ac_p90 (tons/yr)']
        
    else: ## calculated capacity
        
        mine_index = calculated_capacities[calculated_capacities['Proposed Site Name'] == mine_name].index[0]
        
        proposed_index = proposed_mines_capacities[proposed_mines_capacities['Proposed Site Name'] == mine_name].index[0]
        
        proposed_mines_capacities.loc[proposed_index, 'ac_p10 (tons/yr)'] = calculated_capacities.loc[mine_index, 'ac_p10 (tons/yr)']
        proposed_mines_capacities.loc[proposed_index, 'ac_p50 (tons/yr)'] = calculated_capacities.loc[mine_index, 'ac_p50 (tons/yr)']
        proposed_mines_capacities.loc[proposed_index, 'ac_p90 (tons/yr)'] = calculated_capacities.loc[mine_index, 'ac_p90 (tons/yr)']
        

## saving percentiles!        

proposed_mines_capacities.to_csv(f'/{file_path}/Uncertainty Analysis/proposed_mines_capacities.csv')

#%%

## triangle distributions for deposit type and water consumption

class triangular:
    def __init__(self, low, mean, high,c = None, p10 = None, p50 = None, p90 = None):
        self.low = low
        self.mean = mean
        self.high = high
        self.c = c # (high - mean) / (high - low), Calculate shape parameter c for scipy's triang (normalized location of mode)
        self.p10 = p10
        self.p50 = p50
        self.p90 = p90
        
## creating variables to store water LCI information, starting with blanks. Have to put a 1 to avoid division by zero

brineevap = triangular(0, 0, 1)
brinedle = triangular(0, 0, 1)
hr = triangular(0, 0, 1)
csc = triangular(0, 0, 1)
ofb = triangular(0, 0, 1)

## an array to iterate over

deposits = [brineevap, brinedle, hr, csc, ofb]

## loading water lci data

water_estimates_lit_stats = pd.read_csv(f'/{file_path}/Uncertainty Analysis/water_estimates_lit_stats.csv')

for i in range(0,len(deposits)): # for each deposit

    deposits[i].mean = water_estimates_lit_stats.loc[i,'Average (kg/kg LCE)']
    deposits[i].low = water_estimates_lit_stats.loc[i,'Minimum (kg/kg LCE)']
    deposits[i].high = water_estimates_lit_stats.loc[i,'Maximum (kg/kg LCE)']
    
    deposits[i].c = (deposits[i].high - deposits[i].mean) / (deposits[i].high - deposits[i].low) # this is for scipy, not needed in np.random.triangular
    
water_estimates_uncertainty_values = pd.read_csv(f'/{file_path}/Uncertainty Analysis/water_estimates_uncertainty_values.csv') 

runwaterlci = False

if runwaterlci == True:
    
    water_estimates_uncertainty_values['brine evap (m3/t LCE)'] = np.random.triangular(brineevap.low, brineevap.mean, brineevap.high, samples)
    water_estimates_uncertainty_values['brine dle (m3/t LCE)'] = np.random.triangular(brinedle.low, brinedle.mean, brinedle.high, samples)
    water_estimates_uncertainty_values['hr (m3/t LCE)'] = np.random.triangular(brineevap.low, brineevap.mean, brineevap.high, samples)
    water_estimates_uncertainty_values['csc (m3/t LCE)'] = np.random.triangular(csc.low, csc.mean, csc.high, samples)
    water_estimates_uncertainty_values['ofb (m3/t LCE)'] = np.random.triangular(ofb.low, ofb.mean, ofb.high, samples)



water_estimates_uncertainty_values = water_estimates_uncertainty_values.loc[:, ~water_estimates_uncertainty_values.columns.str.contains('^Unnamed|^$', na=False)]

water_estimates_uncertainty_values.to_csv(f'/{file_path}/Uncertainty Analysis/water_estimates_uncertainty_values.csv')     

#%%

## need to pull out p10, p50, and p90 for water estimates

## get column names of deposit estimates

deposit_list = [col for col in water_estimates_uncertainty_values.columns if col != 'sample number']

percs_water_estimates_lit = pd.DataFrame({'Deposit': deposit_list})

## making a new function... chances are I could have coded this better but oh well!

for i in range(0,len(deposit_list)):
        
    percs_water_estimates_lit.loc[i,'p10'] = float(np.percentile(water_estimates_uncertainty_values[deposit_list[i]], 10))
    percs_water_estimates_lit.loc[i,'p50'] = float(np.percentile(water_estimates_uncertainty_values[deposit_list[i]], 50))
    percs_water_estimates_lit.loc[i,'p90'] = float(np.percentile(water_estimates_uncertainty_values[deposit_list[i]], 90))


## saving to csv

percs_water_estimates_lit.to_csv(f'/{file_path}/Uncertainty Analysis/percs_water_estimates_lit.csv')

#%%

## now time for water demand uncertainty

## aq: aquaculture,  dp: domestic, i: industrial, ir: irrigation , ls: livestock, th: thermoelectric

## rcp 4.5

aq_ssp1 = pd.read_csv(f'/{file_path}/Uncertainty Analysis/aq_ssp1_HUC8.csv')
dp_ssp1 = pd.read_csv(f'/{file_path}/Uncertainty Analysis/dp_ssp1_HUC8.csv')
i_ssp1 = pd.read_csv(f'/{file_path}/Uncertainty Analysis/i_ssp1_HUC8.csv')
ir_ssp1 = pd.read_csv(f'/{file_path}/Uncertainty Analysis/ir_ssp1_HUC8.csv')
ls_ssp1 = pd.read_csv(f'/{file_path}/Uncertainty Analysis/ls_ssp1_HUC8.csv')
th_ssp1 = pd.read_csv(f'/{file_path}/Uncertainty Analysis/th_ssp1_HUC8.csv')


## rcp 8.5

aq_ssp5 = pd.read_csv(f'/{file_path}/Uncertainty Analysis/aq_ssp5_HUC8.csv')
dp_ssp5 = pd.read_csv(f'/{file_path}/Uncertainty Analysis/dp_ssp5_HUC8.csv')
i_ssp5 = pd.read_csv(f'/{file_path}/Uncertainty Analysis/i_ssp5_HUC8.csv')
ir_ssp5 = pd.read_csv(f'/{file_path}/Uncertainty Analysis/ir_ssp5_HUC8.csv')
ls_ssp5 = pd.read_csv(f'/{file_path}/Uncertainty Analysis/ls_ssp5_HUC8.csv')
th_ssp5 = pd.read_csv(f'/{file_path}/Uncertainty Analysis/th_ssp5_HUC8.csv')


## creating a function to make dfs for each water demand type and climate scenario

def water_demand_df(unique_hucs):
    
    return pd.DataFrame({
        'HUC8 ID': unique_hucs['HUC8 ID'],
        'mean': 0,  # Blank column
        'min': 0,    # Blank column
        'max': 0
    })

aq_ssp1_stats = water_demand_df(unique_huc8_ids_df)
dp_ssp1_stats = water_demand_df(unique_huc8_ids_df)
i_ssp1_stats = water_demand_df(unique_huc8_ids_df)
ir_ssp1_stats = water_demand_df(unique_huc8_ids_df)
ls_ssp1_stats = water_demand_df(unique_huc8_ids_df)
th_ssp1_stats = water_demand_df(unique_huc8_ids_df)

aq_ssp5_stats = water_demand_df(unique_huc8_ids_df)
dp_ssp5_stats = water_demand_df(unique_huc8_ids_df)
i_ssp5_stats = water_demand_df(unique_huc8_ids_df)
ir_ssp5_stats = water_demand_df(unique_huc8_ids_df)
ls_ssp5_stats = water_demand_df(unique_huc8_ids_df)
th_ssp5_stats = water_demand_df(unique_huc8_ids_df)

rcp45_stats = [aq_ssp1_stats, dp_ssp1_stats, i_ssp1_stats, ir_ssp1_stats, ls_ssp1_stats, th_ssp1_stats]
rcp85_stats = [aq_ssp5_stats, dp_ssp5_stats, i_ssp5_stats, ir_ssp5_stats, ls_ssp5_stats, th_ssp5_stats]

#%%

## writing function to filter down df to just 2040 - 2060

def filter_years(df, minyear, maxyear):
    
    years = [f'Y{year}' for year in range(minyear, maxyear + 1)]
    
    df_filtered = df[['huc8'] + years]
    
    return df_filtered


## writing functions to pull mean and stds

def calc_water_demand(waterdemand,hucs):
    
    for i in range(0,len(hucs['HUC8 ID'])):
        
        huc = hucs['HUC8 ID'][i]
        
        waterdemand_reduced = waterdemand[waterdemand['huc8'] == huc]
        
        ## isolating values that are not huc 8
        
        yearvalues = waterdemand_reduced.iloc[0, 1:].astype(float)
        
        hucs.loc[i, 'mean'] = float(yearvalues.mean()) ## useful for normal and triangular
        hucs.loc[i, 'min'] = float(yearvalues.min()) ## useful for normal and triangular
        hucs.loc[i, 'max'] = float(yearvalues.max()) ## useful for normal and triangular
        
        hucs.loc[i, 'std'] = float(yearvalues.std()) ## useful for normal 
        
        shape, loc, scale = weibull_min.fit(yearvalues, floc=0)  ## finding shape of weibull

        hucs.loc[i, 'a'] = float(shape) ## useful for weibull
        hucs.loc[i, 'k'] = float(scale) ## useful for weibull


## filtering datasets

aq_ssp1_filtered = filter_years(aq_ssp1, 2040, 2060)
dp_ssp1_filtered = filter_years(dp_ssp1, 2040, 2060)
i_ssp1_filtered = filter_years(i_ssp1, 2040, 2060)
ir_ssp1_filtered = filter_years(ir_ssp1, 2040, 2060)
ls_ssp1_filtered = filter_years(ls_ssp1, 2040, 2060)
th_ssp1_filtered = filter_years(th_ssp1, 2040, 2060)

aq_ssp5_filtered = filter_years(aq_ssp5, 2040, 2060)
dp_ssp5_filtered = filter_years(dp_ssp5, 2040, 2060)
i_ssp5_filtered = filter_years(i_ssp5, 2040, 2060)
ir_ssp5_filtered = filter_years(ir_ssp5, 2040, 2060)
ls_ssp5_filtered = filter_years(ls_ssp5, 2040, 2060)
th_ssp5_filtered = filter_years(th_ssp5, 2040, 2060)

## creating iterable arrays

rcp45 = [aq_ssp1_filtered, dp_ssp1_filtered, i_ssp1_filtered, ir_ssp1_filtered, ls_ssp1_filtered, th_ssp1_filtered]
rcp85 = [aq_ssp5_filtered, dp_ssp5_filtered, i_ssp5_filtered, ir_ssp5_filtered, ls_ssp5_filtered, th_ssp5_filtered]

## calculating stats, rcp45 and rcp85 arrays should be the same lengths!

for i in range(0,len(rcp45)):
    
    calc_water_demand(rcp45[i], rcp45_stats[i])
    calc_water_demand(rcp85[i], rcp85_stats[i])



#%%

## creating dfs to store uncertainty values

aq_ssp1_uncertainty_values = pd.DataFrame({'sample number': range(1,samples+1)})
aq_ssp1_uncertainty_values[huc8_list] = pd.NA # storing blanks initially in the columns

dp_ssp1_uncertainty_values = pd.DataFrame({'sample number': range(1,samples+1)})
dp_ssp1_uncertainty_values[huc8_list] = pd.NA # storing blanks initially in the columns

i_ssp1_uncertainty_values = pd.DataFrame({'sample number': range(1,samples+1)})
i_ssp1_uncertainty_values[huc8_list] = pd.NA # storing blanks initially in the columns

ir_ssp1_uncertainty_values = pd.DataFrame({'sample number': range(1,samples+1)})
ir_ssp1_uncertainty_values[huc8_list] = pd.NA # storing blanks initially in the columns

ls_ssp1_uncertainty_values = pd.DataFrame({'sample number': range(1,samples+1)})
ls_ssp1_uncertainty_values[huc8_list] = pd.NA # storing blanks initially in the columns

th_ssp1_uncertainty_values = pd.DataFrame({'sample number': range(1,samples+1)})
th_ssp1_uncertainty_values[huc8_list] = pd.NA # storing blanks initially in the columns

aq_ssp5_uncertainty_values = pd.DataFrame({'sample number': range(1,samples+1)})
aq_ssp5_uncertainty_values[huc8_list] = pd.NA # storing blanks initially in the columns

dp_ssp5_uncertainty_values = pd.DataFrame({'sample number': range(1,samples+1)})
dp_ssp5_uncertainty_values[huc8_list] = pd.NA # storing blanks initially in the columns

i_ssp5_uncertainty_values = pd.DataFrame({'sample number': range(1,samples+1)})
i_ssp5_uncertainty_values[huc8_list] = pd.NA # storing blanks initially in the columns

ir_ssp5_uncertainty_values = pd.DataFrame({'sample number': range(1,samples+1)})
ir_ssp5_uncertainty_values[huc8_list] = pd.NA # storing blanks initially in the columns

ls_ssp5_uncertainty_values = pd.DataFrame({'sample number': range(1,samples+1)})
ls_ssp5_uncertainty_values[huc8_list] = pd.NA # storing blanks initially in the columns

th_ssp5_uncertainty_values = pd.DataFrame({'sample number': range(1,samples+1)})
th_ssp5_uncertainty_values[huc8_list] = pd.NA # storing blanks initially in the columns


## putting into iterable arrays

rcp45_uncertainty_values = [aq_ssp1_uncertainty_values, dp_ssp1_uncertainty_values, i_ssp1_uncertainty_values, ir_ssp1_uncertainty_values, ls_ssp1_uncertainty_values, th_ssp1_uncertainty_values]
rcp85_uncertainty_values = [aq_ssp5_uncertainty_values, dp_ssp5_uncertainty_values, i_ssp5_uncertainty_values, ir_ssp5_uncertainty_values, ls_ssp5_uncertainty_values, th_ssp5_uncertainty_values]

#%%

runwaterdemand = True

if runwaterdemand == True:
    
    ## rcp 4.5
    
    for i in range(0,len(rcp45_uncertainty_values)):
        
        water_df = rcp45_uncertainty_values[i]
        
        for j in range(0,len(huc8_list)): #iterating over columns
        
            stats_df = rcp45_stats[i]
            
            huc8_float = float(huc8_list[j])
        
            huc_index = stats_df[stats_df['HUC8 ID'] == huc8_float].index[0]
            
            huc_mean = float(stats_df.loc[huc_index, 'mean'])
            
            huc_std = float(stats_df.loc[huc_index, 'std'])
            
            huc_max = float(stats_df.loc[huc_index, 'max'])
            
            huc_min = float(stats_df.loc[huc_index, 'min'])
            
            huc_a = float(stats_df.loc[huc_index, 'a'])
            
            huc_k = float(stats_df.loc[huc_index, 'k'])
            
            
            ## triangular
            
            if huc_min < huc_mean: ## this is to get rid of the cases where water demand is zero
            
                water_df[huc8_list[j]] = (np.random.triangular(huc_min, huc_mean, huc_max, samples))
            
            else:
                
                water_df[huc8_list[j]] = huc_mean
    
    
    aq_ssp1_uncertainty_values.to_csv(f'/{file_path}/Uncertainty Analysis/aq_ssp1_uncertainty_values.csv')
    dp_ssp1_uncertainty_values.to_csv(f'/{file_path}/Uncertainty Analysis/dp_ssp1_uncertainty_values.csv')
    i_ssp1_uncertainty_values.to_csv(f'/{file_path}/Uncertainty Analysis/i_ssp1_uncertainty_values.csv')
    ir_ssp1_uncertainty_values.to_csv(f'/{file_path}/Uncertainty Analysis/ir_ssp1_uncertainty_values.csv')
    ls_ssp1_uncertainty_values.to_csv(f'/{file_path}/Uncertainty Analysis/ls_ssp1_uncertainty_values.csv')
    th_ssp1_uncertainty_values.to_csv(f'/{file_path}/Uncertainty Analysis/th_ssp1_uncertainty_values.csv')
    
    
    ## rcp 8.5
    
    for i in range(0,len(rcp85_uncertainty_values)):
        
        water_df = rcp85_uncertainty_values[i]
        
        for j in range(0,len(huc8_list)): #iterating over columns
        
            stats_df = rcp85_stats[i]
            
            huc8_float = float(huc8_list[j])
        
            huc_index = stats_df[stats_df['HUC8 ID'] == huc8_float].index[0]
            
            huc_mean = float(stats_df.loc[huc_index, 'mean'])
            
            huc_std = float(stats_df.loc[huc_index, 'std'])
            
            huc_max = float(stats_df.loc[huc_index, 'max'])
            
            huc_min = float(stats_df.loc[huc_index, 'min'])
            
            huc_a = float(stats_df.loc[huc_index, 'a'])
            
            huc_k = float(stats_df.loc[huc_index, 'k'])
            
            
            ## triangular
            
            if huc_min < huc_mean: ## this is to get rid of the cases where water demand is zero
            
                water_df[huc8_list[j]] = (np.random.triangular(huc_min, huc_mean, huc_max, samples))
            
            else:
                
                water_df[huc8_list[j]] = huc_mean
    
    
    aq_ssp5_uncertainty_values.to_csv(f'/{file_path}/Uncertainty Analysis/aq_ssp5_uncertainty_values.csv')
    dp_ssp5_uncertainty_values.to_csv(f'/{file_path}/Uncertainty Analysis/dp_ssp5_uncertainty_values.csv')
    i_ssp5_uncertainty_values.to_csv(f'/{file_path}/Uncertainty Analysis/i_ssp5_uncertainty_values.csv')
    ir_ssp5_uncertainty_values.to_csv(f'/{file_path}/Uncertainty Analysis/ir_ssp5_uncertainty_values.csv')
    ls_ssp5_uncertainty_values.to_csv(f'/{file_path}/Uncertainty Analysis/ls_ssp5_uncertainty_values.csv')
    th_ssp5_uncertainty_values.to_csv(f'/{file_path}/Uncertainty Analysis/th_ssp5_uncertainty_values.csv')
    

#%%

## creating a function to make percentile dfs for each water demand type and climate scenario

def water_demand_percs(unique_hucs):
    
    return pd.DataFrame({
        'HUC8 ID': unique_hucs['HUC8 ID'],
        'p10': 0,  # Blank column
        'p50': 0,    # Blank column
        'p90': 0
    })

percs_aq_ssp1 = water_demand_percs(unique_huc8_ids_df)
percs_dp_ssp1 = water_demand_percs(unique_huc8_ids_df)
percs_i_ssp1 = water_demand_percs(unique_huc8_ids_df)
percs_ir_ssp1 = water_demand_percs(unique_huc8_ids_df)
percs_ls_ssp1 = water_demand_percs(unique_huc8_ids_df)
percs_th_ssp1 = water_demand_percs(unique_huc8_ids_df)

percs_aq_ssp5 = water_demand_percs(unique_huc8_ids_df)
percs_dp_ssp5 = water_demand_percs(unique_huc8_ids_df)
percs_i_ssp5 = water_demand_percs(unique_huc8_ids_df)
percs_ir_ssp5 = water_demand_percs(unique_huc8_ids_df)
percs_ls_ssp5 = water_demand_percs(unique_huc8_ids_df)
percs_th_ssp5 = water_demand_percs(unique_huc8_ids_df)

## now time to calculate percs

calc_percs(percs_aq_ssp1, aq_ssp1_uncertainty_values, huc8_list)
calc_percs(percs_dp_ssp1, dp_ssp1_uncertainty_values, huc8_list)
calc_percs(percs_i_ssp1, i_ssp1_uncertainty_values, huc8_list)
calc_percs(percs_ir_ssp1, ir_ssp1_uncertainty_values, huc8_list)
calc_percs(percs_ls_ssp1, ls_ssp1_uncertainty_values, huc8_list)
calc_percs(percs_th_ssp1, th_ssp1_uncertainty_values, huc8_list)

calc_percs(percs_aq_ssp5, aq_ssp5_uncertainty_values, huc8_list)
calc_percs(percs_dp_ssp5, dp_ssp5_uncertainty_values, huc8_list)
calc_percs(percs_i_ssp5, i_ssp5_uncertainty_values, huc8_list)
calc_percs(percs_ir_ssp5, ir_ssp5_uncertainty_values, huc8_list)
calc_percs(percs_ls_ssp5, ls_ssp5_uncertainty_values, huc8_list)
calc_percs(percs_th_ssp5, th_ssp5_uncertainty_values, huc8_list)


## saving to csvs


percs_aq_ssp1.to_csv(f'/{file_path}/Uncertainty Analysis/percs_aq_ssp1.csv')
percs_dp_ssp1.to_csv(f'/{file_path}/Uncertainty Analysis/percs_dp_ssp1.csv')
percs_i_ssp1.to_csv(f'/{file_path}/Uncertainty Analysis/percs_i_ssp1.csv')
percs_ir_ssp1.to_csv(f'/{file_path}/Uncertainty Analysis/percs_ir_ssp1.csv')
percs_ls_ssp1.to_csv(f'/{file_path}/Uncertainty Analysis/percs_ls_ssp1.csv')
percs_th_ssp1.to_csv(f'/{file_path}/Uncertainty Analysis/percs_th_ssp1.csv')

percs_aq_ssp5.to_csv(f'/{file_path}/Uncertainty Analysis/percs_aq_ssp5.csv')
percs_dp_ssp5.to_csv(f'/{file_path}/Uncertainty Analysis/percs_dp_ssp5.csv')
percs_i_ssp5.to_csv(f'/{file_path}/Uncertainty Analysis/percs_i_ssp5.csv')
percs_ir_ssp5.to_csv(f'/{file_path}/Uncertainty Analysis/percs_ir_ssp5.csv')
percs_ls_ssp5.to_csv(f'/{file_path}/Uncertainty Analysis/percs_ls_ssp5.csv')
percs_th_ssp5.to_csv(f'/{file_path}/Uncertainty Analysis/percs_th_ssp5.csv')


#%%

## oops forgot to do unceratinty with given water consumption

## we'll start with the easier one: given capacities

given_water = pd.read_csv(f'/{file_path}/Uncertainty Analysis/proposed_mines_givenwater.csv')


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

if rungivencapacities == True:
    
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

given_water.to_csv(f'/{file_path}/Uncertainty Analysis/proposed_mines_givenwater.csv')






        















        