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
import seaborn as sns

samples = 10000

#%%

## IMPORTANT! UPDATE YOUR FILE PATH HERE

file_path = ''

#%%

## new uncertainty section! time to model uncertainty in estimating capacities

proposed_mines_capacities = pd.read_csv(f'/{file_path}/Uncertainty Analysis/Capacity/proposed_mines_capacities.csv')

proposed_mines_list = proposed_mines_capacities['Proposed Site Name'].astype(str).tolist()

#%%

## we'll start with the easier one: given capacities

given_capacities = pd.read_csv(f'/{file_path}/Uncertainty Analysis/Capacity/proposed_mines_givencapacities.csv')


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

given_capacities.to_csv(f'/{file_path}/Uncertainty Analysis/Capacity/proposed_mines_givencapacities.csv')


#%%

## now time for calculated capacities

calculated_capacities = pd.read_csv(f'/{file_path}/Uncertainty Analysis/Capacity/proposed_mines_calculatedcapacities.csv')

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

runregression = True

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

calculated_capacities.to_csv(f'/{file_path}/Uncertainty Analysis/Capacity/proposed_mines_calculatedcapacities.csv')


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

proposed_mines_capacities = proposed_mines_capacities.loc[:, ~proposed_mines_capacities.columns.str.contains('^Unnamed')]
   

proposed_mines_capacities.to_csv(f'/{file_path}/Uncertainty Analysis/Capacity/proposed_mines_capacities.csv')

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

water_estimates_lit_stats = pd.read_csv(f'/{file_path}/Uncertainty Analysis/Capacity/water_estimates_lit_stats.csv')

for i in range(0,len(deposits)): # for each deposit

    deposits[i].mean = water_estimates_lit_stats.loc[i,'Average (kg/kg LCE)']
    deposits[i].low = water_estimates_lit_stats.loc[i,'Minimum (kg/kg LCE)']
    deposits[i].high = water_estimates_lit_stats.loc[i,'Maximum (kg/kg LCE)']
    
    deposits[i].c = (deposits[i].high - deposits[i].mean) / (deposits[i].high - deposits[i].low) # this is for scipy, not needed in np.random.triangular
    
water_estimates_uncertainty_values = pd.read_csv(f'/{file_path}/Uncertainty Analysis/Capacity/water_estimates_uncertainty_values.csv') 

runwaterlci = True

if runwaterlci == True:
    
    water_estimates_uncertainty_values['brine evap (m3/t LCE)'] = np.random.triangular(brineevap.low, brineevap.mean, brineevap.high, samples)
    water_estimates_uncertainty_values['brine dle (m3/t LCE)'] = np.random.triangular(brinedle.low, brinedle.mean, brinedle.high, samples)
    water_estimates_uncertainty_values['hr (m3/t LCE)'] = np.random.triangular(hr.low, hr.mean, hr.high, samples)
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

percs_water_estimates_lit.to_csv(f'/{file_path}/Uncertainty Analysis/Capacity/percs_water_estimates_lit.csv')



#%%

## quantifying uncertainty

r2_stats = regression_stats(1.687, 0.923, 0.981, 0.073) # resource 2 regression
capacity_stats = regression_stats(4.439, 0.374, 0.642, 0.042) # capacity

quantifying_capacity_uncertainty_values = pd.DataFrame({ ## creating a blank df to store all of our values!
    'sample number': range(1, samples+1)
    })


#%%

## calculating capacity without uncertainty. Just using the numbers reported in Buarque Andrade et al (2024)

nonunc_calculated_capacities = calculated_capacities[['Proposed Site Name', 'resource_1']]

#%%

## calculating capacity

nonunc_calculated_capacities['r1'] = nonunc_calculated_capacities['resource_1']*1e6 # tons LCE

nonunc_calculated_capacities['r2'] = np.exp(r2_stats.intercept + (r2_stats.slope*np.log(nonunc_calculated_capacities['r1'])))

nonunc_calculated_capacities['capacity'] = np.exp(capacity_stats.intercept + (capacity_stats.slope*np.log(nonunc_calculated_capacities['r2']))) # tons/year


#%%
    
## what we need to do is find the percent difference between each uncertainty value and the "expected" value for each mine

quantifying_capacity_uncertainty_values = calculatedcapacity_uncertainty_values[['sample number', 'Basin', 'Big Sandy', 'Clayton Ridge', 'Clayton Valley (ACME)', 'Horizon', 'Lone Mountain',\
                                                                                'McGee', 'Nevada North', 'West Tonopah']]

    
#%%

percent_diff_capacity_uncertainty_values = pd.DataFrame({ ## creating a blank df to store all of our values!
    'sample number': range(1, samples+1)
    })    

for i in range(0,len(calculated_capacities)):
    
    mine_name = calculated_mines_list[i]
    
    avg_cap = nonunc_calculated_capacities.loc[nonunc_calculated_capacities['Proposed Site Name'] == mine_name, 'capacity'].values[0] ## pulling out capacity for each mine

    percent_diff_capacity_uncertainty_values[f'{mine_name}'] = ((quantifying_capacity_uncertainty_values[f'{mine_name}'] - avg_cap)/avg_cap)*100 # calculating percent change

#%%

## plotting uncertainty

nosamp = percent_diff_capacity_uncertainty_values.drop(columns=['sample number'])

plt.figure(figsize=(14, 6))

sns.boxplot(nosamp)

#%%

nonunc_calculated_capacities.loc[:,'p10 percent diff'] = 0 # establishing a blank column
nonunc_calculated_capacities.loc[:,'p50 percent diff'] = 0 # establishing a blank column
nonunc_calculated_capacities.loc[:,'p90 percent diff'] = 0 # establishing a blank column

for i in range(0,len(calculated_capacities)):
    
    mine_name = calculated_mines_list[i]
    
    ## pulling 50th percentile of percent change
    
    p10 = float(np.percentile(percent_diff_capacity_uncertainty_values[f'{mine_name}'], 10))
    p50 = float(np.percentile(percent_diff_capacity_uncertainty_values[f'{mine_name}'], 50))
    p90 = float(np.percentile(percent_diff_capacity_uncertainty_values[f'{mine_name}'], 90))
    
    
    nonunc_calculated_capacities.loc[i,'p10 percent diff'] = p10
    nonunc_calculated_capacities.loc[i,'p50 percent diff'] = p50
    nonunc_calculated_capacities.loc[i,'p90 percent diff'] = p90
    
nonunc_calculated_capacities.to_csv(f'/{file_path}/Uncertainty Analysis/Capacity/nonunc_calculated_capacities.csv')
#%%
    
## what we need to do is find the percent difference between each uncertainty value and the "expected" value for each mine

quantifying_water_uncertainty_values = water_estimates_uncertainty_values

quantifying_water_uncertainty_values = quantifying_water_uncertainty_values.drop(columns = ['Unnamed: 0'])
    
#%%

percent_diff_water_uncertainty_values = pd.DataFrame(columns = quantifying_water_uncertainty_values.columns)    
percent_diff_water_uncertainty_values['sample number'] = range(1, samples+1)

#%%

nosamps = percent_diff_water_uncertainty_values.drop(columns = ['sample number'])
deposit_names = nosamps.columns.tolist()

deposits = [brineevap, brinedle, hr, csc, ofb]

#%%

for i in range(0,len(deposit_names)):
    
    deposit_name = deposit_names[i]
    
    avg_water = deposits[i].mean
    
    percent_diff_water_uncertainty_values[f'{deposit_name}'] = ((quantifying_water_uncertainty_values[f'{deposit_name}'] - avg_water)/avg_water)*100 # calculating percent change



#%%

water_estimates_lit_stats.loc[:,'p10 percent diff'] = 0 # establishing a blank column
water_estimates_lit_stats.loc[:,'p50 percent diff'] = 0 # establishing a blank column
water_estimates_lit_stats.loc[:,'p90 percent diff'] = 0 # establishing a blank column

for i in range(0,len(deposit_names)):
    
    deposit_name = deposit_names[i]
    
    ## pulling 50th percentile of percent change
    
    p10 = float(np.percentile(percent_diff_water_uncertainty_values[f'{deposit_name}'], 10))
    p50 = float(np.percentile(percent_diff_water_uncertainty_values[f'{deposit_name}'], 50))
    p90 = float(np.percentile(percent_diff_water_uncertainty_values[f'{deposit_name}'], 90))
    
    
    water_estimates_lit_stats.loc[i,'p10 percent diff'] = p10
    water_estimates_lit_stats.loc[i,'p50 percent diff'] = p50
    water_estimates_lit_stats.loc[i,'p90 percent diff'] = p90
    

water_estimates_lit_stats.to_csv(f'/{file_path}/Uncertainty Analysis/Capacity/water_estimates_lit_stats.csv')







#%%

## demonstrating ranges when capacity is held constant (water consump by deposit varies) vs when water consump by deposit is constant

capconst = nonunc_calculated_capacities[['Proposed Site Name', 'capacity']].copy()

capconst_uncertainty_values = pd.DataFrame(columns = quantifying_capacity_uncertainty_values.columns)    
capconst_uncertainty_values['sample number'] = range(1, samples+1)



for i in range(0,len(calculated_capacities)):
    
    mine_name = calculated_mines_list[i]
    
    avg_cap = nonunc_calculated_capacities.loc[nonunc_calculated_capacities['Proposed Site Name'] == mine_name, 'capacity'].values[0] ## pulling out capacity for each mine    
    
    capconst_uncertainty_values[f'{mine_name}'] = avg_cap*water_estimates_uncertainty_values['csc (m3/t LCE)']


capconst.loc[:,'p10 percent diff'] = 0 # establishing a blank column
capconst.loc[:,'p50 percent diff'] = 0 # establishing a blank column
capconst.loc[:,'p90 percent diff'] = 0 # establishing a blank column
capconst.loc[:, 'perc diff p10 p90'] = 0

for i in range(0,len(calculated_capacities)):
    
    mine_name = calculated_mines_list[i]
    
    ## pulling 50th percentile of percent change
    
    p10 = float(np.percentile(capconst_uncertainty_values[f'{mine_name}'], 10))
    p50 = float(np.percentile(capconst_uncertainty_values[f'{mine_name}'], 50))
    p90 = float(np.percentile(capconst_uncertainty_values[f'{mine_name}'], 90))
    
    
    capconst.loc[i,'p10 percent diff'] = p10
    capconst.loc[i,'p50 percent diff'] = p50
    capconst.loc[i,'p90 percent diff'] = p90
    capconst.loc[i,'perc diff p10 p90'] = (p90-p10)/np.average([p10,p90])
    
capconst.to_csv(f'/{file_path}/Uncertainty Analysis/Capacity/capconst_uncertainty.csv')
#%%

watconst = nonunc_calculated_capacities[['Proposed Site Name']].copy()

watconst_uncertainty_values = pd.DataFrame(columns = quantifying_capacity_uncertainty_values.columns)    
watconst_uncertainty_values['sample number'] = range(1, samples+1)

watconst['csc water'] = csc.mean

#%%



for i in range(0,len(calculated_capacities)):
    
    mine_name = calculated_mines_list[i]
    
    avg_wat = csc.mean ## all of these mines are csc
    
    watconst_uncertainty_values[f'{mine_name}'] = calculatedcapacity_uncertainty_values[f'{mine_name}']*avg_wat

#%%

watconst.loc[:,'p10 percent diff'] = 0 # establishing a blank column
watconst.loc[:,'p50 percent diff'] = 0 # establishing a blank column
watconst.loc[:,'p90 percent diff'] = 0 # establishing a blank column
watconst.loc[:, 'perc diff p10 p90'] = 0

for i in range(0,len(calculated_capacities)):
    
    mine_name = calculated_mines_list[i]
    
    ## pulling 50th percentile of percent change
    
    p10 = float(np.percentile(watconst_uncertainty_values[f'{mine_name}'], 10))
    p50 = float(np.percentile(watconst_uncertainty_values[f'{mine_name}'], 50))
    p90 = float(np.percentile(watconst_uncertainty_values[f'{mine_name}'], 90))
    
    
    watconst.loc[i,'p10 percent diff'] = p10
    watconst.loc[i,'p50 percent diff'] = p50
    watconst.loc[i,'p90 percent diff'] = p90
    watconst.loc[i,'perc diff p10 p90'] = (p90-p10)/np.average([p10,p90])
    
watconst.to_csv(f'/{file_path}/Uncertainty Analysis/Capacity/watconst_uncertainty.csv')


