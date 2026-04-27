#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 22:11:17 2025

@author: jennatrost
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%%

## IMPORTANT! UPDATE YOUR FILE PATH HERE

file_path = ''

#%%

## pulling in big data set, from https://www.nature.com/articles/sdata20195

alldata = pd.read_csv(f'/{file_path}/Population_Projections/SSP_asrc.csv')

## pulling in dataframe to store everything

us_projections = pd.read_csv(f'/{file_path}/Population_Projections/US_Population_Projections.csv')

#%%

## filtering down alldata to only have 2025 - 2100

alldata_filtered = alldata[alldata['YEAR'].between(2020, 2100)]

#%%

## need to pull out unique years and county fips

years = alldata_filtered['YEAR'].unique() # not a dataframe
years_df = pd.DataFrame({'Year': years}) # now a dataframe

#%%

## summing population data for each SSP and year

for i in range(1,len(years)): ## iterating over years, except for 2020

    current_year = years[i]
    
    current_year_data = alldata_filtered[alldata_filtered['YEAR'] == current_year]
    
    us_projections.loc[i,'SSP1'] = current_year_data['SSP1'].sum()
    us_projections.loc[i,'SSP2'] = current_year_data['SSP2'].sum()
    us_projections.loc[i,'SSP3'] = current_year_data['SSP3'].sum()
    us_projections.loc[i,'SSP4'] = current_year_data['SSP4'].sum()
    us_projections.loc[i,'SSP5'] = current_year_data['SSP5'].sum()
    
## putting in 2020 data

us_population_2020 = 331449281 # from 2020 Census https://data.census.gov/profile/United_States?g=010XX00US
us_projections.loc[0, 'SSP1'] = us_population_2020
us_projections.loc[0, 'SSP2'] = us_population_2020
us_projections.loc[0, 'SSP3'] = us_population_2020
us_projections.loc[0, 'SSP4'] = us_population_2020
us_projections.loc[0, 'SSP5'] = us_population_2020
        
#%%

## going to make a new df with populations divided by the millions

us_projections_scaled = pd.DataFrame({'YEAR': years})
us_projections_scaled['SSP1'] = us_projections['SSP1']/1e6
us_projections_scaled['SSP2'] = us_projections['SSP2']/1e6
us_projections_scaled['SSP3'] = us_projections['SSP3']/1e6
us_projections_scaled['SSP4'] = us_projections['SSP4']/1e6
us_projections_scaled['SSP5'] = us_projections['SSP5']/1e6


#%%

## let's plot, matplotlib

plt.plot(us_projections_scaled['YEAR'], us_projections_scaled['SSP1'])
plt.plot(us_projections_scaled['YEAR'], us_projections_scaled['SSP2'])
plt.plot(us_projections_scaled['YEAR'], us_projections_scaled['SSP3'])
plt.plot(us_projections_scaled['YEAR'], us_projections_scaled['SSP4'])
plt.plot(us_projections_scaled['YEAR'], us_projections_scaled['SSP5'])

#%%

## seaborn

plt.figure(figsize=(6, 6))
sns.lineplot(data = us_projections_scaled, x = 'YEAR', y = 'SSP1')
sns.lineplot(data = us_projections_scaled, x = 'YEAR', y = 'SSP2')
sns.lineplot(data = us_projections_scaled, x = 'YEAR', y = 'SSP3')
sns.lineplot(data = us_projections_scaled, x = 'YEAR', y = 'SSP4')
sns.lineplot(data = us_projections_scaled, x = 'YEAR', y = 'SSP5')
plt.grid(True)


plt.savefig(f'/{file_path}/Population_Projections/ssp_us_pop_projections.pdf', format='pdf', bbox_inches='tight')




