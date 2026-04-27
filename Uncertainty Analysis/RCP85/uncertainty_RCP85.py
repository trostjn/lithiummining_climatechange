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

## importing climate models for SSP2 (RCP 8.5)

CNRM_85 = pd.read_csv(f'/{file_path}/Uncertainty Analysis/RCP85/WaSSI_CNRMCM5_RCP85.TXT')
HadGEM2_85 = pd.read_csv(f'/{file_path}/Uncertainty Analysis/RCP85/WaSSI_HadGEM2ES365_RCP85.TXT')
IPSL_85 = pd.read_csv(f'/{file_path}/Uncertainty Analysis/RCP85/WaSSI_IPSLCM5AMR_RCP85.TXT')
CGCM3_85 = pd.read_csv(f'/{file_path}/Uncertainty Analysis/RCP85/WaSSI_MRICGCM3_RCP85.TXT')
NORESM_85 = pd.read_csv(f'/{file_path}/Uncertainty Analysis/RCP85/WaSSI_NORESM1M_RCP85.TXT')

#%%

## loading huc8 values for proposed mines

proposed_mines_huc8 = pd.read_csv(f'/{file_path}/Uncertainty Analysis/RCP85/proposed_mines_huc8.csv')

## pulling out unique huc8s

unique_huc8_ids = proposed_mines_huc8['HUC8 ID'].unique() # not a dataframe
unique_huc8_ids_df = pd.DataFrame({'HUC8 ID': unique_huc8_ids}) # now a dataframe

#%%

## filtering CGCM files to only contain unique_huc8_ids

CNRM_85 = CNRM_85[CNRM_85['CELL'].isin(unique_huc8_ids)]
HadGEM2_85 = HadGEM2_85[HadGEM2_85['CELL'].isin(unique_huc8_ids)]
IPSL_85 = IPSL_85[IPSL_85['CELL'].isin(unique_huc8_ids)]
CGCM3_85 = CGCM3_85[CGCM3_85['CELL'].isin(unique_huc8_ids)]
NORESM_85 = NORESM_85[NORESM_85['CELL'].isin(unique_huc8_ids)]

#%%

## writing a function that makes a stat df for each climate model

def climate_stats_df(unique_hucs): # unique_hucs MUST be a df
    
    return pd.DataFrame({
        'HUC8 ID': unique_hucs['HUC8 ID'],
        'mean': 0,  # Blank column
        'min': 0,    # Blank column
        'max': 0,
        'std': 0
    })
    

## zeroes for now, will update with below

CNRM_stats = climate_stats_df(unique_huc8_ids_df)
HadGEM2_stats = climate_stats_df(unique_huc8_ids_df)
IPSL_stats = climate_stats_df(unique_huc8_ids_df)
CGCM3_stats = climate_stats_df(unique_huc8_ids_df)
NORESM_stats = climate_stats_df(unique_huc8_ids_df)

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

## climate array

climate_models = [CNRM_85, HadGEM2_85, IPSL_85, CGCM3_85, NORESM_85]
climate_stats = [CNRM_stats, HadGEM2_stats, IPSL_stats, CGCM3_stats, NORESM_stats]

for i in range(0,len(climate_models)): ## should iterate over these arrays
    
    calc_stats(climate_models[i], climate_stats[i])

#%%

## creating monte carlo

huc8_list = unique_huc8_ids_df['HUC8 ID'].astype(str).tolist()  # string list of HUCs... these will be the column names in our iterated dfs

#%%

## establishing uncertainty values - ONLY doing triangular for now

CNRM_85_uncertainty_values_triangular = pd.read_csv(f'/{file_path}/Uncertainty Analysis/RCP85/CNRM_85_uncertainty_values_triangular.csv')
HadGEM2_85_uncertainty_values_triangular = pd.read_csv(f'/{file_path}/Uncertainty Analysis/RCP85/HadGEM2_85_uncertainty_values_triangular.csv')
IPSL_85_uncertainty_values_triangular = pd.read_csv(f'/{file_path}/Uncertainty Analysis/RCP85/IPSL_85_uncertainty_values_triangular.csv')
CGCM3_85_uncertainty_values_triangular = pd.read_csv(f'/{file_path}/Uncertainty Analysis/RCP85/CGCM3_85_uncertainty_values_triangular.csv')
NORESM_85_uncertainty_values_triangular = pd.read_csv(f'/{file_path}/Uncertainty Analysis/RCP85/NORESM_85_uncertainty_values_triangular.csv')


for i in huc8_list:
    CNRM_85_uncertainty_values_triangular[huc8_list] = pd.NA # storing blanks initially in the columns
    HadGEM2_85_uncertainty_values_triangular[huc8_list] = pd.NA # storing blanks initially in the columns
    IPSL_85_uncertainty_values_triangular[huc8_list] = pd.NA # storing blanks initially in the columns
    CGCM3_85_uncertainty_values_triangular[huc8_list] = pd.NA # storing blanks initially in the columns
    NORESM_85_uncertainty_values_triangular[huc8_list] = pd.NA # storing blanks initially in the columns
    
    
climate_85_names = { # storing names to help automate file naming
                 'CNRM_85': CNRM_85_uncertainty_values_triangular,
                 'HadGEM2_85': HadGEM2_85_uncertainty_values_triangular,
                 'IPSL_85': IPSL_85_uncertainty_values_triangular,
                 'CGCM3_85': CGCM3_85_uncertainty_values_triangular,
                 'NORESM_85': NORESM_85_uncertainty_values_triangular
    
    }

## putting all uncs in an array to make Monte Carlo easier

climate_unc_values = [CNRM_85_uncertainty_values_triangular,HadGEM2_85_uncertainty_values_triangular,IPSL_85_uncertainty_values_triangular,CGCM3_85_uncertainty_values_triangular,NORESM_85_uncertainty_values_triangular]

## making these names callable

climate_85_names_iterable = [] ## creating a blank array

for key in climate_85_names:
    climate_85_names_iterable.append(key)

#%%

runclimate = True

## running monte carlo

if runclimate == True:
    
    CNRM_85_uncertainty_values_triangular['sample number'] = range(1,samples+1) # since we're not starting at zero, only need to do this once
    HadGEM2_85_uncertainty_values_triangular['sample number'] = range(1,samples+1) # since we're not starting at zero, only need to do this once
    IPSL_85_uncertainty_values_triangular['sample number'] = range(1,samples+1) # since we're not starting at zero, only need to do this once
    CGCM3_85_uncertainty_values_triangular['sample number'] = range(1,samples+1) # since we're not starting at zero, only need to do this once
    NORESM_85_uncertainty_values_triangular['sample number'] = range(1,samples+1) # since we're not starting at zero, only need to do this once
    
    for j in range(0,len(climate_unc_values)): #iterate over climate uncs
    
        for i in range(0,len(huc8_list)): #iterating over columns
            
            huc8_float = float(huc8_list[i])
            
            stats = climate_stats[j]
        
            huc_index = stats[stats['HUC8 ID'] == huc8_float].index[0]
            
            huc_mean = stats.loc[huc_index, 'mean']
            
            huc_std = stats.loc[huc_index, 'std']
            
            huc_max = stats.loc[huc_index, 'max']
            
            huc_min = stats.loc[huc_index, 'min']
            
            huc_a = stats.loc[huc_index, 'a']
            
            huc_k = stats.loc[huc_index, 'k']
            
            ## REMEMBER THE VALUES ARE TOTAL WATER SUPPLY IN MGAL/DAY
    
            ## triangular
            
            climate_unc_values[j][huc8_list[i]] = (np.random.triangular(huc_min, huc_mean, huc_max, samples))
        
        
    ## saving into csvs
    
        name = climate_85_names_iterable[j]
        filename = f"{name}_uncertainty_values_triangular.csv"
        filepath = f'/{file_path}/Uncertainty Analysis/RCP85/{filename}'
    
        climate_unc_values[j].to_csv(filepath, index = False)

  
#%%

## pulling out 10, 50, and 90 percentiles 

percs_CNRM_85_triangular = pd.DataFrame({'HUC8 ID': unique_huc8_ids_df['HUC8 ID']})
percs_HadGEM2_85_triangular = pd.DataFrame({'HUC8 ID': unique_huc8_ids_df['HUC8 ID']})
percs_IPSL_85_triangular = pd.DataFrame({'HUC8 ID': unique_huc8_ids_df['HUC8 ID']})
percs_CGCM3_85_triangular = pd.DataFrame({'HUC8 ID': unique_huc8_ids_df['HUC8 ID']})
percs_NORESM_85_triangular = pd.DataFrame({'HUC8 ID': unique_huc8_ids_df['HUC8 ID']})

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

## rerunning this array to contain updated unc values

climate_unc_values = [CNRM_85_uncertainty_values_triangular,HadGEM2_85_uncertainty_values_triangular,IPSL_85_uncertainty_values_triangular,CGCM3_85_uncertainty_values_triangular,NORESM_85_uncertainty_values_triangular]
percs_85 = [percs_CNRM_85_triangular,percs_HadGEM2_85_triangular,percs_IPSL_85_triangular, percs_CGCM3_85_triangular, percs_NORESM_85_triangular]

#%%
for i in range(0,len(percs_85)):
    
    calc_percs(percs_85[i], climate_unc_values[i], huc8_list)
    
    name = climate_85_names_iterable[i]
    filename = f"percs_{name}_triangular.csv"
    filepath = f'/{file_path}/Uncertainty Analysis/RCP85/{filename}'
    
    percs_85[i].to_csv(filepath, index = False)



        















        