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

## now time for water demand uncertainty

## aq: aquaculture,  dp: domestic, i: industrial, ir: irrigation , ls: livestock, th: thermoelectric

## rcp 8.5, SSP5 for all 5 climate scenarios

## NorESM

CNRM_aq = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/CNRM_aq_ssp5.csv')
CNRM_dp = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/CNRM_dp_ssp5.csv')
CNRM_i = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/CNRM_i_ssp5.csv')
CNRM_ir = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/CNRM_ir_ssp5.csv')
CNRM_ls = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/CNRM_ls_ssp5.csv')
CNRM_th = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/CNRM_th_ssp5.csv')

## HadGEM2

HadGEM2_aq = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/HadGEM2_aq_ssp5.csv')
HadGEM2_dp = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/HadGEM2_dp_ssp5.csv')
HadGEM2_i = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/HadGEM2_i_ssp5.csv')
HadGEM2_ir = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/HadGEM2_ir_ssp5.csv')
HadGEM2_ls = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/HadGEM2_ls_ssp5.csv')
HadGEM2_th = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/HadGEM2_th_ssp5.csv')

## IPSL

IPSL_aq = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/IPSL_aq_ssp5.csv')
IPSL_dp = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/IPSL_dp_ssp5.csv')
IPSL_i = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/IPSL_i_ssp5.csv')
IPSL_ir = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/IPSL_ir_ssp5.csv')
IPSL_ls = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/IPSL_ls_ssp5.csv')
IPSL_th = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/IPSL_th_ssp5.csv')

## CGCM3

CGCM3_aq = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/CGCM3_aq_ssp5.csv')
CGCM3_dp = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/CGCM3_dp_ssp5.csv')
CGCM3_i = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/CGCM3_i_ssp5.csv')
CGCM3_ir = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/CGCM3_ir_ssp5.csv')
CGCM3_ls = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/CGCM3_ls_ssp5.csv')
CGCM3_th = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/CGCM3_th_ssp5.csv')

## NorESM

NorESM_aq = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/NorESM_aq_ssp5.csv')
NorESM_dp = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/NorESM_dp_ssp5.csv')
NorESM_i = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/NorESM_i_ssp5.csv')
NorESM_ir = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/NorESM_ir_ssp5.csv')
NorESM_ls = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/NorESM_ls_ssp5.csv')
NorESM_th = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/NorESM_th_ssp5.csv')

#%%

## loading huc8 values for proposed mines

proposed_mines_huc8 = pd.read_csv(f'/{file_path}/Uncertainty Analysis/proposed_mines_huc8.csv')

## pulling out unique huc8s

unique_huc8_ids = proposed_mines_huc8['HUC8 ID'].unique() # not a dataframe
unique_huc8_ids_df = pd.DataFrame({'HUC8 ID': unique_huc8_ids}) # now a dataframe

## creating monte carlo

huc8_list = unique_huc8_ids_df['HUC8 ID'].astype(str).tolist()  # string list of HUCs... these will be the column names in our iterated dfs

#%%

## creating a function to make dfs for each water demand type and climate scenario

def water_demand_df(unique_hucs):
    
    return pd.DataFrame({
        'HUC8 ID': unique_hucs['HUC8 ID'],
        'mean': 0,  # Blank column
        'min': 0,    # Blank column
        'max': 0
    })

## all climate models

## CNRM

CNRM_aq_stats = water_demand_df(unique_huc8_ids_df)
CNRM_dp_stats = water_demand_df(unique_huc8_ids_df)
CNRM_i_stats = water_demand_df(unique_huc8_ids_df)
CNRM_ir_stats  = water_demand_df(unique_huc8_ids_df)
CNRM_ls_stats  = water_demand_df(unique_huc8_ids_df)
CNRM_th_stats  = water_demand_df(unique_huc8_ids_df)

## HadGEM2

HadGEM2_aq_stats = water_demand_df(unique_huc8_ids_df)
HadGEM2_dp_stats = water_demand_df(unique_huc8_ids_df)
HadGEM2_i_stats = water_demand_df(unique_huc8_ids_df)
HadGEM2_ir_stats  = water_demand_df(unique_huc8_ids_df)
HadGEM2_ls_stats  = water_demand_df(unique_huc8_ids_df)
HadGEM2_th_stats  = water_demand_df(unique_huc8_ids_df)

## IPSL

IPSL_aq_stats = water_demand_df(unique_huc8_ids_df)
IPSL_dp_stats = water_demand_df(unique_huc8_ids_df)
IPSL_i_stats = water_demand_df(unique_huc8_ids_df)
IPSL_ir_stats  = water_demand_df(unique_huc8_ids_df)
IPSL_ls_stats  = water_demand_df(unique_huc8_ids_df)
IPSL_th_stats  = water_demand_df(unique_huc8_ids_df)

## CGCM3

CGCM3_aq_stats = water_demand_df(unique_huc8_ids_df)
CGCM3_dp_stats = water_demand_df(unique_huc8_ids_df)
CGCM3_i_stats = water_demand_df(unique_huc8_ids_df)
CGCM3_ir_stats  = water_demand_df(unique_huc8_ids_df)
CGCM3_ls_stats  = water_demand_df(unique_huc8_ids_df)
CGCM3_th_stats  = water_demand_df(unique_huc8_ids_df)

## NorESM

NorESM_aq_stats = water_demand_df(unique_huc8_ids_df)
NorESM_dp_stats = water_demand_df(unique_huc8_ids_df)
NorESM_i_stats = water_demand_df(unique_huc8_ids_df)
NorESM_ir_stats  = water_demand_df(unique_huc8_ids_df)
NorESM_ls_stats  = water_demand_df(unique_huc8_ids_df)
NorESM_th_stats  = water_demand_df(unique_huc8_ids_df)

#%%

## storing into arrays (aka prepping for that double nested!)

CNRM_demands = [CNRM_aq,CNRM_dp,CNRM_i,CNRM_ir,CNRM_ls,CNRM_th]
CNRM_stats = [CNRM_aq_stats,CNRM_dp_stats,CNRM_i_stats,CNRM_ir_stats,CNRM_ls_stats,CNRM_th_stats]

HadGEM2_demands = [HadGEM2_aq,HadGEM2_dp,HadGEM2_i,HadGEM2_ir,HadGEM2_ls,HadGEM2_th]
HadGEM2_stats = [HadGEM2_aq_stats,HadGEM2_dp_stats,HadGEM2_i_stats,HadGEM2_ir_stats,HadGEM2_ls_stats,HadGEM2_th_stats]

IPSL_demands = [IPSL_aq,IPSL_dp,IPSL_i,IPSL_ir,IPSL_ls,IPSL_th]
IPSL_stats = [IPSL_aq_stats,IPSL_dp_stats,IPSL_i_stats,IPSL_ir_stats,IPSL_ls_stats,IPSL_th_stats]

CGCM3_demands = [CGCM3_aq,CGCM3_dp,CGCM3_i,CGCM3_ir,CGCM3_ls,CGCM3_th]
CGCM3_stats = [CGCM3_aq_stats,CGCM3_dp_stats,CGCM3_i_stats,CGCM3_ir_stats,CGCM3_ls_stats,CGCM3_th_stats]

NorESM_demands = [NorESM_aq,NorESM_dp,NorESM_i,NorESM_ir,NorESM_ls,NorESM_th]
NorESM_stats = [NorESM_aq_stats,NorESM_dp_stats,NorESM_i_stats,NorESM_ir_stats,NorESM_ls_stats,CNRM_th_stats]

## demands and stats

demands = [CNRM_demands, HadGEM2_demands, IPSL_demands, CGCM3_demands, NorESM_demands]
stats = [CNRM_stats, HadGEM2_stats, IPSL_stats, CGCM3_stats, NorESM_stats]


#%%

## writing function to filter down df to just 2040 - 2060

def filter_years(df, minyear, maxyear):
    
    years = [f'Avg_Y{year}' for year in range(minyear, maxyear + 1)]
    
    df_filtered = df[['HUC8'] + years]
    
    return df_filtered


## writing functions to pull mean and stds

def calc_water_demand(waterdemand,hucs):
    
    for i in range(0,len(hucs['HUC8 ID'])):
        
        huc = hucs['HUC8 ID'][i]
        
        waterdemand_reduced = waterdemand[waterdemand['HUC8'] == huc]
        
        ## isolating values that are not huc 8
        
        yearvalues = waterdemand_reduced.iloc[0, 1:].astype(float)
        
        hucs.loc[i, 'mean'] = float(yearvalues.mean()) ## useful for normal and triangular
        hucs.loc[i, 'min'] = float(yearvalues.min()) ## useful for normal and triangular
        hucs.loc[i, 'max'] = float(yearvalues.max()) ## useful for normal and triangular
        
        hucs.loc[i, 'std'] = float(yearvalues.std()) ## useful for normal 
        
        # shape, loc, scale = weibull_min.fit(yearvalues, floc=0)  ## finding shape of weibull

        # hucs.loc[i, 'a'] = float(shape) ## useful for weibull
        # hucs.loc[i, 'k'] = float(scale) ## useful for weibull

#%%
## filtering datasets

for i in range(0,len(demands)): # iterating over big array

    for j in range(0,len(CNRM_demands)): # iterating over individual arrays
    
        demands[i][j] = filter_years(demands[i][j], 2040, 2060)

for i in range(0,len(demands)): # iterating over big array

    for j in range(0,len(CNRM_demands)): # iterating over individual arrays
    
        demands[i][j] = filter_years(demands[i][j], 2040, 2060)
        calc_water_demand(demands[i][j], stats[i][j])


#%%

## creating all csvs

sample_df = pd.DataFrame({'sample number': range(1,samples+1)})

climate_models_str = ['CNRM', 'HadGEM2', 'IPSL', 'CGCM3', 'NorESM']
demand_types_str = ['aq', 'dp', 'i', 'ir', 'ls', 'th']

#%%

for i in range(0,len(climate_models_str)):
    for j in range(0,len(demand_types_str)):
        
        climate_name = climate_models_str[i]
        demand_type_name = demand_types_str[j]
        
        filename = f'{climate_name}_{demand_type_name}_uncertainty_values.csv'
        filepath = ff'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/Uncertainty_Values/{filename}'
    
        
        sample_df.to_csv(filepath, index = False)


#%%
## creating dfs to store uncertainty values

CNRM_aq_uncertainty_values = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/Uncertainty_Values/CNRM_aq_uncertainty_values.csv')
CNRM_dp_uncertainty_values = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/Uncertainty_Values/CNRM_dp_uncertainty_values.csv')
CNRM_i_uncertainty_values = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/Uncertainty_Values/CNRM_i_uncertainty_values.csv')
CNRM_ir_uncertainty_values = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/Uncertainty_Values/CNRM_ir_uncertainty_values.csv')
CNRM_ls_uncertainty_values = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/Uncertainty_Values/CNRM_ls_uncertainty_values.csv')
CNRM_th_uncertainty_values = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/Uncertainty_Values/CNRM_th_uncertainty_values.csv')

HadGEM2_aq_uncertainty_values = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/Uncertainty_Values/HadGEM2_aq_uncertainty_values.csv')
HadGEM2_dp_uncertainty_values = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/Uncertainty_Values/HadGEM2_dp_uncertainty_values.csv')
HadGEM2_i_uncertainty_values = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/Uncertainty_Values/HadGEM2_i_uncertainty_values.csv')
HadGEM2_ir_uncertainty_values = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/Uncertainty_Values/HadGEM2_ir_uncertainty_values.csv')
HadGEM2_ls_uncertainty_values = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/Uncertainty_Values/HadGEM2_ls_uncertainty_values.csv')
HadGEM2_th_uncertainty_values = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/Uncertainty_Values/HadGEM2_th_uncertainty_values.csv')

IPSL_aq_uncertainty_values = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/Uncertainty_Values/IPSL_aq_uncertainty_values.csv')
IPSL_dp_uncertainty_values = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/Uncertainty_Values/IPSL_dp_uncertainty_values.csv')
IPSL_i_uncertainty_values = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/Uncertainty_Values/IPSL_i_uncertainty_values.csv')
IPSL_ir_uncertainty_values = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/Uncertainty_Values/IPSL_ir_uncertainty_values.csv')
IPSL_ls_uncertainty_values = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/Uncertainty_Values/IPSL_ls_uncertainty_values.csv')
IPSL_th_uncertainty_values = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/Uncertainty_Values/IPSL_th_uncertainty_values.csv')

CGCM3_aq_uncertainty_values = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/Uncertainty_Values/CGCM3_aq_uncertainty_values.csv')
CGCM3_dp_uncertainty_values = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/Uncertainty_Values/CGCM3_dp_uncertainty_values.csv')
CGCM3_i_uncertainty_values = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/Uncertainty_Values/CGCM3_i_uncertainty_values.csv')
CGCM3_ir_uncertainty_values = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/Uncertainty_Values/CGCM3_ir_uncertainty_values.csv')
CGCM3_ls_uncertainty_values = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/Uncertainty_Values/CGCM3_ls_uncertainty_values.csv')
CGCM3_th_uncertainty_values = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/Uncertainty_Values/CGCM3_th_uncertainty_values.csv')

NorESM_aq_uncertainty_values = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/Uncertainty_Values/NorESM_aq_uncertainty_values.csv')
NorESM_dp_uncertainty_values = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/Uncertainty_Values/NorESM_dp_uncertainty_values.csv')
NorESM_i_uncertainty_values = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/Uncertainty_Values/NorESM_i_uncertainty_values.csv')
NorESM_ir_uncertainty_values = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/Uncertainty_Values/NorESM_ir_uncertainty_values.csv')
NorESM_ls_uncertainty_values = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/Uncertainty_Values/NorESM_ls_uncertainty_values.csv')
NorESM_th_uncertainty_values = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/Uncertainty_Values/NorESM_th_uncertainty_values.csv')

#%%

## making uncs arrays

CNRM_uncs = [CNRM_aq_uncertainty_values, CNRM_dp_uncertainty_values, CNRM_i_uncertainty_values, CNRM_ir_uncertainty_values, CNRM_ls_uncertainty_values, CNRM_th_uncertainty_values]
HadGEM2_uncs = [HadGEM2_aq_uncertainty_values, HadGEM2_dp_uncertainty_values, HadGEM2_i_uncertainty_values, HadGEM2_ir_uncertainty_values, HadGEM2_ls_uncertainty_values, HadGEM2_th_uncertainty_values]
IPSL_uncs = [IPSL_aq_uncertainty_values, IPSL_dp_uncertainty_values, IPSL_i_uncertainty_values, IPSL_ir_uncertainty_values, IPSL_ls_uncertainty_values, IPSL_th_uncertainty_values]
CGCM3_uncs = [CGCM3_aq_uncertainty_values, CGCM3_dp_uncertainty_values, CGCM3_i_uncertainty_values, CGCM3_ir_uncertainty_values, CGCM3_ls_uncertainty_values, CGCM3_th_uncertainty_values]
NorESM_uncs = [NorESM_aq_uncertainty_values, NorESM_dp_uncertainty_values, NorESM_i_uncertainty_values, NorESM_ir_uncertainty_values, NorESM_ls_uncertainty_values, NorESM_th_uncertainty_values]

climate_uncs = [CNRM_uncs,HadGEM2_uncs,IPSL_uncs,CGCM3_uncs,NorESM_uncs]

#%%

for i in range(0,len(climate_uncs)):
    for j in range(0,len(CNRM_uncs)):
        
        climate_uncs[i][j][huc8_list] = pd.NA

#%%

demands = [CNRM_demands, HadGEM2_demands, IPSL_demands, CGCM3_demands, NorESM_demands]
stats = [CNRM_stats, HadGEM2_stats, IPSL_stats, CGCM3_stats, NorESM_stats]

runwaterdemand = True

if runwaterdemand == True:
    
    ## THE TRIPLE DIPLE NESTER! WOW
    
    for i in range(0,len(demands)): ## tracker for which climate demand
    
        for j in range(0,len(CNRM_demands)): ## tracker for which demand type within climate demand
            
            demands_df = demands[i][j]
            unc_df = climate_uncs[i][j] ## specific unc for climate and demand type
            
            for k in range(0,len(huc8_list)): #iterating over columns
            
                stats_df = stats[i][j]
                
                huc8_float = float(huc8_list[k])
            
                huc_index = stats_df[stats_df['HUC8 ID'] == huc8_float].index[0]
                
                huc_mean = float(stats_df.loc[huc_index, 'mean'])
                
                huc_std = float(stats_df.loc[huc_index, 'std'])
                
                huc_max = float(stats_df.loc[huc_index, 'max'])
                
                huc_min = float(stats_df.loc[huc_index, 'min'])
                
                # huc_a = float(stats_df.loc[huc_index, 'a'])
                
                # huc_k = float(stats_df.loc[huc_index, 'k'])
                
                
                ## triangular
                
                if huc_min < huc_mean: ## this is to get rid of the cases where water demand is zero
                
                    unc_df[huc8_list[k]] = (np.random.triangular(huc_min, huc_mean, huc_max, samples))
                
                else:
                    
                    unc_df[huc8_list[k]] = huc_mean
                    
            ## saving into csvs
            
            climate_name = climate_models_str[i]
            demand_type_name = demand_types_str[j]
            
            filename = f'{climate_name}_{demand_type_name}_uncertainty_values.csv'
            filepath = ff'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/Uncertainty_Values/{filename}'
        
            
            unc_df.to_csv(filepath, index = False)

#%%

## creating a function to make percentile dfs for each water demand type and climate scenario

percs_df = pd.DataFrame({
        'HUC8 ID': unique_huc8_ids_df['HUC8 ID'],
        'p10': 0,  # Blank column
        'p50': 0,    # Blank column
        'p90': 0
    })

for i in range(0,len(climate_models_str)):
    for j in range(0,len(demand_types_str)):
        
        climate_name = climate_models_str[i]
        demand_type_name = demand_types_str[j]
        
        filename = f'{climate_name}_{demand_type_name}_percs.csv'
        filepath = ff'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/percs/{filename}'
    
        
        percs_df.to_csv(filepath, index = False)
        
#%%

CNRM_aq_percs = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/percs/CNRM_aq_percs.csv')
CNRM_dp_percs = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/percs/CNRM_dp_percs.csv')
CNRM_i_percs = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/percs/CNRM_i_percs.csv')
CNRM_ir_percs = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/percs/CNRM_ir_percs.csv')
CNRM_ls_percs = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/percs/CNRM_ls_percs.csv')
CNRM_th_percs = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/percs/CNRM_th_percs.csv')

HadGEM2_aq_percs = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/percs/HadGEM2_aq_percs.csv')
HadGEM2_dp_percs = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/percs/HadGEM2_dp_percs.csv')
HadGEM2_i_percs = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/percs/HadGEM2_i_percs.csv')
HadGEM2_ir_percs = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/percs/HadGEM2_ir_percs.csv')
HadGEM2_ls_percs = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/percs/HadGEM2_ls_percs.csv')
HadGEM2_th_percs = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/percs/HadGEM2_th_percs.csv')

IPSL_aq_percs = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/percs/IPSL_aq_percs.csv')
IPSL_dp_percs = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/percs/IPSL_dp_percs.csv')
IPSL_i_percs = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/percs/IPSL_i_percs.csv')
IPSL_ir_percs = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/percs/IPSL_ir_percs.csv')
IPSL_ls_percs = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/percs/IPSL_ls_percs.csv')
IPSL_th_percs = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/percs/IPSL_th_percs.csv')

CGCM3_aq_percs = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/percs/CGCM3_aq_percs.csv')
CGCM3_dp_percs = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/percs/CGCM3_dp_percs.csv')
CGCM3_i_percs = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/percs/CGCM3_i_percs.csv')
CGCM3_ir_percs = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/percs/CGCM3_ir_percs.csv')
CGCM3_ls_percs = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/percs/CGCM3_ls_percs.csv')
CGCM3_th_percs = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/percs/CGCM3_th_percs.csv')

NorESM_aq_percs = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/percs/NorESM_aq_percs.csv')
NorESM_dp_percs = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/percs/NorESM_dp_percs.csv')
NorESM_i_percs = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/percs/NorESM_i_percs.csv')
NorESM_ir_percs = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/percs/NorESM_ir_percs.csv')
NorESM_ls_percs = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/percs/NorESM_ls_percs.csv')
NorESM_th_percs = pd.read_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/percs/NorESM_th_percs.csv')


#%%

## into iterable arrays

CNRM_percs = [CNRM_aq_percs, CNRM_dp_percs, CNRM_i_percs, CNRM_ir_percs, CNRM_ls_percs, CNRM_th_percs]
HadGEM2_percs = [HadGEM2_aq_percs, HadGEM2_dp_percs, HadGEM2_i_percs, HadGEM2_ir_percs, HadGEM2_ls_percs, HadGEM2_th_percs]
IPSL_percs = [IPSL_aq_percs, IPSL_dp_percs, IPSL_i_percs, IPSL_ir_percs, IPSL_ls_percs, IPSL_th_percs]
CGCM3_percs = [CGCM3_aq_percs, CGCM3_dp_percs, CGCM3_i_percs, CGCM3_ir_percs, CGCM3_ls_percs, CGCM3_th_percs]
NorESM_percs = [NorESM_aq_percs, NorESM_dp_percs, NorESM_i_percs, NorESM_ir_percs, NorESM_ls_percs, NorESM_th_percs]

percs = [CNRM_percs, HadGEM2_percs, IPSL_percs, CGCM3_percs, NorESM_percs]

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

for i in range(0,len(percs)): # iterating over each climate model
    for j in range(0,len(CNRM_percs)): # iterating over each demand type within climate model
    
    
        calc_percs(percs[i][j], climate_uncs[i][j] , huc8_list)
    
        climate_name = climate_models_str[i]
        demand_type_name = demand_types_str[j]
        
        filename = f'{climate_name}_{demand_type_name}_percs.csv'
        filepath = ff'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/percs/{filename}'
    
        
        percs[i][j].to_csv(filepath, index = False)
    

#%%
aq_percs = pd.DataFrame({ 'HUC8 ID': unique_huc8_ids_df['HUC8 ID'] })
dp_percs = pd.DataFrame({ 'HUC8 ID': unique_huc8_ids_df['HUC8 ID'] })
i_percs = pd.DataFrame({ 'HUC8 ID': unique_huc8_ids_df['HUC8 ID'] })
ir_percs = pd.DataFrame({ 'HUC8 ID': unique_huc8_ids_df['HUC8 ID'] })
ls_percs = pd.DataFrame({ 'HUC8 ID': unique_huc8_ids_df['HUC8 ID'] })
th_percs = pd.DataFrame({ 'HUC8 ID': unique_huc8_ids_df['HUC8 ID'] })



for i in range(0,len(percs)): ## iterating over all climate models
    for j in range(0,len(demand_types_str)): ## iterating over all demand types for one climate model
        
        climate_name = climate_models_str[i]
        
        c_p10 = f'{climate_name}_p10'
        c_p50 = f'{climate_name}_p50'
        c_p90 = f'{climate_name}_p90'
        
        
        if j == 0: ## aquaculture
        
            aq_percs[c_p10] = percs[i][j]['p10']
            aq_percs[c_p50] = percs[i][j]['p50']
            aq_percs[c_p90] = percs[i][j]['p90']
            
        if j == 1: ## domestic
        
            dp_percs[c_p10] = percs[i][j]['p10']
            dp_percs[c_p50] = percs[i][j]['p50']
            dp_percs[c_p90] = percs[i][j]['p90']
            
        if j == 2: ## industrial
        
            i_percs[c_p10] = percs[i][j]['p10']
            i_percs[c_p50] = percs[i][j]['p50']
            i_percs[c_p90] = percs[i][j]['p90']
            
        if j == 3: ## irrigation
        
            ir_percs[c_p10] = percs[i][j]['p10']
            ir_percs[c_p50] = percs[i][j]['p50']
            ir_percs[c_p90] = percs[i][j]['p90']
            
        if j == 4: ## livestock
        
            ls_percs[c_p10] = percs[i][j]['p10']
            ls_percs[c_p50] = percs[i][j]['p50']
            ls_percs[c_p90] = percs[i][j]['p90']
            
        if j == 5: ## thermoelectric
        
            th_percs[c_p10] = percs[i][j]['p10']
            th_percs[c_p50] = percs[i][j]['p50']
            th_percs[c_p90] = percs[i][j]['p90']
            
        
## saving csvs

aq_percs.to_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/percs/aq_percs.csv')
dp_percs.to_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/percs/dp_percs.csv')
i_percs.to_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/percs/i_percs.csv')
ir_percs.to_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/percs/ir_percs.csv')
ls_percs.to_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/percs/ls_percs.csv')
th_percs.to_csv(f'/{file_path}/Uncertainty Analysis/SSP5/SSP5_Demand/percs/th_percs.csv')          
        


        















        