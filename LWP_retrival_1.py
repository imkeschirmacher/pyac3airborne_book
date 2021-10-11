#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 08:55:50 2021

@author: ischirma
"""
import xarray as xr
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import ac3airborne
import pandas as pd
import numpy as np
import datetime

def cubic_regression(x,c1, c2, c3):return c1*x+c2*x**2+c3*x**3

def consecutive_valid(data, N):
    """
    This function selects all sections longer than N consecutive data points that are not equal nan.
    """
    #if (N % 2) != 0: N=2*(N//2)+1 # check for add N
    #return data.loc[np.isfinite(data.rolling(time=N, min_periods=N, center=True).sum().rolling(time=N, min_periods=1, center=True).sum())]
    return data.loc[np.isfinite(data[::-1].rolling(time=N, min_periods=N, center=True).sum()[::-1].rolling(time=N, min_periods=1, center=True).sum())]


ds_pam = xr.open_dataset('/home/mech/teaching/metrs_project/data/pamtra_icon_passive.nc') # open data set with PAMTRA simulations
tb = ds_pam['tb'][:,0,0,4,0,1] # dims(nx,ny,nf,nang,nobs,npol) get the horizontal (npol=1) polarization at ~25° (nang=4), cause MiRAC-A 89 GHz is horizontally polarized
twp = ds_pam['cwp']+ds_pam['iwp']+ds_pam['rwp']+ds_pam['swp']+ds_pam['gwp']+ds_pam['hwp'] # this is the total integrated hydrometeor content from the ICON simulation
tb[twp[:,0] < 0.00001].plot.line("r.") # brightness temperature where total water path (twp) content is very small
tb_clear = tb[twp[:,0] < 0.00001].mean() #  find the mean clear sky brightness temperature
tb_diff = tb - tb_clear # this is the field of brightness temperature differences that enters the retrieval development
c1,c2,c3 = curve_fit(cubic_regression, tb_diff.values,ds_pam['cwp'][:,0].values)[0]
lwp = c1*tb_diff+c2*tb_diff**2+c3*tb_diff**3
plt.figure()
sc=plt.scatter(lwp,ds_pam['cwp'][:,0],c=[float(total) for total in twp.values])#c=twp
plt.plot([0,1,2],'k')
plt.xlabel('LWP retrieved [$\mathrm{kg/{m^2}}$]')
plt.ylabel('LWP simulated [$\mathrm{kg/{m^2}}$]')
cbar = plt.colorbar(sc)
cbar.set_label('TWP')

# öffnen der Tb 89 GHz Dateien 
cat = ac3airborne.get_intake_catalog()
#liste=list(cat.P5.MIRAC_A)
string = 'MOSAiC-ACA'
campaign='MOSAiC-ACA_P5_RF07'

    
meta = ac3airborne.get_flight_segments() 
segments = {s.get("segment_id"): {**s, "flight_id": flight["flight_id"]}
              for platform in meta.values()
              for flight in platform.values()
              for s in flight["segments"]
             }
list_seg=list(segments.keys())

ds_mirac_a = cat['P5']['MIRAC_A'][campaign].to_dask()
ds_sea_ice = cat['P5']['AMSR2_SIC'][campaign].to_dask()
track = cat.P5.GPS_INS[campaign].to_dask() # get position and attitude of platform

open_ocean=ds_sea_ice.where(ds_sea_ice.sic == 0)
long_ocean_sections = consecutive_valid(open_ocean['sic'],60*30).dropna('time')

track_ocean = track.loc[dict(time=long_ocean_sections.time)]
time_rng = pd.date_range(datetime.datetime.now().date(), periods=0, freq='S') # define empty pandas time range

for seg in meta['P5'][campaign]['segments']:
    if 'high_level' in seg['kinds']:
        time_rng = time_rng.union(pd.date_range(seg['start'],seg['end'],freq='S'))
        
track_ocean_straight_legs = track_ocean.loc[dict(time=time_rng.isin(track_ocean.time))]   
track_ocean_straight_legs.attrs = dict()
track_ocean_straight_legs.drop_vars(['tas','vs','gs','pitch','roll','alt','heading']).to_netcdf('polar5_ocean_track_straight_legs.nc')
tb_ocean = ds_mirac_a.tb.loc[isin(mirac.time,long_ocean_sections.time)]
tb_straight_legs = mirac.tb.loc[isin(mirac.time,track_ocean_straight_legs.time)]
tb_clear = mean([tb_ocean.sel(time="2020-09-07 08:58:00").values, tb_ocean.sel(time="2020-09-07 12:13:13").values, tb_ocean.sel(time="2020-09-07 12:46:00").values])
lwp_mirac = c1*(tb_straight_legs-tb_clear)+c2*(tb_straight_legs-tb_clear)**2+c3*(tb_straight_legs-tb_clear)**3