#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 14:15:08 2021

@author: ischirma
"""
import matplotlib.pyplot as plt
import ac3airborne
import numpy as np
from numpy import log10
import matplotlib.dates as mdates
import ac3airborne.tools.is_land as il
from simplification.cutil import simplify_coords_idx
#import xarray as xr
import pandas as pd

def simplify_dataset(ds, tolerance):
    indices_to_take = simplify_coords_idx(np.stack([ds.lat.values, ds.lon.values], axis=1), tolerance)
    return ds.isel(time=indices_to_take)



cat = ac3airborne.get_intake_catalog()

liste=list(set(cat.P5.CLOUD_TOP_HEIGHT).intersection(cat.P5.MIRAC_A))
string = 'MOSAiC-ACA'
i=0
index=[]
for test in liste:
    if string in test:
        print(i)
        index.append(i) #index: all inddicies that have ACLOUD and contain AMALI and MIRAC data
    i +=1   


#campaign='ACLOUD_P5_RF07'
#spec_segment='ACLOUD_P5_RF07_hl09'
meta = ac3airborne.get_flight_segments() 
segments = {s.get("segment_id"): {**s, "flight_id": flight["flight_id"]}
              for platform in meta.values()
              for flight in platform.values()
              for s in flight["segments"]
             }
list_seg=list(segments.keys())

for ind in index:
    campaign=liste[ind]
       
    #herausfinden was es für hl gibt
    
    ds_mirac_a = cat['P5']['MIRAC_A'][campaign].to_dask()
    ds_cloud_top_height = cat['P5']['CLOUD_TOP_HEIGHT'][campaign].to_dask()
    ds_gps_origin = cat['P5']['GPS_INS'][campaign].to_dask()
    ds_sea_ice = cat['P5']['AMSR2_SIC'][campaign].to_dask()
    #spec_segment=campaign+'_hl01'
    #matching=[bla for bla in list_seg if campaign in bla]
    matching=[]
    for bla in list_seg:
     if campaign+'_hl' in str(bla):
        matching.append(bla)
        
        
    for spec_segment in matching:
     seg = segments[spec_segment]
    
     ds_mirac_a_sel = ds_mirac_a.sel(time=slice(seg["start"], seg["end"]))
     ds_cloud_top_height_sel = ds_cloud_top_height.sel(time=slice(seg["start"], seg["end"]))
     ds_gps = ds_gps_origin.sel(time=slice(seg["start"], seg["end"]))
     ds_sea_ice_sel = ds_sea_ice.sel(time=slice(seg["start"], seg["end"]))
     # Unterteilung in Eis und nur über Ocean 
     dsreduced = simplify_dataset(ds_gps, 1e-3)
     ocean=[]
     time=[]
     for x, y in zip(dsreduced.lon, dsreduced.lat):
         if il.is_land(x, y):
             print(str(x.time.values),' land')
             ocean.append(False)
             time.append(x.time.values)
         else:
             print(str(x.time.values),' ocean')
             ocean.append(True)
             time.append(x.time.values)
     #test=xr.DataArray(ocean, dims='time', )
     #ds_mirac_a_sel.assign(ocean=ocean)
     test=pd.DataFrame({'ocean':ocean})
     test.index=time
     test_2=pd.DataFrame()
     test_2.index=ds_mirac_a_sel.time
     test_2['ocean']=test
     ocean_mirac=test_2.ocean.fillna(method='ffill') #ocean or land for nall following nan
     test_2=pd.DataFrame()
     test_2.index=ds_cloud_top_height_sel.time
     test_2['ocean']=test
     ocean_amali=test_2.ocean.fillna(method='ffill') #ocean or land for nall following nan
     ocean_mirac=ocean_mirac.fillna(ocean_amali)
     ocean_amali=ocean_amali.fillna(ocean_mirac)
     ice=pd.Series(ds_sea_ice_sel.sic>90)
     ice.index=ds_sea_ice_sel.time
     name=''
            
     """
     #alle Werte auch über land
     ds_mirac_a_sel_amount=(ds_mirac_a_sel.Ze).count(axis=0)
     ds_mirac_a_sel_fraction=ds_mirac_a_sel_amount/len(ds_mirac_a_sel.time)
     stack = ds_cloud_top_height_sel.cloud_top_height.stack({'tl': ['time', 'cloud_layer']})
     
     #ds_mirac_a_sel_amount_time=(ds_mirac_a_sel.Ze[ds_mirac_a_sel.height>150]).count('height')
     #ds_mirac_a_sel_fraction_time=ds_mirac_a_sel_amount_time/len(ds_mirac_a_sel.height)
     ds_mirac_a_sel_amount_time=(ds_mirac_a_sel.Ze[:,ds_mirac_a_sel.height>150]).count('height')
     ds_mirac_a_sel_fraction_time=ds_mirac_a_sel_amount_time/sum(ds_mirac_a_sel.height>150)
     
     if str(ds_cloud_top_height_sel.cloud_top_height.max())=="<xarray.DataArray 'cloud_top_height' ()>\narray(nan)":
         print('no amali cloud top height')
         plt.style.use("/home/ischirma/pyac3airborne_book/pyac3airborne_book/mplstyle/book")
         fig, (ax1) = plt.subplots(1)
         ax1.plot(ds_mirac_a_sel_fraction,ds_mirac_a_sel.height*1e-3, color='k',label='MIRAC')
         ax1.set_title(spec_segment+' '+str(seg["start"])+'-'+str(seg["end"]))
         ax1.legend(frameon=False, loc='upper right')
         ax1.set_ylim(-0.25, 3.5)
         ax1.set_xlim(0., 1.)
         ax1.set_ylabel('Height [km]')
         ax1.set_xlabel('Signal fraction')
         for i, j in zip(range(-40,0,10), range(9,5,-1)):
             print(i,j)
             ds_mirac_a_sel_amount_20=((10*log10(ds_mirac_a_sel.Ze))>=i).sum(axis=0)
             ds_mirac_a_sel_fraction_20=ds_mirac_a_sel_amount_20/len(ds_mirac_a_sel.time)
             ax1.plot(ds_mirac_a_sel_fraction_20,ds_mirac_a_sel.height*1e-3,'--', color='k',alpha=j*0.1,label='MIRAC '+str(i))
             ax1.set_xlim(0., 1.)
         ax1.legend(frameon=False, loc='upper right')  
         ax1.axhline(y=0.15,linewidth=2, color='#d62728')
         plt.savefig('/home/ischirma/plots/CF_'+spec_segment+name, format='png')
         
         #cloud fraction nach Zeit nicht Höhe 
         bins_time=ds_cloud_top_height_sel.cloud_top_height.count('cloud_layer')
         ds_amali_a_sel_fraction_time=bins_time/len(ds_cloud_top_height_sel.cloud_layer)
         fig, (ax1) = plt.subplots(1)
         ax1.plot(ds_mirac_a_sel.time,ds_mirac_a_sel_fraction_time, color='k',label='MIRAC')
         ax1.plot(ds_cloud_top_height_sel.time,ds_amali_a_sel_fraction_time,'.', color='b',label='AMALI')
         ax1.set_title(spec_segment+' '+str(seg["start"])+'-'+str(seg["end"]))
         ax1.legend(frameon=False, loc='upper right')
         ax1.set_ylim(0., 1.)
         ax1.set_xlabel('Time (hh:mm) [UTC]')
         ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
         ax1.set_ylabel('Signal fraction')
         for i, j in zip(range(-40,0,10), range(9,5,-1)):
             print(i,j)
             ds_mirac_a_sel_amount_20_time=((10*log10(ds_mirac_a_sel.Ze))>=i).sum(axis=1)
             ds_mirac_a_sel_fraction_20_time=ds_mirac_a_sel_amount_20_time/len(ds_mirac_a_sel.height)
             ax1.plot(ds_mirac_a_sel.time,ds_mirac_a_sel_fraction_20_time,'--', color='k',alpha=j*0.1,label='MIRAC '+str(i))
             ax1.set_ylim(0., 1.)
         ax1.legend(frameon=False, loc='upper right')  
     else:    
         bins=stack.groupby_bins(group=stack*1e-3,bins=np.arange(0.,(ds_cloud_top_height_sel.cloud_top_height.max()*1e-3)+0.0050,0.0050)).count()
         ds_amali_a_sel_fraction=bins/len(ds_cloud_top_height_sel.time)

         bins_time=ds_cloud_top_height_sel.cloud_top_height.count('cloud_layer')
         ds_amali_a_sel_fraction_time=bins_time/len(ds_cloud_top_height_sel.cloud_layer)
         
         plt.style.use("/home/ischirma/pyac3airborne_book/pyac3airborne_book/mplstyle/book")
         fig, (ax1) = plt.subplots(1)
         ax1.plot(ds_mirac_a_sel_fraction,ds_mirac_a_sel.height*1e-3, color='k',label='MIRAC')
         ax1.plot(ds_amali_a_sel_fraction,np.arange(0.,(ds_cloud_top_height_sel.cloud_top_height.max()*1e-3),0.0050)+0.0025, color='b',label='AMALI')
         ax1.set_title(spec_segment+' '+str(seg["start"])+'-'+str(seg["end"]))
         ax1.legend(frameon=False, loc='upper right')
         ax1.set_ylim(-0.25, 3.5)
         ax1.set_xlim(0., 1.)
         ax1.set_ylabel('Height [km]')
         ax1.set_xlabel('Signal fraction')
         for i, j in zip(range(-40,0,10), range(9,5,-1)):
             print(i,j)           
             ds_mirac_a_sel_amount_20=((10*log10(ds_mirac_a_sel.Ze))>=i).sum(axis=0)
             ds_mirac_a_sel_fraction_20=ds_mirac_a_sel_amount_20/len(ds_mirac_a_sel.time)
             ax1.plot(ds_mirac_a_sel_fraction_20,ds_mirac_a_sel.height*1e-3,'--', color='k',alpha=j*0.1,label='MIRAC '+str(i))
             ax1.set_xlim(0., 1.)
         ax1.legend(frameon=False, loc='upper right')  
         ax1.axhline(y=0.15,linewidth=2, color='#d62728')
         plt.savefig('/home/ischirma/plots/CF_'+spec_segment+name, format='png')
         
         #cloud fraction nach Zeit nicht Höhe 
         fig, (ax1) = plt.subplots(1)
         ax1.plot(ds_mirac_a_sel.time,ds_mirac_a_sel_fraction_time, color='k',label='MIRAC')
         ax1.plot(ds_cloud_top_height_sel.time,ds_amali_a_sel_fraction_time,'.', color='b',label='AMALI')
         ax1.set_title(spec_segment+' '+str(seg["start"])+'-'+str(seg["end"]))
         ax1.legend(frameon=False, loc='upper right')
         ax1.set_ylim(0., 1.)
         ax1.set_xlabel('Time (hh:mm) [UTC]')
         ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
         ax1.set_ylabel('Signal fraction')
         for i, j in zip(range(-40,0,10), range(9,5,-1)):
             print(i,j)
             ds_mirac_a_sel_amount_20_time=((10*log10(ds_mirac_a_sel.Ze))>=i).sum(axis=1)
             ds_mirac_a_sel_fraction_20_time=ds_mirac_a_sel_amount_20_time/len(ds_mirac_a_sel.height)
             ax1.plot(ds_mirac_a_sel.time,ds_mirac_a_sel_fraction_20_time,'--', color='k',alpha=j*0.1,label='MIRAC '+str(i))
             ax1.set_ylim(0., 1.)
         ax1.legend(frameon=False, loc='upper right')  
         plt.savefig('/home/ischirma/plots/CF_time_'+spec_segment+name, format='png')
         """

    #"""
    #nur über ocean
    
    #wenn zusätlich nur über Ice oder nur über water: dann wird in ocean mirac und amali zusätzlich diese Info eingebunden
     ocean_mirac=(ocean_mirac*ice).fillna(0).astype('bool') #0 wenn eins falsch ist , 1 wenn beide richtig sind, nan wenn eins nicht vorhanden ist bei einem Zeitpunkt-> das mache ich dann auch zu 0: ich will alle 1
     ocean_amali=(ocean_amali*ice).fillna(0).astype('bool')
     ocean_mirac=ocean_mirac[ds_mirac_a_sel.time.values]
     ocean_amali=ocean_amali[ds_cloud_top_height_sel.time.values]
     name='ice'
     
     if sum(ocean_mirac>0):
      ds_mirac_a_sel_amount=(ds_mirac_a_sel.Ze[ocean_mirac,:]).count(axis=0)
      ds_mirac_a_sel_fraction=ds_mirac_a_sel_amount/len(ds_mirac_a_sel.time[np.array(ocean_mirac)])
      stack = ds_cloud_top_height_sel.cloud_top_height[ocean_amali,:].stack({'tl': ['time', 'cloud_layer']})
     
      #ds_mirac_a_sel_amount_time=(ds_mirac_a_sel.Ze[ds_mirac_a_sel.height>150]).count('height')
      #ds_mirac_a_sel_fraction_time=ds_mirac_a_sel_amount_time/len(ds_mirac_a_sel.height)
      ds_mirac_a_sel_amount_time=(ds_mirac_a_sel.Ze[ocean_mirac,ds_mirac_a_sel.height>150]).count('height')
      ds_mirac_a_sel_fraction_time=ds_mirac_a_sel_amount_time/sum(ds_mirac_a_sel.height>150)
     
      #if str(ds_cloud_top_height_sel.cloud_top_height.max())=="<xarray.DataArray 'cloud_top_height' ()>\narray(nan)":
      if str(stack.values.max())=='nan':
         print('no amali cloud top height')
         plt.style.use("/home/ischirma/pyac3airborne_book/pyac3airborne_book/mplstyle/book")
         fig, (ax1) = plt.subplots(1)
         ax1.plot(ds_mirac_a_sel_fraction,ds_mirac_a_sel.height*1e-3, color='k',label='MIRAC')
         ax1.set_title(spec_segment+' '+str(seg["start"])+'-'+str(seg["end"]))
         ax1.legend(frameon=False, loc='upper right')
         ax1.set_ylim(-0.25, 3.5)
         ax1.set_xlim(0., 1.)
         ax1.set_ylabel('Height [km]')
         ax1.set_xlabel('Signal fraction')
         for i, j in zip(range(-40,0,10), range(9,5,-1)):
             print(i,j)
             ds_mirac_a_sel_amount_20=((10*log10(ds_mirac_a_sel.Ze[ocean_mirac,:]))>=i).sum(axis=0)
             ds_mirac_a_sel_fraction_20=ds_mirac_a_sel_amount_20/len(ds_mirac_a_sel.time[np.array(ocean_mirac)])
             ax1.plot(ds_mirac_a_sel_fraction_20,ds_mirac_a_sel.height*1e-3,'--', color='k',alpha=j*0.1,label='MIRAC '+str(i))
             ax1.set_xlim(0., 1.)
         ax1.legend(frameon=False, loc='upper right')  
         ax1.axhline(y=0.15,linewidth=2, color='#d62728')
         plt.savefig('/home/ischirma/plots/CF_'+spec_segment+'_ocean'+name, format='png')
         
         #cloud fraction nach Zeit nicht Höhe 
         bins_time=ds_cloud_top_height_sel.cloud_top_height[ocean_amali,:].count('cloud_layer')
         ds_amali_a_sel_fraction_time=bins_time/len(ds_cloud_top_height_sel.cloud_layer)
         fig, (ax1) = plt.subplots(1)
         ax1.plot(ds_mirac_a_sel.time[np.array(ocean_mirac)],ds_mirac_a_sel_fraction_time,'.', color='k',label='MIRAC')
         ax1.plot(ds_cloud_top_height_sel.time[np.array(ocean_amali)],ds_amali_a_sel_fraction_time,'.', color='b',label='AMALI')
         ax1.set_title(spec_segment+' '+str(seg["start"])+'-'+str(seg["end"]))
         ax1.legend(frameon=False, loc='upper right')
         ax1.set_ylim(0., 1.)
         ax1.set_xlabel('Time (hh:mm) [UTC]')
         ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
         ax1.set_ylabel('Signal fraction')
         for i, j in zip(range(-40,0,10), range(9,5,-1)):
             print(i,j)
             ds_mirac_a_sel_amount_20_time=((10*log10(ds_mirac_a_sel.Ze[ocean_mirac,:]))>=i).sum(axis=1)
             ds_mirac_a_sel_fraction_20_time=ds_mirac_a_sel_amount_20_time/len(ds_mirac_a_sel.height)
             ax1.plot(ds_mirac_a_sel.time[np.array(ocean_mirac)],ds_mirac_a_sel_fraction_20_time,'+', color='k',alpha=j*0.1,label='MIRAC '+str(i))
             ax1.set_ylim(0., 1.)
         ax1.legend(frameon=False, loc='upper right')  
      else:    
         bins=stack.groupby_bins(group=stack*1e-3,bins=np.arange(0.,(ds_cloud_top_height_sel.cloud_top_height.max()*1e-3)+0.0050,0.0050)).count()# stack hat schon weniger time
         ds_amali_a_sel_fraction=bins/len(ds_cloud_top_height_sel.time[np.array(ocean_amali)])

         bins_time=ds_cloud_top_height_sel.cloud_top_height[ocean_amali,:].count('cloud_layer')
         ds_amali_a_sel_fraction_time=bins_time/len(ds_cloud_top_height_sel.cloud_layer)
         
         plt.style.use("/home/ischirma/pyac3airborne_book/pyac3airborne_book/mplstyle/book")
         fig, (ax1) = plt.subplots(1)
         ax1.plot(ds_mirac_a_sel_fraction,ds_mirac_a_sel.height*1e-3, color='k',label='MIRAC')
         ax1.plot(ds_amali_a_sel_fraction,np.arange(0.,(ds_cloud_top_height_sel.cloud_top_height.max()*1e-3),0.0050)+0.0025, color='b',label='AMALI')
         ax1.set_title(spec_segment+' '+str(seg["start"])+'-'+str(seg["end"]))
         ax1.legend(frameon=False, loc='upper right')
         ax1.set_ylim(-0.25, 3.5)
         ax1.set_xlim(0., 1.)
         ax1.set_ylabel('Height [km]')
         ax1.set_xlabel('Signal fraction')
         for i, j in zip(range(-40,0,10), range(9,5,-1)):
             print(i,j)
             ds_mirac_a_sel_amount_20=((10*log10(ds_mirac_a_sel.Ze[ocean_mirac,:]))>=i).sum(axis=0)
             ds_mirac_a_sel_fraction_20=ds_mirac_a_sel_amount_20/len(ds_mirac_a_sel.time[np.array(ocean_mirac)])
             ax1.plot(ds_mirac_a_sel_fraction_20,ds_mirac_a_sel.height*1e-3,'--', color='k',alpha=j*0.1,label='MIRAC '+str(i))
             ax1.set_xlim(0., 1.)
         ax1.legend(frameon=False, loc='upper right')  
         ax1.axhline(y=0.15,linewidth=2, color='#d62728')
         plt.savefig('/home/ischirma/plots/CF_'+spec_segment+'_ocean'+name, format='png')
         
         #cloud fraction nach Zeit nicht Höhe 
         fig, (ax1) = plt.subplots(1)
         ax1.plot(ds_mirac_a_sel.time[np.array(ocean_mirac)],ds_mirac_a_sel_fraction_time,'.', color='k',label='MIRAC')
         ax1.plot(ds_cloud_top_height_sel.time[np.array(ocean_amali)],ds_amali_a_sel_fraction_time,'.', color='b',label='AMALI')
         ax1.set_title(spec_segment+' '+str(seg["start"])+'-'+str(seg["end"]))
         ax1.legend(frameon=False, loc='upper right')
         ax1.set_ylim(0., 1.)
         ax1.set_xlabel('Time (hh:mm) [UTC]')
         ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
         ax1.set_ylabel('Signal fraction')
         for i, j in zip(range(-40,0,10), range(9,5,-1)):
             print(i,j)
             ds_mirac_a_sel_amount_20_time=((10*log10(ds_mirac_a_sel.Ze[ocean_mirac,:]))>=i).sum(axis=1)
             ds_mirac_a_sel_fraction_20_time=ds_mirac_a_sel_amount_20_time/len(ds_mirac_a_sel.height)
             ax1.plot(ds_mirac_a_sel.time[np.array(ocean_mirac)],ds_mirac_a_sel_fraction_20_time,'+', color='k',alpha=j*0.1,label='MIRAC '+str(i))
             ax1.set_ylim(0., 1.)
         ax1.legend(frameon=False, loc='upper right')  
         plt.savefig('/home/ischirma/plots/CF_time_'+spec_segment+'_ocean'+name, format='png')
         
#"""         