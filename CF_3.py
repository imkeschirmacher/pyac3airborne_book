#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 13:33:56 2021

@author: ischirma
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 14:15:08 2021

@author: ischirma
"""


#WICHTIG!!!!
#immer nur einen Abschnitt zum Plotten durchlaufen lassen. Wenn zb als erstes _ocean lief und sofort ocean_ice, dann ist ocean_ice falsch!!


import matplotlib.pyplot as plt
import ac3airborne
import numpy as np
from numpy import log10
import matplotlib.dates as mdates
import ac3airborne.tools.is_land as il
from simplification.cutil import simplify_coords_idx
import matplotlib
matplotlib.rcParams.update({'font.size': 18})
import pandas as pd

def simplify_dataset(ds, tolerance):
    indices_to_take = simplify_coords_idx(np.stack([ds.lat.values, ds.lon.values], axis=1), tolerance)
    return ds.isel(time=indices_to_take)


###############################################################################
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
     #bei AFLUX gibt es das problem dass mirac einmal 4 min später anfängt als alles andere! dESWEGEN MUSS TIME VON MIRAC WIE DIE ANDREN WERDEN UND MIT NAN AUFGEFÜLLT WERDEN

     # Unterteilung in Eis und nur über Ocean 
     #dsreduced = simplify_dataset(ds_gps, 1e-3)
     ocean=[]
     time=[]
     for x, y in zip(ds_gps.lon, ds_gps.lat):
         if il.is_land(x, y):
             print(str(x.time.values),' land')
             ocean.append(False)
             time.append(x.time.values)
         else:
             print(str(x.time.values),' ocean')
             ocean.append(True)
             time.append(x.time.values)

     test=pd.DataFrame({'ocean':ocean})
     test.index=time
     test_2=pd.DataFrame()
     test_2.index=ds_mirac_a_sel.time
     test_2['ocean']=test
     #problem dass bei AFLUX mirac mal 4 min später misst als segment: test 2 hat time von mirac
     ocean_mirac_old=test_2.ocean.fillna(method='ffill')#ocean or land for nall following nan
     test_2=pd.DataFrame()
     test_2.index=ds_cloud_top_height_sel.time
     test_2['ocean']=test
     ocean_amali=test_2.ocean.fillna(method='ffill') #ocean or land for nall following nan
     #problem amali ist schon früher als mirac
     ocean_mirac=ocean_mirac_old.fillna(ocean_amali) #trotzdem mirac zeit
     ocean_amali=ocean_amali.fillna(ocean_mirac_old)
     ice=pd.Series(ds_sea_ice_sel.sic>90)
     ice.index=ds_sea_ice_sel.time
     water=pd.Series(ds_sea_ice_sel.sic<15)
     water.index=ds_sea_ice_sel.time#ice und water früher als mirac
     name=''
###############################################################################      
     #alle Werte auch über land
     alt=ds_mirac_a_sel.altitude.mean()/1000
     ds_mirac_a_sel_amount=(ds_mirac_a_sel.Ze).count(axis=0)
     ds_mirac_a_sel_fraction=ds_mirac_a_sel_amount/len(ds_mirac_a_sel.time)
     stack = ds_cloud_top_height_sel.cloud_top_height.stack({'tl': ['time', 'cloud_layer']})
     #ds_mirac_a_sel_amount_time=(ds_mirac_a_sel.Ze[ds_mirac_a_sel.height>150]).count('height')
     #ds_mirac_a_sel_fraction_time=ds_mirac_a_sel_amount_time/len(ds_mirac_a_sel.height)
     ds_mirac_a_sel_amount_time=(ds_mirac_a_sel.Ze[:,ds_mirac_a_sel.height>150]).count('height')
     ds_mirac_a_sel_fraction_time=ds_mirac_a_sel_amount_time/sum(ds_mirac_a_sel.height>150)
     
     if len(ds_mirac_a_sel.Ze)>0:
      if str(ds_cloud_top_height_sel.cloud_top_height.max())=="<xarray.DataArray 'cloud_top_height' ()>\narray(nan)":
         print('no amali cloud top height')
         
         fig = plt.figure(figsize=(20,15))
         ax1=plt.subplot2grid((3,2),(0,1),rowspan=3)
         ax2=plt.subplot2grid((3,2),(0,0),colspan=1)
         ax3=plt.subplot2grid((3,2),(1,0),colspan=1)
         ax4=plt.subplot2grid((3,2),(2,0),colspan=1)
         
         ax1.plot(ds_mirac_a_sel_fraction[230:],ds_mirac_a_sel.height[230:]*1e-3, color='k',label='MiRAC')
         fig.suptitle(spec_segment+' '+str(seg["start"])+' - '+str(seg["end"])+'; alt: '+'%3.2f'%alt+' km')
         ax1.legend(frameon=False, loc='upper right', fontsize=12)
         ax1.set_ylim(0., 3.5)
         ax1.set_xlim(0., 1.)
         ax1.set_ylabel('Height/ km')
         ax1.set_xlabel('Signal fraction')
         for i, j in zip(range(-40,0,10), range(9,5,-1)):
             print(i,j)
             ds_mirac_a_sel_amount_20=((10*log10(ds_mirac_a_sel.Ze))>=i).sum(axis=0)
             ds_mirac_a_sel_fraction_20=ds_mirac_a_sel_amount_20/len(ds_mirac_a_sel.time)
             ax1.plot(ds_mirac_a_sel_fraction_20[230:],ds_mirac_a_sel.height[230:]*1e-3,'--', color='k',alpha=j*0.1,label='MiRAC '+str(i))
             ax1.set_xlim(0., 1.)
         ax1.legend(frameon=False, loc='best', fontsize=12)  
                  
         immirac = ax2.pcolormesh(ds_mirac_a_sel.time, ds_mirac_a_sel.height*1e-3, 10*log10(
         ds_mirac_a_sel.Ze).T, vmin=-40, vmax=30, cmap='jet', shading='flat')
         fig.colorbar(immirac, ax=ax2, label='Radar reflectivity / dBz')
         ax2.set_ylim(0., 3.5)
         ax2.set_ylabel('Height / km')
         #ax2.set_xlabel('Time (hh:mm) / UTC')
         #ax2.xaxis.set_ticks([str(ds_mirac_a_sel.time[0]), str(ds_mirac_a_sel.time[int(len(ds_mirac_a_sel.time)/2)]), str(ds_mirac_a_sel.time[-1])])
         #ax2.set_xticklabels([ds_mirac_a_sel.time[0], ds_mirac_a_sel.time[int(len(ds_mirac_a_sel.time)/2)], ds_mirac_a_sel.time[-1]], rotation=0)
         ax2.xaxis.set_ticks(ds_mirac_a_sel.time[::int(len(ds_mirac_a_sel.time)/3)].values)
         ax2.set_xticklabels(ds_mirac_a_sel.time[::int(len(ds_mirac_a_sel.time)/3)].values, rotation=0)
         ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
         
         #imamali = ax3.scatter(x=stack.time, y=stack*1e-3, c=stack.cloud_layer, s=2, vmin=1, vmax=9, cmap='Set1')
         #fig.colorbar(imamali, ax=ax3, label='cloud layer')
         ax3.plot(stack.time, stack*1e-3,'.', markersize=1)
         ax3.set_ylim(0, 3.5)
         ax3.set_ylabel('Cloud top height / km')
         #ax3.set_xlabel('Time (hh:mm) / UTC')
         ax3.xaxis.set_ticks(stack.time[::int(len(stack.time)/3)].values)
         ax3.set_xticklabels(stack.time[::int(len(stack.time)/3)].values, rotation=0)
         ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))     
         
         ax4.plot(test[test==False].index,test[test==False].ocean+1, '.', color='brown', label='land')
         ax4.plot(test[test==True].index,test[test==True].ocean, '.', color='b', label='ocean')
         ax4.plot(water[water==True].index,water[water==True]*0.9, '.', color='lightblue', label='sea ice fraction < 15%')
         ax4.plot(ice[ice==True].index,ice[ice==True]*0.9, '.', color='gray', label='sea ice fraction > 90%')
         #ax4.scatter(test[test=='False'].index,test[test=='False'].ocean, s=2, color='gray')
         #ax4.xaxis.set_ticks([stack.time[0], stack.time[int(len(stack.time)/2)], stack.time[-1]])
         ax4.set_ylim(0.89,1.01)
         ax4.set_yticklabels([])
         ax4.set_yticks([])
         ax4.set_xlabel('Time (hh:mm) / UTC')
         ax4.legend(frameon=False, loc='best', fontsize=12)  
         #ax4.set_xticklabels([test.index[0], test.index[int(len(test.index)/2)], test.index[-1]], rotation=0)
         if int(len(test.index)/3) >0:
          ax4.xaxis.set_ticks(test.index[::int(len(test.index)/3)].values)
          ax4.set_xticklabels(test.index[::int(len(test.index)/3)].values, rotation=0)
         else:
          ax4.xaxis.set_ticks(test.index[:].values)
          ax4.set_xticklabels(test.index[:].values, rotation=0)
         ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')) 
         ##plt.tight_layout()
         plt.savefig('/home/ischirma/plots/CF_'+spec_segment+name, format='png', bbox_inches = "tight")
         
         
         #cloud fraction nach Zeit nicht Höhe 
         bins_time=ds_cloud_top_height_sel.cloud_top_height.count('cloud_layer')
         ds_amali_a_sel_fraction_time=bins_time/len(ds_cloud_top_height_sel.cloud_layer)
         fig, (ax1) = plt.subplots(1)
         ax1.plot(ds_mirac_a_sel.time,ds_mirac_a_sel_fraction_time, color='k',label='MiRAC')
         ax1.plot(ds_cloud_top_height_sel.time,ds_amali_a_sel_fraction_time,'.', color='b',label='AMALI')
         ax1.set_title(spec_segment+' '+str(seg["start"])+' - '+str(seg["end"])+'\nalt: '+'%3.2f'%alt+' km')
         ax1.legend(frameon=False, loc='upper right')
         ax1.set_ylim(0., 1.)
         ax1.set_xlabel('Time (hh:mm) [UTC]')
         ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
         ax1.set_ylabel('Signal fraction')
         for i, j in zip(range(-40,0,10), range(9,5,-1)):
             print(i,j)
             ds_mirac_a_sel_amount_20_time=((10*log10(ds_mirac_a_sel.Ze))>=i).sum(axis=1)
             ds_mirac_a_sel_fraction_20_time=ds_mirac_a_sel_amount_20_time/len(ds_mirac_a_sel.height)
             ax1.plot(ds_mirac_a_sel.time,ds_mirac_a_sel_fraction_20_time,'--', color='k',alpha=j*0.1,label='MiRAC '+str(i))
             ax1.set_ylim(0., 1.)
         ax1.legend(frameon=False, loc='upper right')   
         #plt.tight_layout()
         plt.savefig('/home/ischirma/plots/CF_time_'+spec_segment+name, format='png', bbox_inches = "tight")
      else:    
         bins=stack.groupby_bins(group=stack*1e-3,bins=np.arange(0.,(ds_cloud_top_height_sel.cloud_top_height.max()*1e-3)+0.0050,0.0050)).count()
         ds_amali_a_sel_fraction=bins/len(ds_cloud_top_height_sel.time)

         bins_time=ds_cloud_top_height_sel.cloud_top_height.count('cloud_layer')
         ds_amali_a_sel_fraction_time=bins_time/len(ds_cloud_top_height_sel.cloud_layer)
         
         fig = plt.figure(figsize=(20,15))
         ax1=plt.subplot2grid((3,2),(0,1),rowspan=3)
         ax2=plt.subplot2grid((3,2),(0,0),colspan=1)
         ax3=plt.subplot2grid((3,2),(1,0),colspan=1)
         ax4=plt.subplot2grid((3,2),(2,0),colspan=1)
         
         ax1.plot(ds_mirac_a_sel_fraction[230:],ds_mirac_a_sel.height[230:]*1e-3, color='k',label='MiRAC')
         ax1.plot(ds_amali_a_sel_fraction,np.arange(0.,(ds_cloud_top_height_sel.cloud_top_height.max()*1e-3),0.0050)+0.0025, color='b',label='AMALI')
         fig.suptitle(spec_segment+' '+str(seg["start"])+' - '+str(seg["end"])+'; alt: '+'%3.2f'%alt+' km')
         ax1.legend(frameon=False, loc='upper right', fontsize=12)
         ax1.set_ylim(0., 3.5)
         ax1.set_xlim(0., 1.)
         ax1.set_ylabel('Height/ km')
         ax1.set_xlabel('Signal fraction')
         for i, j in zip(range(-40,0,10), range(9,5,-1)):
             print(i,j)
             ds_mirac_a_sel_amount_20=((10*log10(ds_mirac_a_sel.Ze))>=i).sum(axis=0)
             ds_mirac_a_sel_fraction_20=ds_mirac_a_sel_amount_20/len(ds_mirac_a_sel.time)
             ax1.plot(ds_mirac_a_sel_fraction_20[230:],ds_mirac_a_sel.height[230:]*1e-3,'--', color='k',alpha=j*0.1,label='MiRAC '+str(i))
             ax1.set_xlim(0., 1.)
         ax1.legend(frameon=False, loc='best', fontsize=12)  
         
         
         immirac = ax2.pcolormesh(ds_mirac_a_sel.time, ds_mirac_a_sel.height*1e-3, 10*log10(
         ds_mirac_a_sel.Ze).T, vmin=-40, vmax=30, cmap='jet', shading='flat')
         fig.colorbar(immirac, ax=ax2, label='Radar reflectivity / dBz')
         ax2.set_ylim(0., 3.5)
         ax2.set_ylabel('Height / km')
         #ax2.set_xlabel('Time (hh:mm) / UTC')
         #ax2.xaxis.set_ticks([str(ds_mirac_a_sel.time[0]), str(ds_mirac_a_sel.time[int(len(ds_mirac_a_sel.time)/2)]), str(ds_mirac_a_sel.time[-1])])
         #ax2.set_xticklabels([ds_mirac_a_sel.time[0], ds_mirac_a_sel.time[int(len(ds_mirac_a_sel.time)/2)], ds_mirac_a_sel.time[-1]], rotation=0)
         ax2.xaxis.set_ticks(ds_mirac_a_sel.time[::int(len(ds_mirac_a_sel.time)/3)].values)
         ax2.set_xticklabels(ds_mirac_a_sel.time[::int(len(ds_mirac_a_sel.time)/3)].values, rotation=0)
         ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
         
         #imamali = ax3.scatter(x=stack.time, y=stack*1e-3, c=stack.cloud_layer, s=2, vmin=1, vmax=9, cmap='Set1')
         #fig.colorbar(imamali, ax=ax3, label='cloud layer')
         ax3.plot(stack.time, stack*1e-3,'.', markersize=1)
         ax3.set_ylim(0, 3.5)
         ax3.set_ylabel('Cloud top height / km')
         #ax3.set_xlabel('Time (hh:mm) / UTC')
         ax3.xaxis.set_ticks(stack.time[::int(len(stack.time)/3)].values)
         ax3.set_xticklabels(stack.time[::int(len(stack.time)/3)].values, rotation=0)
         ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))     
         
         ax4.plot(test[test==False].index,test[test==False].ocean+1, '.', color='brown', label='land')
         ax4.plot(test[test==True].index,test[test==True].ocean, '.', color='b', label='ocean')
         ax4.plot(water[water==True].index,water[water==True]*0.9, '.', color='lightblue', label='sea ice fraction < 15%')
         ax4.plot(ice[ice==True].index,ice[ice==True]*0.9, '.', color='gray', label='sea ice fraction > 90%')
         #ax4.scatter(test[test=='False'].index,test[test=='False'].ocean, s=2, color='gray')
         #ax4.xaxis.set_ticks([stack.time[0], stack.time[int(len(stack.time)/2)], stack.time[-1]])
         ax4.set_ylim(0.89,1.01)
         ax4.set_yticklabels([])
         ax4.set_yticks([])
         ax4.set_xlabel('Time (hh:mm) / UTC')
         ax4.legend(frameon=False, loc='best', fontsize=12)  
         #ax4.set_xticklabels([test.index[0], test.index[int(len(test.index)/2)], test.index[-1]], rotation=0)
         if int(len(test.index)/3) >0:
          ax4.xaxis.set_ticks(test.index[::int(len(test.index)/3)].values)
          ax4.set_xticklabels(test.index[::int(len(test.index)/3)].values, rotation=0)
         else:
          ax4.xaxis.set_ticks(test.index[:].values)
          ax4.set_xticklabels(test.index[:].values, rotation=0)             
         ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')) 
         ##plt.tight_layout()
         plt.savefig('/home/ischirma/plots/CF_'+spec_segment+name, format='png', bbox_inches = "tight")
         
         
         #cloud fraction nach Zeit nicht Höhe 
         fig, (ax1) = plt.subplots(1)
         ax1.plot(ds_mirac_a_sel.time,ds_mirac_a_sel_fraction_time, color='k',label='MiRAC')
         ax1.plot(ds_cloud_top_height_sel.time,ds_amali_a_sel_fraction_time,'.', color='b',label='AMALI')
         ax1.set_title(spec_segment+' '+str(seg["start"])+' - '+str(seg["end"])+'\nalt: '+'%3.2f'%alt+' km')
         ax1.legend(frameon=False, loc='upper right')
         ax1.set_ylim(0., 1.)
         ax1.set_xlabel('Time (hh:mm) [UTC]')
         ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
         ax1.set_ylabel('Signal fraction')
         for i, j in zip(range(-40,0,10), range(9,5,-1)):
             print(i,j)
             ds_mirac_a_sel_amount_20_time=((10*log10(ds_mirac_a_sel.Ze))>=i).sum(axis=1)
             ds_mirac_a_sel_fraction_20_time=ds_mirac_a_sel_amount_20_time/len(ds_mirac_a_sel.height)
             ax1.plot(ds_mirac_a_sel.time,ds_mirac_a_sel_fraction_20_time,'--', color='k',alpha=j*0.1,label='MiRAC '+str(i))
             ax1.set_ylim(0., 1.)
         ax1.legend(frameon=False, loc='upper right')  
         #plt.tight_layout()
         plt.savefig('/home/ischirma/plots/CF_time_'+spec_segment+name, format='png', bbox_inches = "tight")
        
    
###############################################################################
  
     
     #nur über ocean
     name=''
     if sum(ocean_mirac>0) & len(ds_mirac_a_sel.Ze)>0:
      alt=(ds_mirac_a_sel.altitude[np.array(ocean_mirac)]).mean()/1000   
      ds_mirac_a_sel_amount=(ds_mirac_a_sel.Ze[ocean_mirac,:]).count(axis=0)
      ds_mirac_a_sel_fraction=ds_mirac_a_sel_amount/len(ds_mirac_a_sel.time[np.array(ocean_mirac)])
      stack = ds_cloud_top_height_sel.cloud_top_height[ocean_amali,:].stack({'tl': ['time', 'cloud_layer']})
     
      #ds_mirac_a_sel_amount_time=(ds_mirac_a_sel.Ze[ds_mirac_a_sel.height>150]).count('height')
      #ds_mirac_a_sel_fraction_time=ds_mirac_a_sel_amount_time/len(ds_mirac_a_sel.height)
      ds_mirac_a_sel_amount_time=(ds_mirac_a_sel.Ze[ocean_mirac,ds_mirac_a_sel.height>150]).count('height')
      ds_mirac_a_sel_fraction_time=ds_mirac_a_sel_amount_time/sum(ds_mirac_a_sel.height>150)
     
      #if str(ds_cloud_top_height_sel.cloud_top_height.max())=="<xarray.DataArray 'cloud_top_height' ()>\narray(nan)":
      if str(stack.max())=="<xarray.DataArray 'cloud_top_height' ()>\narray(nan)":
         print('no amali cloud top height')
         fig, (ax1) = plt.subplots(1)
         ax1.plot(ds_mirac_a_sel_fraction[230:],ds_mirac_a_sel.height[230:]*1e-3, color='k',label='MiRAC')
         ax1.set_title(spec_segment+' '+str(seg["start"])+' - '+str(seg["end"])+'\nalt: '+'%3.2f'%alt+' km')
         ax1.legend(frameon=False, loc='upper right')
         ax1.set_ylim(0, 3.5)
         ax1.set_xlim(0., 1.)
         ax1.set_ylabel('Height /  km')
         ax1.set_xlabel('Signal fraction')
         for i, j in zip(range(-40,0,10), range(9,5,-1)):
             print(i,j)
             ds_mirac_a_sel_amount_20=((10*log10(ds_mirac_a_sel.Ze[ocean_mirac,:]))>=i).sum(axis=0)
             ds_mirac_a_sel_fraction_20=ds_mirac_a_sel_amount_20/len(ds_mirac_a_sel.time[np.array(ocean_mirac)])
             ax1.plot(ds_mirac_a_sel_fraction_20[230:],ds_mirac_a_sel.height[230:]*1e-3,'--', color='k',alpha=j*0.1,label='MiRAC '+str(i))
             ax1.set_xlim(0., 1.)
         ax1.legend(frameon=False, loc='upper right')  
         #ax1.axhline(y=0.15,linewidth=2, color='#d62728')
         #plt.tight_layout()
         plt.savefig('/home/ischirma/plots/CF_'+spec_segment+'_ocean'+name, format='png', bbox_inches = "tight")
         
         
         #cloud fraction nach Zeit nicht Höhe 
         bins_time=ds_cloud_top_height_sel.cloud_top_height[ocean_amali,:].count('cloud_layer')
         ds_amali_a_sel_fraction_time=bins_time/len(ds_cloud_top_height_sel.cloud_layer)
         fig, (ax1) = plt.subplots(1)
         ax1.plot(ds_mirac_a_sel.time[np.array(ocean_mirac)],ds_mirac_a_sel_fraction_time,'.', color='k',label='MiRAC')
         ax1.plot(ds_cloud_top_height_sel.time[np.array(ocean_amali)],ds_amali_a_sel_fraction_time,'.', color='b',label='AMALI')
         ax1.set_title(spec_segment+' '+str(seg["start"])+' - '+str(seg["end"])+'\nalt: '+'%3.2f'%alt+' km')
         ax1.legend(frameon=False, loc='upper right')
         ax1.set_ylim(0., 1.)
         ax1.set_xlabel('Time (hh:mm) [UTC]')
         ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
         ax1.set_ylabel('Signal fraction')
         for i, j in zip(range(-40,0,10), range(9,5,-1)):
             print(i,j)
             ds_mirac_a_sel_amount_20_time=((10*log10(ds_mirac_a_sel.Ze[ocean_mirac,:]))>=i).sum(axis=1)
             ds_mirac_a_sel_fraction_20_time=ds_mirac_a_sel_amount_20_time/len(ds_mirac_a_sel.height)
             ax1.plot(ds_mirac_a_sel.time[np.array(ocean_mirac)],ds_mirac_a_sel_fraction_20_time,'+', color='k',alpha=j*0.1,label='MiRAC '+str(i))
             ax1.set_ylim(0., 1.)
         ax1.legend(frameon=False, loc='upper right')  
      else:    
         bins=stack.groupby_bins(group=stack*1e-3,bins=np.arange(0.,(ds_cloud_top_height_sel.cloud_top_height.max()*1e-3)+0.0050,0.0050)).count()# stack hat schon weniger time
         ds_amali_a_sel_fraction=bins/len(ds_cloud_top_height_sel.time[np.array(ocean_amali)])

         bins_time=ds_cloud_top_height_sel.cloud_top_height[ocean_amali,:].count('cloud_layer')
         ds_amali_a_sel_fraction_time=bins_time/len(ds_cloud_top_height_sel.cloud_layer)
         
         fig, (ax1) = plt.subplots(1)
         ax1.plot(ds_mirac_a_sel_fraction[230:],ds_mirac_a_sel.height[230:]*1e-3, color='k',label='MiRAC')
         ax1.plot(ds_amali_a_sel_fraction,np.arange(0.,(ds_cloud_top_height_sel.cloud_top_height.max()*1e-3),0.0050)+0.0025, color='b',label='AMALI')
         ax1.set_title(spec_segment+' '+str(seg["start"])+' - '+str(seg["end"])+'\nalt: '+'%3.2f'%alt+' km')
         ax1.legend(frameon=False, loc='upper right')
         ax1.set_ylim(0, 3.5)
         ax1.set_xlim(0., 1.)
         ax1.set_ylabel('Height /  km')
         ax1.set_xlabel('Signal fraction')
         for i, j in zip(range(-40,0,10), range(9,5,-1)):
             print(i,j)
             ds_mirac_a_sel_amount_20=((10*log10(ds_mirac_a_sel.Ze[ocean_mirac,:]))>=i).sum(axis=0)
             ds_mirac_a_sel_fraction_20=ds_mirac_a_sel_amount_20/len(ds_mirac_a_sel.time[np.array(ocean_mirac)])
             ax1.plot(ds_mirac_a_sel_fraction_20[230:],ds_mirac_a_sel.height[230:]*1e-3,'--', color='k',alpha=j*0.1,label='MiRAC '+str(i))
             ax1.set_xlim(0., 1.)
         ax1.legend(frameon=False, loc='upper right')  
         #ax1.axhline(y=0.15,linewidth=2, color='#d62728')
         #plt.tight_layout()
         plt.savefig('/home/ischirma/plots/CF_'+spec_segment+'_ocean'+name, format='png', bbox_inches = "tight")
         

############################################################################### 
     
     #wenn zusätlich nur über Ice oder nur über water: dann wird in ocean mirac und amali zusätzlich diese Info eingebunden
     #nur über wasser wo gleichzeitig ice     
     ocean_mirac_ice=(ocean_mirac*ice).fillna(0).astype('bool') #0 wenn eins falsch ist , 1 wenn beide richtig sind, nan wenn eins nicht vorhanden ist bei einem Zeitpunkt-> das mache ich dann auch zu 0: ich will alle 1
     ocean_amali_ice=(ocean_amali*ice).fillna(0).astype('bool')
     ocean_mirac_ice=ocean_mirac_ice[ds_mirac_a_sel.time.values]
     ocean_amali_ice=ocean_amali_ice[ds_cloud_top_height_sel.time.values]
     name='ice'
     
     if sum(ocean_mirac_ice>0):
       alt=ds_mirac_a_sel.altitude[np.array(ocean_mirac_ice)].mean()/1000  
       ds_mirac_a_sel_amount=(ds_mirac_a_sel.Ze[ocean_mirac_ice,:]).count(axis=0)
       ds_mirac_a_sel_fraction=ds_mirac_a_sel_amount/len(ds_mirac_a_sel.time[np.array(ocean_mirac_ice)])
       stack = ds_cloud_top_height_sel.cloud_top_height[ocean_amali_ice,:].stack({'tl': ['time', 'cloud_layer']})
     
       #ds_mirac_a_sel_amount_time=(ds_mirac_a_sel.Ze[ds_mirac_a_sel.height>150]).count('height')
       #ds_mirac_a_sel_fraction_time=ds_mirac_a_sel_amount_time/len(ds_mirac_a_sel.height)
       ds_mirac_a_sel_amount_time=(ds_mirac_a_sel.Ze[ocean_mirac_ice,ds_mirac_a_sel.height>150]).count('height')
       ds_mirac_a_sel_fraction_time=ds_mirac_a_sel_amount_time/sum(ds_mirac_a_sel.height>150)
     
      #if str(ds_cloud_top_height_sel.cloud_top_height.max())=="<xarray.DataArray 'cloud_top_height' ()>\narray(nan)":
       if str(stack.max())=="<xarray.DataArray 'cloud_top_height' ()>\narray(nan)":
         print('no amali cloud top height')
         fig, (ax1) = plt.subplots(1)
         ax1.plot(ds_mirac_a_sel_fraction[230:],ds_mirac_a_sel.height[230:]*1e-3, color='k',label='MiRAC')
         ax1.set_title(spec_segment+' '+str(seg["start"])+' - '+str(seg["end"])+'\nalt: '+'%3.2f'%alt+' km')
         ax1.legend(frameon=False, loc='upper right')
         ax1.set_ylim(0, 3.5)
         ax1.set_xlim(0., 1.)
         ax1.set_ylabel('Height /  km')
         ax1.set_xlabel('Signal fraction')
         for i, j in zip(range(-40,0,10), range(9,5,-1)):
             print(i,j)
             ds_mirac_a_sel_amount_20=((10*log10(ds_mirac_a_sel.Ze[ocean_mirac_ice,:]))>=i).sum(axis=0)
             ds_mirac_a_sel_fraction_20=ds_mirac_a_sel_amount_20/len(ds_mirac_a_sel.time[np.array(ocean_mirac_ice)])
             ax1.plot(ds_mirac_a_sel_fraction_20[230:],ds_mirac_a_sel.height[230:]*1e-3,'--', color='k',alpha=j*0.1,label='MiRAC '+str(i))
             ax1.set_xlim(0., 1.)
         ax1.legend(frameon=False, loc='upper right')  
         #ax1.axhline(y=0.15,linewidth=2, color='#d62728')
         #plt.tight_layout()
         plt.savefig('/home/ischirma/plots/CF_'+spec_segment+'_ocean'+name, format='png', bbox_inches = "tight")
         
         
         #cloud fraction nach Zeit nicht Höhe 
         bins_time=ds_cloud_top_height_sel.cloud_top_height[ocean_amali_ice,:].count('cloud_layer')
         ds_amali_a_sel_fraction_time=bins_time/len(ds_cloud_top_height_sel.cloud_layer)
         fig, (ax1) = plt.subplots(1)
         ax1.plot(ds_mirac_a_sel.time[np.array(ocean_mirac_ice)],ds_mirac_a_sel_fraction_time,'.', color='k',label='MiRAC')
         ax1.plot(ds_cloud_top_height_sel.time[np.array(ocean_amali_ice)],ds_amali_a_sel_fraction_time,'.', color='b',label='AMALI')
         ax1.set_title(spec_segment+' '+str(seg["start"])+' - '+str(seg["end"])+'\nalt: '+'%3.2f'%alt+' km')
         ax1.legend(frameon=False, loc='upper right')
         ax1.set_ylim(0., 1.)
         ax1.set_xlabel('Time (hh:mm) [UTC]')
         ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
         ax1.set_ylabel('Signal fraction')
         for i, j in zip(range(-40,0,10), range(9,5,-1)):
             print(i,j)
             ds_mirac_a_sel_amount_20_time=((10*log10(ds_mirac_a_sel.Ze[ocean_mirac_ice,:]))>=i).sum(axis=1)
             ds_mirac_a_sel_fraction_20_time=ds_mirac_a_sel_amount_20_time/len(ds_mirac_a_sel.height)
             ax1.plot(ds_mirac_a_sel.time[np.array(ocean_mirac_ice)],ds_mirac_a_sel_fraction_20_time,'+', color='k',alpha=j*0.1,label='MiRAC '+str(i))
             ax1.set_ylim(0., 1.)
         ax1.legend(frameon=False, loc='upper right')  
       else:    
         bins=stack.groupby_bins(group=stack*1e-3,bins=np.arange(0.,(ds_cloud_top_height_sel.cloud_top_height.max()*1e-3)+0.0050,0.0050)).count()# stack hat schon weniger time
         ds_amali_a_sel_fraction=bins/len(ds_cloud_top_height_sel.time[np.array(ocean_amali_ice)])

         bins_time=ds_cloud_top_height_sel.cloud_top_height[ocean_amali_ice,:].count('cloud_layer')
         ds_amali_a_sel_fraction_time=bins_time/len(ds_cloud_top_height_sel.cloud_layer)
         
         fig, (ax1) = plt.subplots(1)
         ax1.plot(ds_mirac_a_sel_fraction[230:],ds_mirac_a_sel.height[230:]*1e-3, color='k',label='MiRAC')
         ax1.plot(ds_amali_a_sel_fraction,np.arange(0.,(ds_cloud_top_height_sel.cloud_top_height.max()*1e-3),0.0050)+0.0025, color='b',label='AMALI')
         ax1.set_title(spec_segment+' '+str(seg["start"])+' - '+str(seg["end"])+'\nalt: '+'%3.2f'%alt+' km')
         ax1.legend(frameon=False, loc='upper right')
         ax1.set_ylim(0, 3.5)
         ax1.set_xlim(0., 1.)
         ax1.set_ylabel('Height /  km')
         ax1.set_xlabel('Signal fraction')
         for i, j in zip(range(-40,0,10), range(9,5,-1)):
             print(i,j)
             ds_mirac_a_sel_amount_20=((10*log10(ds_mirac_a_sel.Ze[ocean_mirac_ice,:]))>=i).sum(axis=0)
             ds_mirac_a_sel_fraction_20=ds_mirac_a_sel_amount_20/len(ds_mirac_a_sel.time[np.array(ocean_mirac_ice)])
             ax1.plot(ds_mirac_a_sel_fraction_20[230:],ds_mirac_a_sel.height[230:]*1e-3,'--', color='k',alpha=j*0.1,label='MiRAC '+str(i))
             ax1.set_xlim(0., 1.)
         ax1.legend(frameon=False, loc='upper right')  
         #ax1.axhline(y=0.15,linewidth=2, color='#d62728')
         #plt.tight_layout()
         plt.savefig('/home/ischirma/plots/CF_'+spec_segment+'_ocean'+name, format='png', bbox_inches = "tight")
         
         
         #cloud fraction nach Zeit nicht Höhe 
         fig, (ax1) = plt.subplots(1)
         ax1.plot(ds_mirac_a_sel.time[np.array(ocean_mirac_ice)],ds_mirac_a_sel_fraction_time,'.', color='k',label='MiRAC')
         ax1.plot(ds_cloud_top_height_sel.time[np.array(ocean_amali_ice)],ds_amali_a_sel_fraction_time,'.', color='b',label='AMALI')
         ax1.set_title(spec_segment+' '+str(seg["start"])+' - '+str(seg["end"])+'\nalt: '+'%3.2f'%alt+' km')
         ax1.legend(frameon=False, loc='upper right')
         ax1.set_ylim(0., 1.)
         ax1.set_xlabel('Time (hh:mm) [UTC]')
         ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
         ax1.set_ylabel('Signal fraction')
         for i, j in zip(range(-40,0,10), range(9,5,-1)):
             print(i,j)
             ds_mirac_a_sel_amount_20_time=((10*log10(ds_mirac_a_sel.Ze[ocean_mirac_ice,:]))>=i).sum(axis=1)
             ds_mirac_a_sel_fraction_20_time=ds_mirac_a_sel_amount_20_time/len(ds_mirac_a_sel.height)
             ax1.plot(ds_mirac_a_sel.time[np.array(ocean_mirac_ice)],ds_mirac_a_sel_fraction_20_time,'+', color='k',alpha=j*0.1,label='MiRAC '+str(i))
             ax1.set_ylim(0., 1.)
         ax1.legend(frameon=False, loc='upper right')  
         #plt.tight_layout()
         plt.savefig('/home/ischirma/plots/CF_time_'+spec_segment+'_ocean'+name, format='png', bbox_inches = "tight")
         
     
###############################################################################        
    #nur über water and free water no ice
     name='ocean'
     ocean_mirac_water=(ocean_mirac*water).fillna(0).astype('bool') #0 wenn eins falsch ist , 1 wenn beide richtig sind, nan wenn eins nicht vorhanden ist bei einem Zeitpunkt-> das mache ich dann auch zu 0: ich will alle 1
     ocean_amali_water=(ocean_amali*water).fillna(0).astype('bool')
     ocean_mirac_water=ocean_mirac_water[ds_mirac_a_sel.time.values]
     ocean_amali_water=ocean_amali_water[ds_cloud_top_height_sel.time.values]

     if sum(ocean_mirac_water>0):
       alt=ds_mirac_a_sel.altitude[np.array(ocean_mirac_water)].mean()/1000  
       ds_mirac_a_sel_amount=(ds_mirac_a_sel.Ze[ocean_mirac_water,:]).count(axis=0)
       ds_mirac_a_sel_fraction=ds_mirac_a_sel_amount/len(ds_mirac_a_sel.time[np.array(ocean_mirac_water)])
       stack = ds_cloud_top_height_sel.cloud_top_height[ocean_amali_water,:].stack({'tl': ['time', 'cloud_layer']})
     
       #ds_mirac_a_sel_amount_time=(ds_mirac_a_sel.Ze[ds_mirac_a_sel.height>150]).count('height')
       #ds_mirac_a_sel_fraction_time=ds_mirac_a_sel_amount_time/len(ds_mirac_a_sel.height)
       ds_mirac_a_sel_amount_time=(ds_mirac_a_sel.Ze[ocean_mirac_water,ds_mirac_a_sel.height>150]).count('height')
       ds_mirac_a_sel_fraction_time=ds_mirac_a_sel_amount_time/sum(ds_mirac_a_sel.height>150)
     
      #if str(ds_cloud_top_height_sel.cloud_top_height.max())=="<xarray.DataArray 'cloud_top_height' ()>\narray(nan)":
       if str(stack.max())=="<xarray.DataArray 'cloud_top_height' ()>\narray(nan)":
         print('no amali cloud top height')
         fig, (ax1) = plt.subplots(1)
         ax1.plot(ds_mirac_a_sel_fraction[230:],ds_mirac_a_sel.height[230:]*1e-3, color='k',label='MiRAC')
         ax1.set_title(spec_segment+' '+str(seg["start"])+' - '+str(seg["end"])+'\nalt: '+'%3.2f'%alt+' km')
         ax1.legend(frameon=False, loc='upper right')
         ax1.set_ylim(0, 3.5)
         ax1.set_xlim(0., 1.)
         ax1.set_ylabel('Height /  km')
         ax1.set_xlabel('Signal fraction')
         for i, j in zip(range(-40,0,10), range(9,5,-1)):
             print(i,j)
             ds_mirac_a_sel_amount_20=((10*log10(ds_mirac_a_sel.Ze[ocean_mirac_water,:]))>=i).sum(axis=0)
             ds_mirac_a_sel_fraction_20=ds_mirac_a_sel_amount_20/len(ds_mirac_a_sel.time[np.array(ocean_mirac_water)])
             ax1.plot(ds_mirac_a_sel_fraction_20[230:],ds_mirac_a_sel.height[230:]*1e-3,'--', color='k',alpha=j*0.1,label='MiRAC '+str(i))
             ax1.set_xlim(0., 1.)
         ax1.legend(frameon=False, loc='upper right')  
         #ax1.axhline(y=0.15,linewidth=2, color='#d62728')
         #plt.tight_layout()
         plt.savefig('/home/ischirma/plots/CF_'+spec_segment+'_ocean'+name, format='png', bbox_inches = "tight")
         
         
         #cloud fraction nach Zeit nicht Höhe 
         bins_time=ds_cloud_top_height_sel.cloud_top_height[ocean_amali_water,:].count('cloud_layer')
         ds_amali_a_sel_fraction_time=bins_time/len(ds_cloud_top_height_sel.cloud_layer)
         fig, (ax1) = plt.subplots(1)
         ax1.plot(ds_mirac_a_sel.time[np.array(ocean_mirac_water)],ds_mirac_a_sel_fraction_time,'.', color='k',label='MiRAC')
         ax1.plot(ds_cloud_top_height_sel.time[np.array(ocean_amali_water)],ds_amali_a_sel_fraction_time,'.', color='b',label='AMALI')
         ax1.set_title(spec_segment+' '+str(seg["start"])+' - '+str(seg["end"])+'\nalt: '+'%3.2f'%alt+' km')
         ax1.legend(frameon=False, loc='upper right')
         ax1.set_ylim(0., 1.)
         ax1.set_xlabel('Time (hh:mm) [UTC]')
         ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
         ax1.set_ylabel('Signal fraction')
         for i, j in zip(range(-40,0,10), range(9,5,-1)):
             print(i,j)
             ds_mirac_a_sel_amount_20_time=((10*log10(ds_mirac_a_sel.Ze[ocean_mirac_water,:]))>=i).sum(axis=1)
             ds_mirac_a_sel_fraction_20_time=ds_mirac_a_sel_amount_20_time/len(ds_mirac_a_sel.height)
             ax1.plot(ds_mirac_a_sel.time[np.array(ocean_mirac_water)],ds_mirac_a_sel_fraction_20_time,'+', color='k',alpha=j*0.1,label='MiRAC '+str(i))
             ax1.set_ylim(0., 1.)
         ax1.legend(frameon=False, loc='upper right')  
       else:    
         bins=stack.groupby_bins(group=stack*1e-3,bins=np.arange(0.,(ds_cloud_top_height_sel.cloud_top_height.max()*1e-3)+0.0050,0.0050)).count()# stack hat schon weniger time
         ds_amali_a_sel_fraction=bins/len(ds_cloud_top_height_sel.time[np.array(ocean_amali_water)])

         bins_time=ds_cloud_top_height_sel.cloud_top_height[ocean_amali_water,:].count('cloud_layer')
         ds_amali_a_sel_fraction_time=bins_time/len(ds_cloud_top_height_sel.cloud_layer)
         
         fig, (ax1) = plt.subplots(1)
         ax1.plot(ds_mirac_a_sel_fraction[230:],ds_mirac_a_sel.height[230:]*1e-3, color='k',label='MiRAC')
         ax1.plot(ds_amali_a_sel_fraction,np.arange(0.,(ds_cloud_top_height_sel.cloud_top_height.max()*1e-3),0.0050)+0.0025, color='b',label='AMALI')
         ax1.set_title(spec_segment+' '+str(seg["start"])+' - '+str(seg["end"])+'\nalt: '+'%3.2f'%alt+' km')
         ax1.legend(frameon=False, loc='upper right')
         ax1.set_ylim(0, 3.5)
         ax1.set_xlim(0., 1.)
         ax1.set_ylabel('Height /  km')
         ax1.set_xlabel('Signal fraction')
         for i, j in zip(range(-40,0,10), range(9,5,-1)):
             print(i,j)
             ds_mirac_a_sel_amount_20=((10*log10(ds_mirac_a_sel.Ze[ocean_mirac_water,:]))>=i).sum(axis=0)
             ds_mirac_a_sel_fraction_20=ds_mirac_a_sel_amount_20/len(ds_mirac_a_sel.time[np.array(ocean_mirac_water)])
             ax1.plot(ds_mirac_a_sel_fraction_20[230:],ds_mirac_a_sel.height[230:]*1e-3,'--', color='k',alpha=j*0.1,label='MiRAC '+str(i))
             ax1.set_xlim(0., 1.)
         ax1.legend(frameon=False, loc='upper right')  
         #ax1.axhline(y=0.15,linewidth=2, color='#d62728')
         #plt.tight_layout()
         plt.savefig('/home/ischirma/plots/CF_'+spec_segment+'_ocean'+name, format='png', bbox_inches = "tight")
         
         
         #cloud fraction nach Zeit nicht Höhe 
         fig, (ax1) = plt.subplots(1)
         ax1.plot(ds_mirac_a_sel.time[np.array(ocean_mirac_water)],ds_mirac_a_sel_fraction_time,'.', color='k',label='MiRAC')
         ax1.plot(ds_cloud_top_height_sel.time[np.array(ocean_amali_water)],ds_amali_a_sel_fraction_time,'.', color='b',label='AMALI')
         ax1.set_title(spec_segment+' '+str(seg["start"])+' - '+str(seg["end"])+'\nalt: '+'%3.2f'%alt+' km')
         ax1.legend(frameon=False, loc='upper right')
         ax1.set_ylim(0., 1.)
         ax1.set_xlabel('Time (hh:mm) [UTC]')
         ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
         ax1.set_ylabel('Signal fraction')
         for i, j in zip(range(-40,0,10), range(9,5,-1)):
             print(i,j)
             ds_mirac_a_sel_amount_20_time=((10*log10(ds_mirac_a_sel.Ze[ocean_mirac_water,:]))>=i).sum(axis=1)
             ds_mirac_a_sel_fraction_20_time=ds_mirac_a_sel_amount_20_time/len(ds_mirac_a_sel.height)
             ax1.plot(ds_mirac_a_sel.time[np.array(ocean_mirac_water)],ds_mirac_a_sel_fraction_20_time,'+', color='k',alpha=j*0.1,label='MiRAC '+str(i))
             ax1.set_ylim(0., 1.)
         ax1.legend(frameon=False, loc='upper right')  
         #plt.tight_layout()
         plt.savefig('/home/ischirma/plots/CF_time_'+spec_segment+'_ocean'+name, format='png', bbox_inches = "tight")