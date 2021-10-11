#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 14:29:00 2021

@author: ischirma
"""
import ac3airborne
import numpy as np
from numpy import log10
import ac3airborne.tools.is_land as il
from simplification.cutil import simplify_coords_idx
import pandas as pd
import matplotlib
matplotlib.rcParams.update({'font.size': 22})
import matplotlib.pyplot as plt

def simplify_dataset(ds, tolerance):
    indices_to_take = simplify_coords_idx(np.stack([ds.lat.values, ds.lon.values], axis=1), tolerance)
    return ds.isel(time=indices_to_take)
"""
#creating a dictionary
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
CFdict=dict()
percentdict=dict()
amountdict=dict()
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
     ocean_mirac_old=test_2.ocean.fillna(method='ffill') #ocean or land for nall following nan
     test_2=pd.DataFrame()
     test_2.index=ds_cloud_top_height_sel.time
     test_2['ocean']=test
     ocean_amali=test_2.ocean.fillna(method='ffill') #ocean or land for nall following nan
     ocean_mirac=ocean_mirac_old.fillna(ocean_amali)
     ocean_amali=ocean_amali.fillna(ocean_mirac_old)
     ice=pd.Series(ds_sea_ice_sel.sic>90)
     ice.index=ds_sea_ice_sel.time
     water=pd.Series(ds_sea_ice_sel.sic<15)
     water.index=ds_sea_ice_sel.time
     
     name=''
     amount_sensitivity=pd.DataFrame()
     amount_insgesamt=len(ds_mirac_a_sel.time)*len(ds_mirac_a_sel.height)
     amount_insgesamt_amali=len(ds_cloud_top_height_sel.time)
     #alle Werte auch über land
     ds_mirac_a_sel_amount=(ds_mirac_a_sel.Ze).count(axis=0)
     ds_mirac_a_sel_fraction_overall=ds_mirac_a_sel_amount/len(ds_mirac_a_sel.time)
     stack_overall = ds_cloud_top_height_sel.cloud_top_height.stack({'tl': ['time', 'cloud_layer']})
     #CFdict={str(spec_segment): list(ds_mirac_a_sel_fraction_overall.values)}
     #CFdict={str(spec_segment): {}}
     #CFdict[str(spec_segment)].append((ds_mirac_a_sel_fraction_overall.values))
     CFdict[str(spec_segment)]={}
     CFdict[str(spec_segment)]['MiRAC_overall']=ds_mirac_a_sel_fraction_overall.values
     if str(ds_cloud_top_height_sel.cloud_top_height.max())=="<xarray.DataArray 'cloud_top_height' ()>\narray(nan)":
         for i, j in zip(range(-40,0,10), range(9,5,-1)):
             ds_mirac_a_sel_amount_20_overall=((10*log10(ds_mirac_a_sel.Ze))>=i).sum(axis=0)
             ds_mirac_a_sel_fraction_20_overall=ds_mirac_a_sel_amount_20_overall/len(ds_mirac_a_sel.time)
             CFdict[str(spec_segment)]['MiRAC_overall_'+str(i)]=ds_mirac_a_sel_fraction_20_overall.values
             amount_sensitivity.loc[0,'MiRAC_overall_'+str(i)]=int(((10*log10(ds_mirac_a_sel.Ze))>=i).sum())
     else:    
         bins_overall=stack_overall.groupby_bins(group=stack_overall*1e-3,bins=np.arange(0.,(ds_cloud_top_height_sel.cloud_top_height.max()*1e-3)+0.0050,0.0050)).count()
         ds_amali_a_sel_fraction_overall=bins_overall/len(ds_cloud_top_height_sel.time)
         CFdict[str(spec_segment)]['AMALI_overall']=ds_amali_a_sel_fraction_overall.values
         for i, j in zip(range(-40,0,10), range(9,5,-1)):         
             ds_mirac_a_sel_amount_20_overall=((10*log10(ds_mirac_a_sel.Ze))>=i).sum(axis=0)
             ds_mirac_a_sel_fraction_20_overall=ds_mirac_a_sel_amount_20_overall/len(ds_mirac_a_sel.time)
             CFdict[str(spec_segment)]['MiRAC_overall_'+str(i)]=ds_mirac_a_sel_fraction_20_overall.values
             amount_sensitivity.loc[0,'MiRAC_overall_'+str(i)]=int(((10*log10(ds_mirac_a_sel.Ze))>=i).sum())
             
    #ab jetzt nur Werte über ocean nicht mehr über land       
     ocean_mirac_water=(ocean_mirac).fillna(0).astype('bool') #0 wenn eins falsch ist , 1 wenn beide richtig sind, nan wenn eins nicht vorhanden ist bei einem Zeitpunkt-> das mache ich dann auch zu 0: ich will alle 1
     ocean_amali_water=(ocean_amali).fillna(0).astype('bool')
     ocean_mirac_water=ocean_mirac_water[ds_mirac_a_sel.time.values]
     ocean_amali_water=ocean_amali_water[ds_cloud_top_height_sel.time.values]
     amount_ocean=ocean_mirac.sum()*len(ds_mirac_a_sel.height)
     amount_ocean_amali=ocean_amali.sum()

     if sum(ocean_mirac_water>0):
      ds_mirac_a_sel_amount_overwater=(ds_mirac_a_sel.Ze[ocean_mirac_water,:]).count(axis=0)
      ds_mirac_a_sel_fraction_overwater=ds_mirac_a_sel_amount_overwater/len(ds_mirac_a_sel.time[np.array(ocean_mirac_water)])
      stack_overwater = ds_cloud_top_height_sel.cloud_top_height[ocean_amali_water,:].stack({'tl': ['time', 'cloud_layer']})
      #CFdict[str(spec_segment)].append((ds_mirac_a_sel_fraction_overwater.values))
      CFdict[str(spec_segment)]['MiRAC_overwater']=ds_mirac_a_sel_fraction_overwater.values
      if str(stack_overwater.max())=="<xarray.DataArray 'cloud_top_height' ()>\narray(nan)":
         for i, j in zip(range(-40,0,10), range(9,5,-1)):
             print(i,j)
             ds_mirac_a_sel_amount_20_overwater=((10*log10(ds_mirac_a_sel.Ze[ocean_mirac_water,:]))>=i).sum(axis=0)
             ds_mirac_a_sel_fraction_20_overwater=ds_mirac_a_sel_amount_20_overwater/len(ds_mirac_a_sel.time[np.array(ocean_mirac_water)])
             #CFdict[str(spec_segment)].append((ds_mirac_a_sel_fraction_20_overwater.values))
             CFdict[str(spec_segment)]['MiRAC_overwater_'+str(i)]=ds_mirac_a_sel_fraction_20_overwater.values
             amount_sensitivity.loc[0,'MiRAC_overwater_'+str(i)]=int(((10*log10(ds_mirac_a_sel.Ze[ocean_mirac_water,:]))>=i).sum())
      else:    
         bins_overwater=stack_overwater.groupby_bins(group=stack_overwater*1e-3,bins=np.arange(0.,(ds_cloud_top_height_sel.cloud_top_height.max()*1e-3)+0.0050,0.0050)).count()# stack hat schon weniger time
         ds_amali_a_sel_fraction_overwater=bins_overwater/len(ds_cloud_top_height_sel.time[np.array(ocean_amali_water)])
         #CFdict[str(spec_segment)].append((ds_amali_a_sel_fraction_overwater.values))
         CFdict[str(spec_segment)]['AMALI_overwater']=ds_amali_a_sel_fraction_overwater.values
         for i, j in zip(range(-40,0,10), range(9,5,-1)):
             ds_mirac_a_sel_amount_20_overwater=((10*log10(ds_mirac_a_sel.Ze[ocean_mirac_water,:]))>=i).sum(axis=0)
             ds_mirac_a_sel_fraction_20_overwater=ds_mirac_a_sel_amount_20_overwater/len(ds_mirac_a_sel.time[np.array(ocean_mirac_water)])
             #CFdict[str(spec_segment)].append((ds_mirac_a_sel_fraction_20_overwater.values))
             CFdict[str(spec_segment)]['MiRAC_overwater_'+str(i)]=ds_mirac_a_sel_fraction_20_overwater.values
             amount_sensitivity.loc[0,'MiRAC_overwater_'+str(i)]=int(((10*log10(ds_mirac_a_sel.Ze[ocean_mirac_water,:]))>=i).sum())

    #nur über wasser wo gleichzeitig ice
    #wenn zusätlich nur über Ice oder nur über water: dann wird in ocean mirac und amali zusätzlich diese Info eingebunden
     ocean_mirac_ice=(ocean_mirac*ice).fillna(0).astype('bool') #0 wenn eins falsch ist , 1 wenn beide richtig sind, nan wenn eins nicht vorhanden ist bei einem Zeitpunkt-> das mache ich dann auch zu 0: ich will alle 1
     ocean_amali_ice=(ocean_amali*ice).fillna(0).astype('bool')
     ocean_mirac_ice=ocean_mirac_ice[ds_mirac_a_sel.time.values]
     ocean_amali_ice=ocean_amali_ice[ds_cloud_top_height_sel.time.values]
     amount_oceanice=ocean_mirac_ice.sum()*len(ds_mirac_a_sel.height)
     amount_oceanice_amali=ocean_amali_ice.sum()
     
     if sum(ocean_mirac_ice>0):
      ds_mirac_a_sel_amount_overice=(ds_mirac_a_sel.Ze[ocean_mirac_ice,:]).count(axis=0)
      ds_mirac_a_sel_fraction_overice=ds_mirac_a_sel_amount_overice/len(ds_mirac_a_sel.time[np.array(ocean_mirac_ice)])
      stack_overice = ds_cloud_top_height_sel.cloud_top_height[ocean_amali_ice,:].stack({'tl': ['time', 'cloud_layer']})
      CFdict[str(spec_segment)]['MiRAC_overwater_ice']=ds_mirac_a_sel_fraction_overice.values
      #CFdict[str(spec_segment)].append((ds_mirac_a_sel_fraction_overice.values))
      if str(stack_overice.max())=="<xarray.DataArray 'cloud_top_height' ()>\narray(nan)":
         for i, j in zip(range(-40,0,10), range(9,5,-1)):
             print(i,j)
             ds_mirac_a_sel_amount_20_overice=((10*log10(ds_mirac_a_sel.Ze[ocean_mirac_ice,:]))>=i).sum(axis=0)
             ds_mirac_a_sel_fraction_20_overice=ds_mirac_a_sel_amount_20_overice/len(ds_mirac_a_sel.time[np.array(ocean_mirac_ice)])
             CFdict[str(spec_segment)]['MiRAC_overwater_ice_'+str(i)]=ds_mirac_a_sel_fraction_20_overice.values
             amount_sensitivity.loc[0,'MiRAC_overwater_ice_'+str(i)]=int(((10*log10(ds_mirac_a_sel.Ze[ocean_mirac_ice,:]))>=i).sum())
      else:    
         bins_overice=stack_overice.groupby_bins(group=stack_overice*1e-3,bins=np.arange(0.,(ds_cloud_top_height_sel.cloud_top_height.max()*1e-3)+0.0050,0.0050)).count()# stack hat schon weniger time
         ds_amali_a_sel_fraction_overice=bins_overice/len(ds_cloud_top_height_sel.time[np.array(ocean_amali_ice)])
         CFdict[str(spec_segment)]['AMALI_overwater_ice']=ds_amali_a_sel_fraction_overice.values
         #CFdict[str(spec_segment)].append((ds_amali_a_sel_fraction_overice.values))
         for i, j in zip(range(-40,0,10), range(9,5,-1)):
             ds_mirac_a_sel_amount_20_overice=((10*log10(ds_mirac_a_sel.Ze[ocean_mirac_ice,:]))>=i).sum(axis=0)
             ds_mirac_a_sel_fraction_20_overice=ds_mirac_a_sel_amount_20_overice/len(ds_mirac_a_sel.time[np.array(ocean_mirac_ice)])
             CFdict[str(spec_segment)]['MiRAC_overwater_ice_'+str(i)]=ds_mirac_a_sel_fraction_20_overice.values
             amount_sensitivity.loc[0,'MiRAC_overwater_ice_'+str(i)]=int(((10*log10(ds_mirac_a_sel.Ze[ocean_mirac_ice,:]))>=i).sum())
             
    #nur über water and free water no ice
     ocean_mirac_water=(ocean_mirac*water).fillna(0).astype('bool') #0 wenn eins falsch ist , 1 wenn beide richtig sind, nan wenn eins nicht vorhanden ist bei einem Zeitpunkt-> das mache ich dann auch zu 0: ich will alle 1
     ocean_amali_water=(ocean_amali*water).fillna(0).astype('bool')
     ocean_mirac_water=ocean_mirac_water[ds_mirac_a_sel.time.values]
     ocean_amali_water=ocean_amali_water[ds_cloud_top_height_sel.time.values]
     amount_oceanwater=ocean_mirac_water.sum()*len(ds_mirac_a_sel.height)
     amount_oceanwater_amali=ocean_amali_water.sum()
     
     if sum(ocean_mirac_water>0):
      ds_mirac_a_sel_amount_overwater=(ds_mirac_a_sel.Ze[ocean_mirac_water,:]).count(axis=0)
      ds_mirac_a_sel_fraction_overwater=ds_mirac_a_sel_amount_overwater/len(ds_mirac_a_sel.time[np.array(ocean_mirac_water)])
      stack_overwater = ds_cloud_top_height_sel.cloud_top_height[ocean_amali_water,:].stack({'tl': ['time', 'cloud_layer']})
      #CFdict[str(spec_segment)].append((ds_mirac_a_sel_fraction_overwater.values))
      CFdict[str(spec_segment)]['MiRAC_overwater_water']=ds_mirac_a_sel_fraction_overwater.values
      if str(stack_overwater.max())=="<xarray.DataArray 'cloud_top_height' ()>\narray(nan)":
         for i, j in zip(range(-40,0,10), range(9,5,-1)):
             print(i,j)
             ds_mirac_a_sel_amount_20_overwater=((10*log10(ds_mirac_a_sel.Ze[ocean_mirac_water,:]))>=i).sum(axis=0)
             ds_mirac_a_sel_fraction_20_overwater=ds_mirac_a_sel_amount_20_overwater/len(ds_mirac_a_sel.time[np.array(ocean_mirac_water)])
             #CFdict[str(spec_segment)].append((ds_mirac_a_sel_fraction_20_overwater.values))
             CFdict[str(spec_segment)]['MiRAC_overwater_water_'+str(i)]=ds_mirac_a_sel_fraction_20_overwater.values
             amount_sensitivity.loc[0,'MiRAC_overwater_water_'+str(i)]=int(((10*log10(ds_mirac_a_sel.Ze[ocean_mirac_water,:]))>=i).sum())
      else:    
         bins_overwater=stack_overwater.groupby_bins(group=stack_overwater*1e-3,bins=np.arange(0.,(ds_cloud_top_height_sel.cloud_top_height.max()*1e-3)+0.0050,0.0050)).count()# stack hat schon weniger time
         ds_amali_a_sel_fraction_overwater=bins_overwater/len(ds_cloud_top_height_sel.time[np.array(ocean_amali_water)])
         #CFdict[str(spec_segment)].append((ds_amali_a_sel_fraction_overwater.values))
         CFdict[str(spec_segment)]['AMALI_overwater_water']=ds_amali_a_sel_fraction_overwater.values
         for i, j in zip(range(-40,0,10), range(9,5,-1)):
             ds_mirac_a_sel_amount_20_overwater=((10*log10(ds_mirac_a_sel.Ze[ocean_mirac_water,:]))>=i).sum(axis=0)
             ds_mirac_a_sel_fraction_20_overwater=ds_mirac_a_sel_amount_20_overwater/len(ds_mirac_a_sel.time[np.array(ocean_mirac_water)])
             #CFdict[str(spec_segment)].append((ds_mirac_a_sel_fraction_20_overwater.values))
             CFdict[str(spec_segment)]['MiRAC_overwater_water_'+str(i)]=ds_mirac_a_sel_fraction_20_overwater.values
             amount_sensitivity.loc[0,'MiRAC_overwater_water_'+str(i)]=int(((10*log10(ds_mirac_a_sel.Ze[ocean_mirac_water,:]))>=i).sum())
     percentdict[str(spec_segment)]={}  
     amountdict[str(spec_segment)]={}          
     for syst in ['MiRAC','AMALI']:  
      if syst=='MiRAC':  
          percent_ocean=amount_ocean/amount_insgesamt
          percent_oceanwater=amount_oceanwater/amount_insgesamt
          percent_oceanice=amount_oceanice/amount_insgesamt
          amountdict[str(spec_segment)][syst+'_overwater']=amount_ocean
          amountdict[str(spec_segment)][syst+'_overwater_water']=amount_oceanwater
          amountdict[str(spec_segment)][syst+'_overwater_ice']=amount_oceanice
          amountdict[str(spec_segment)][syst+'_intotal']=amount_insgesamt
          amountdict[str(spec_segment)][syst+'_amount_height']=len(ds_mirac_a_sel.height)
          for i in amount_sensitivity.columns:
           percentdict[str(spec_segment)][i]=amount_sensitivity.loc[0,i]/amount_insgesamt
           amountdict[str(spec_segment)][i]=amount_sensitivity.loc[0,i]
           
      elif syst=='AMALI':
          percent_ocean=amount_ocean_amali/amount_insgesamt_amali
          percent_oceanwater=amount_oceanwater_amali/amount_insgesamt_amali
          percent_oceanice=amount_oceanice_amali/amount_insgesamt_amali
          amountdict[str(spec_segment)][syst+'_overwater']=amount_ocean_amali
          amountdict[str(spec_segment)][syst+'_overwater_water']=amount_oceanwater_amali
          amountdict[str(spec_segment)][syst+'_overwater_ice']=amount_oceanice_amali
          amountdict[str(spec_segment)][syst+'_intotal']=amount_insgesamt_amali

      percentdict[str(spec_segment)][syst+'_overwater']=percent_ocean
      percentdict[str(spec_segment)][syst+'_overwater_water']=percent_oceanwater
      percentdict[str(spec_segment)][syst+'_overwater_ice']=percent_oceanice
      
np.save('/home/ischirma/data/CF_'+string+'.npy', CFdict)  
np.save('/home/ischirma/data/datafraction_'+string+'.npy', percentdict)  
np.save('/home/ischirma/data/absamount_'+string+'.npy', amountdict) 
"""
#reading the dictionary
for pre in ['','_-10','_-20','_-30','_-40']:
 string = ['ACLOUD', 'AFLUX','MOSAiC-ACA']
 bedingungen=['MiRAC_overwater'+pre, 'MiRAC_overwater_water'+pre,'MiRAC_overwater_ice'+pre]
 mittel=pd.DataFrame()
 sum2=pd.DataFrame()
 suminsg=pd.DataFrame({string[0]: [0],string[1]: [0], string[2]: [0]})
 for s in string:
  CFdict = np.load('/home/ischirma/data/CF_'+s+'.npy',allow_pickle='TRUE').item()  
  amountdict = np.load('/home/ischirma/data/absamount_'+s+'.npy',allow_pickle='TRUE').item()
  keys= list(CFdict.keys())
  for k in keys:
   suminsg.loc[0,s]=suminsg.loc[0,s]+amountdict[k]['MiRAC_intotal'] 
  for bedingung in bedingungen:
   test=pd.DataFrame()
   test2=pd.DataFrame()
   test_insg=pd.DataFrame()
   summe_intotal_sel=0
   i=0   
   for key in keys:  
     if bedingung in CFdict[key].keys():
      test[i]=CFdict[key][bedingung]
         #test hat jetzt die CF in jeder höhe für ein segment (col), wenn ich jetzt aber einfach über höhen mittle werden alle gleich gewichtet auch wenn segmente unterschiedlich lang sind!deswegen jede column mit anz insgesamt die bedingung erfüllen des segments mulitplizieren, dann bekomme ich wieder absolute anzahl
      test[i]=test[i]*(amountdict[key][bedingung]/amountdict[key]['MiRAC_amount_height'])
      summe_intotal_sel=summe_intotal_sel+(amountdict[key][bedingung]/amountdict[key]['MiRAC_amount_height'])
      test2.loc[i,0]=amountdict[key][bedingung]
      i+=1 
   print(bedingung) 
   mittel[s+'_'+bedingung]=test.sum(axis=1)
   #jetzt habe ich totale anzahl in jeweiliger Höhe, jetzt muss für fraction noch durch totale anzahl aller der selektierten segemnts aus der höhe die bedingung erfüllen geteilt werden
   mittel[s+'_'+bedingung]=mittel[s+'_'+bedingung]/summe_intotal_sel
   sum2[s+'_'+bedingung]=test2.sum()
  if pre=='' and s=='ACLOUD': 
   merke=mittel[[s+'_'+'MiRAC_overwater',s+'_'+'MiRAC_overwater_water',s+'_'+'MiRAC_overwater_ice']]
   merke_label=((sum2[[s+'_'+'MiRAC_overwater',s+'_'+'MiRAC_overwater_water',s+'_'+'MiRAC_overwater_ice']]/suminsg.loc[0,s])*100)
#mittel.plot(style=['-','--','dotted','-','--','dotted','-','--','dotted'],color=['r','r','r','b','b','b','purple','purple','purple'] )
 fig, (ax1) = plt.subplots(1)
 for p, col,  camp,sty, alphas  in zip(mittel.columns,['r','r','r','b','b','b','purple','purple','purple'],['ACLOUD','ACLOUD','ACLOUD', 'AFLUX','AFLUX','AFLUX','MOSAiC-ACA','MOSAiC-ACA','MOSAiC-ACA'],['-','--','dotted','-','--','dotted','-','--','dotted'],np.append(np.append(np.linspace(1, 0.5, 3),np.linspace(1, 0.5, 3)),np.linspace(1, 0.5, 3))):
  #ax1.plot(mittel.loc[:,p],np.arange(-1,6,0.005),linestyle=sty, color=col,label=p, alpha=alphas)
  ax1.plot(mittel.loc[230:,p],np.arange(0.15,6,0.005),linestyle=sty, color=col,label=p+'; '+'%3.2f'%((sum2.loc[0,p]/suminsg.loc[0,camp])*100)+'%', alpha=alphas)
 ax1.set_ylim(0., 3.5)
 ax1.set_xlim(0., 1.)
 ax1.set_ylabel('Height / km')
 ax1.set_xlabel('Signal fraction')
 ax1.legend(frameon=False, loc='upper right', fontsize=8)   
 #ax1.axhline(y=0.15,linewidth=2, color='#d62728')
 plt.savefig('/home/ischirma/plots/CF_Mirac_season'+pre, format='png', bbox_inches = "tight")
 plt.tight_layout()

print('HIER',merke_label)
#nur ACLOUD mit Unterteilung CP, WP und NP
CP=['RF04','RF05','RF06','RF07','RF08']
WP=['RF10','RF11','RF13','RF14','RF15']
NP=['RF16','RF17','RF18','RF19','RF20','RF21','RF22','RF23','RF25']
perioden=[CP,WP,NP]
strperioden=['CP','WP','NP']
#unterteilung in WP,CP,NP
s= 'ACLOUD'
bedingungen=['MiRAC_overwater', 'MiRAC_overwater_water','MiRAC_overwater_ice']
mittel=pd.DataFrame()
CFdict = np.load('/home/ischirma/data/CF_'+s+'.npy',allow_pickle='TRUE').item()  
amountdict = np.load('/home/ischirma/data/absamount_'+s+'.npy',allow_pickle='TRUE').item()
test=pd.DataFrame()
test2=pd.DataFrame()
test_insg=pd.DataFrame()
summe_intotal_sel=0
sum2=pd.DataFrame()
suminsg=pd.DataFrame({s: [0]})
i=0
j=0
keys= list(CFdict.keys())
for key in keys:
 suminsg.loc[0,s]=suminsg.loc[0,s]+amountdict[key]['MiRAC_intotal']
for per in perioden: 
 for bedingung in bedingungen:
  for period in per:
     print(period)
     res = [ia for ia in keys if period in ia]
     print(str(res))
     for r in res:
      print(r)   
      if bedingung in CFdict[str(r)]:
       test[i]=CFdict[str(r)][bedingung]
       test[i]=test[i]*(amountdict[str(r)][bedingung]/amountdict[str(r)]['MiRAC_amount_height'])
       summe_intotal_sel=summe_intotal_sel+(amountdict[str(r)][bedingung]/amountdict[str(r)]['MiRAC_amount_height'])
       test2.loc[i,0]=amountdict[str(r)][bedingung]
       i+=1 
  print(bedingung)    
  mittel[s+'_'+bedingung+'_'+strperioden[j]]=test.sum(axis=1)
  #jetzt habe ich totale anzahl in jeweiliger Höhe, jetzt muss für fraction noch durch totale anzahl aller der selektierten segemnts aus der höhe die bedingung erfüllen geteilt werden
  mittel[s+'_'+bedingung+'_'+strperioden[j]]=mittel[s+'_'+bedingung+'_'+strperioden[j]]/summe_intotal_sel
  sum2[s+'_'+bedingung+'_'+strperioden[j]]=test2.sum()
  test=pd.DataFrame()
  test2=pd.DataFrame()
  summe_intotal_sel=0
 j+=1 
#------------
#nur wenn der Abschnitt davor durchgelaufnen ist und ein Mittel mit allem existiert
fig, (ax1) = plt.subplots(1)
for p, col, sty, alphas  in zip(merke.columns[:3],['k','k','k'],['-','--','dotted'],np.linspace(1, 0.5, 3)):
 ax1.plot(merke.loc[230:,p],np.arange(0.15,6,0.005),linestyle=sty, color=col,label=p+'; '+'%3.2f'%merke_label.loc[0,p]+'%', alpha=alphas)
#----------- 
for p, col, sty, alphas  in zip(mittel.columns,['b','b','b','r','r','r','purple','purple','purple'],['-','--','dotted','-','--','dotted','-','--','dotted'],np.append(np.append(np.linspace(1, 0.5, 3),np.linspace(1, 0.5, 3)),np.linspace(1, 0.5, 3))):
 ax1.plot(mittel.loc[230:,p],np.arange(0.15,6,0.005),linestyle=sty, color=col,label=p+'; '+'%3.2f'%((sum2.loc[0,p]/suminsg.loc[0,])*100)+'%', alpha=alphas)
ax1.set_ylim(0., 3.5)
ax1.set_xlim(0., 1.)
ax1.set_ylabel('Height / km')
ax1.set_xlabel('Signal fraction')
ax1.legend(frameon=False, loc='upper right', fontsize=8)   
 #ax1.axhline(y=0.15,linewidth=2, color='#d62728')
plt.savefig('/home/ischirma/plots/CF_Mirac_ACLOUD_coldwarmperiods', format='png', bbox_inches = "tight")
plt.tight_layout()

      
#AMALI
string = ['ACLOUD', 'AFLUX','MOSAiC-ACA']
bedingungen=['AMALI_overwater', 'AMALI_overwater_water','AMALI_overwater_ice']
mittel=pd.DataFrame()
sum2=pd.DataFrame()
suminsg=pd.DataFrame({string[0]: [0],string[1]: [0], string[2]: [0]})
for s in string:
 CFdict = np.load('/home/ischirma/data/CF_'+s+'.npy',allow_pickle='TRUE').item()  
 amountdict = np.load('/home/ischirma/data/absamount_'+s+'.npy',allow_pickle='TRUE').item()
 #test=pd.DataFrame(np.nan)#ganz groß machen und dann nur noch überschreiebn und dann mitteln!
 keys= list(CFdict.keys())
 for k in keys:
  suminsg.loc[0,s]=suminsg.loc[0,s]+amountdict[k]['AMALI_intotal'] 
 for bedingung in bedingungen:
  test= pd.DataFrame(np.nan, index=range(0,2000), columns=range(0,100))
  test2=pd.DataFrame()
  test_insg=pd.DataFrame()
  summe_intotal_sel=0
  i=0
  for key in keys:
     if bedingung in CFdict[key].keys():
      test.loc[:len(CFdict[key][bedingung])-1,i]=CFdict[key][bedingung]
      test.loc[:len(CFdict[key][bedingung])-1,i]=test.loc[:len(CFdict[key][bedingung])-1,i]*(amountdict[key][bedingung])
      summe_intotal_sel=summe_intotal_sel+(amountdict[key][bedingung])
      test2.loc[i,0]=amountdict[key][bedingung]
      i+=1 
  print(bedingung)    
  mittel[s+'_'+bedingung]=test.sum(axis=1)
  mittel[s+'_'+bedingung]=mittel[s+'_'+bedingung]/summe_intotal_sel
  sum2[s+'_'+bedingung]=test2.sum()
mittel=mittel.fillna(0)  
#gleitender Mittelwert über 100 Meteren 
rolling_mean_AMALI=mittel[:].rolling(40).mean()
#mittel.plot(color=['r','r','r','b','b','b','purple','purple','purple'] )#alpha=alphas
fig, (ax1) = plt.subplots(1)
for p, col,camp, sty, alphas  in zip(mittel.columns,['r','r','r','b','b','b','purple','purple','purple'],['ACLOUD','ACLOUD','ACLOUD', 'AFLUX','AFLUX','AFLUX','MOSAiC-ACA','MOSAiC-ACA','MOSAiC-ACA'],['-','--','dotted','-','--','dotted','-','--','dotted'],np.append(np.append(np.linspace(1, 0.25, 3),np.linspace(1, 0.25, 3)),np.linspace(1, 0.25, 3))):
  #ax1.plot(mittel.loc[:len(np.arange(0.,3.5,0.0050))-1,p],np.arange(0.,3.5,0.0050)+0.0025,linestyle=sty, color=col,label=p, alpha=alphas)
  ax1.plot(rolling_mean_AMALI.loc[:len(np.arange(0.,3.5,0.0050))-1,p],np.arange(0.,3.5,0.0050)+0.0025,linestyle=sty, color=col,label=p+'; '+'%3.2f'%((sum2.loc[0,p]/suminsg.loc[0,camp])*100)+'%',  alpha=alphas)
#ax1.set_ylim(-0.25, 3.5)
ax1.set_xlim(0., .05)
ax1.set_ylabel('Height / km')
ax1.set_xlabel('Signal fraction')
ax1.legend(frameon=False, loc='upper right', fontsize=8)  
plt.savefig('/home/ischirma/plots/CF_AMALI_season_runningmean_200m', format='png', bbox_inches = "tight")
plt.tight_layout()