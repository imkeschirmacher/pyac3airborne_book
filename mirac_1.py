#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 14:12:51 2021

@author: ischirma
"""
import matplotlib.pyplot as plt
import ac3airborne
import matplotlib.dates as mdates
from numpy import log10
import xarray as xr
import pandas as pd

plt.style.use("/home/ischirma/pyac3airborne_book/pyac3airborne_book/mplstyle/book")

cat = ac3airborne.get_intake_catalog()
list(cat.P5.MIRAC_A)

campaign='ACLOUD_P5_RF05'
spec_segment=campaign+'_hl01'

ds_mirac_a = cat['P5']['MIRAC_A'][campaign].to_dask()
ds_mirac_a
meta = ac3airborne.get_flight_segments()
segments = {s.get("segment_id"): {**s, "flight_id": flight["flight_id"]}
	      for platform in meta.values()
	      for flight in platform.values()
	      for s in flight["segments"]
		     }
seg = segments[spec_segment]
ds_mirac_a_sel = ds_mirac_a.sel(time=slice(seg["start"], seg["end"]))


fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
# 1st: plot flight altitude and radar reflectivity
ax1.plot(ds_mirac_a_sel.time, ds_mirac_a_sel.altitude*1e-3, label='Flight altitude', color='k')
im = ax1.pcolormesh(ds_mirac_a_sel.time, ds_mirac_a_sel.height*1e-3, 10*log10(
ds_mirac_a_sel.Ze).T, vmin=-40, vmax=30, cmap='jet', shading='flat')
fig.colorbar(im, ax=ax1, label='Radar reflectivity [dBz]')
ax1.set_ylim(-0.25, 3.5)
ax1.set_ylabel('Height [km]')
ax1.legend(frameon=False, loc='upper left')
# 2nd: plot 89 GHz TB
ax2.plot(ds_mirac_a_sel.time, ds_mirac_a_sel.TB_89, label='Tb(89 GHz)', color=
'k')
ax2.set_ylim(177, 195)
ax2.set_ylabel('$T_b$ [K]')
ax2.set_xlabel('Time (hh:mm) [UTC]')
ax2.legend(frameon=False, loc='upper left')
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.show()

ds_mirac_a_sel_amount=(ds_mirac_a_sel.Ze).count(axis=0)
ds_mirac_a_sel_fraction=ds_mirac_a_sel_amount/len(ds_mirac_a_sel.time)
plt.figure()
plt.plot(ds_mirac_a_sel_amount,ds_mirac_a_sel.height*1e-3)

fig, (ax1) = plt.subplots(1)
ax1.plot(ds_mirac_a_sel_fraction,ds_mirac_a_sel.height*1e-3)
ax1.set_ylim(-0.25, 3.5)
ax1.set_ylabel('Height [km]')
ax1.set_xlabel('Z signal fraction')
plt.show()


cat = ac3airborne.get_intake_catalog()
list(cat.P5.MIRAC_A)
tracks=cat.P5.MIRAC_A

meta = ac3airborne.get_flight_segments()
segments = {s.get("segment_id"): {**s, "flight_id": flight["flight_id"]}
	      for platform in meta.values()
	      for flight in platform.values()
	      for s in flight["segments"]
		     }
ds_mirac_a_sel_fraction=xr.DataArray(data=None)
#ds_mirac_a_sel_fraction=pd.DataFrame(data=None)
i=0.
for campaign in tracks:
	spec_segment=campaign+'_hl01'

	ds_mirac_a = cat['P5']['MIRAC_A'][campaign].to_dask()
	ds_mirac_a
	seg = segments[spec_segment]
	ds_mirac_a_sel = ds_mirac_a.sel(time=slice(seg["start"], seg["end"]))

	ds_mirac_a_sel_amount=(ds_mirac_a_sel.Ze).count(axis=0)
	ds_mirac_a_sel_fraction_0=ds_mirac_a_sel_amount/len(ds_mirac_a_sel.time)
	#ds_mirac_a_sel_fraction=ds_mirac_a_sel_fraction.assign(A=ds_mirac_a_sel_fraction_0)
	ds_mirac_a_sel_fraction_0.rename(i)
	#ds_mirac_a_sel_fraction=xr.DataArray(data=[ds_mirac_a_sel_fraction,ds_mirac_a_sel_fraction_0])
	i=i+1
	ds_mirac_a_sel_fraction=xr.merge([ds_mirac_a_sel_fraction,ds_mirac_a_sel_fraction_0])