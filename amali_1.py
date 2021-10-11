# -*- coding: utf-8 -*-
"""
Spyder Editor

Dies ist eine temporäre Skriptdatei.
"""
import matplotlib.pyplot as plt
import ac3airborne
import matplotlib.dates as mdates
import numpy as np
campaign='ACLOUD_P5_RF06'
spec_segment='ACLOUD_P5_RF06_hl03'


cat = ac3airborne.get_intake_catalog()
list(cat.P5.CLOUD_TOP_HEIGHT)
ds_cloud_top_height = cat['P5']['CLOUD_TOP_HEIGHT'][campaign].to_dask()
ds_cloud_top_height

meta = ac3airborne.get_flight_segments()
segments = {s.get("segment_id"): {**s, "flight_id": flight["flight_id"]}
             for platform in meta.values()
             for flight in platform.values()
             for s in flight["segments"]
            }
seg = segments[spec_segment]
ds_cloud_top_height_sel = ds_cloud_top_height.sel(time=slice(seg["start"], seg["end"]))

plt.style.use("/home/ischirma/pyac3airborne_book/pyac3airborne_book/mplstyle/book")
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[1, 0.25]))

# 1st: plot flight altitude and cloud top height with seperate colors for each layer
ax1.plot(ds_cloud_top_height_sel.time, ds_cloud_top_height_sel.alt*1e-3, color='k', label='Flight altitude')

stack = ds_cloud_top_height_sel.cloud_top_height.stack({'tl': ['time', 'cloud_layer']})
im = ax1.scatter(x=stack.time, y=stack*1e-3, c=stack.cloud_layer, s=2, vmin=1, vmax=9, cmap='Set1')
fig.colorbar(im, ax=ax1, label='cloud layer (1 = highest cloud top)')
ax1.set_ylim(0, 4)
ax1.set_ylabel('Cloud top height [km]')
ax1.legend(frameon=False, loc='upper left')
# 3rd: plot cloud mask in lower part of the figure
ax2.scatter(ds_cloud_top_height_sel.time, ds_cloud_top_height_sel.cloud_mask, s=2, color='k')
ax2.set_yticks([int(x) for x in ds_cloud_top_height_sel.cloud_mask.attrs['flag_masks'].split(', ')])
ax2.set_yticklabels([x for x in ds_cloud_top_height_sel.cloud_mask.attrs['flag_meanings'].split(' ')])
ax2.set_xlabel('Time (hh:mm) [UTC]')
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.show()
#cloud top heigth in 5 m bins unterteilen

bins=stack.groupby_bins(group=stack*1e-3,bins=np.arange(0.,(ds_cloud_top_height_sel.cloud_top_height.max()*1e-3)+0.0050,0.0050)).count()
ds_amali_a_sel_fraction=bins/len(ds_cloud_top_height_sel.time)
plt.figure()
plt.plot(ds_amali_a_sel_fraction)

fig, (ax1) = plt.subplots(1)
ax1.plot(ds_amali_a_sel_fraction,np.arange(0.,(ds_cloud_top_height_sel.cloud_top_height.max()*1e-3),0.0050)+0.0025)
ax1.set_ylim(-0.25, 3.5)
ax1.set_ylabel('Height [km]')
ax1.set_xlabel('Backscatter fraction')
plt.show()