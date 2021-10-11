#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 15:39:15 2021

@author: ischirma
"""
import ac3airborne.tools.is_land as il
import ac3airborne
from simplification.cutil import simplify_coords_idx
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

def simplify_dataset(ds, tolerance):
    indices_to_take = simplify_coords_idx(np.stack([ds.lat.values, ds.lon.values], axis=1), tolerance)
    return ds.isel(time=indices_to_take)

cat = ac3airborne.get_intake_catalog()
ds_gps = cat['P5']['GPS_INS']['ACLOUD_P5_RF14'].to_dask()
ds_gps = ds_gps.isel(time=slice(1,-1))
dsreduced = simplify_dataset(ds_gps, 1e-3)

proj   = ccrs.NorthPolarStereo()
extent = (-5.0, 24.0, 78.0, 83.0)

"""
fig = plt.figure(figsize=(15, 15))
ax = plt.axes(projection=proj)
ax.set_extent(extent)

ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.LAND)
ax.gridlines()
ax.coastlines()

nya_lat = 78.924444
nya_lon = 11.928611

ax.plot(nya_lon, nya_lat, 'ro', transform=ccrs.PlateCarree())
ax.text(nya_lon, nya_lat+0.05, 'Ny-Ålesund', transform=ccrs.PlateCarree())

#for x, y in zip(ds_gps.lon, ds_gps.lat):
for x, y in zip(dsreduced.lon, dsreduced.lat):
    if il.is_land(x, y):
        ax.scatter(x, y, transform=ccrs.PlateCarree(), c='red')
    else:
        ax.scatter(x, y, transform=ccrs.PlateCarree(), c='green')
"""        
for x, y in zip(dsreduced.lon, dsreduced.lat):
    if il.is_land(x, y):
        print(str(x.time.values),' land')
    else:
        print(str(x.time.values),' ocean')