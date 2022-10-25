#%%

import xarray as xr
import rioxarray as rioxr
import cfgrib
import geopandas as gpd
import matplotlib.pyplot as plt

# import cartopy.crs as ccrs

from shapely.geometry import mapping

from mypath import *
from const import *


def get_bound(shp_file=UK_SHP_ADM0):
    geodf = gpd.read_file(shp_file)
    geometry = geodf.geometry.apply(mapping)
    crs = geodf.crs

    return geometry, crs


def read_tif(tif_file):

    rio_ds = rioxr.open_rasterio(tif_file)
    return rio_ds.rename({"x": "lon", "y": "lat"})


def read_nc(nc_file):

    nc_ds = xr.open_dataset(nc_file)


def read_grib(grib_file):

    grib_ds = cfgrib.open_dataset(grib_file)
    return grib_ds.rename({"longitude": "lon", "latitude": "lat"})


def plot_s5p_no2_year(s5p_nc):
    org_ds = xr.open_dataset(s5p_nc)
    var_name = list(org_ds.keys())[0]
    org_ds = org_ds.rename(name_dict={var_name: "s5p_no2"})

    years = [2019, 2020, 2021, 2022]

    for i, y in enumerate(years):
        fig = plt.figure(1 + i, figsize=(10, 7))
        s5p_no2_year = org_ds.sel(time=org_ds.time.dt.year.isin([y]))
        s5p_mean = s5p_no2_year.mean("time")["s5p_no2"]
        s5p_mean = s5p_mean * 1e6
        s5p_mean.plot(
            cmap="YlOrRd",
            vmin=70,
            vmax=100,
            cbar_kwargs={"label": f"$10^{{{-6}}}$ $mol/m^2$"},
        )
        plt.title(f"{y}", fontsize=18)


# %%
