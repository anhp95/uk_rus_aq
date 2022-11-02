#%%

import xarray as xr
import rioxarray as rioxr # conflcict package seaborn conda remove seaborn
import cfgrib
import geopandas as gpd
import matplotlib.pyplot as plt
import datashader as dsh
import matplotlib.lines as mlines
import random
import pandas as pd

from datashader.mpl_ext import dsshow
from shapely.geometry import mapping
from shapely.geometry import Point
from sklearn.metrics import mean_squared_error, r2_score

from mypath import *
from const import *


def get_bound(shp_file=UK_SHP_ADM0):
    geodf = gpd.read_file(shp_file)
    geometry = geodf.geometry.apply(mapping)
    crs = geodf.crs

    return geometry, crs


def get_bound_pop_lv2():

    city_pop_df = pd.read_csv(CITY_POP)

    geo_df_lv2 = gpd.read_file(UK_SHP_ADM2)

    merge_df = pd.merge(city_pop_df, geo_df_lv2, on="ADM2_EN", how="inner")

    merge_df = gpd.GeoDataFrame(
        merge_df, crs=geo_df_lv2.crs, geometry=merge_df.geometry
    )

    return merge_df[["ADM2_EN", "Population", "geometry"]], geo_df_lv2.crs


# def read_tif(tif_file):

#     rio_ds = rioxr.open_rasterio(tif_file)
#     return rio_ds.rename({"x": "lon", "y": "lat"})


def read_grib(grib_file):

    grib_ds = cfgrib.open_dataset(grib_file)
    return grib_ds.rename({"longitude": "lon", "latitude": "lat"})


# %%
