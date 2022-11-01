#%%

import xarray as xr
import rioxarray as rioxr
import cfgrib
import geopandas as gpd
import matplotlib.pyplot as plt
import datashader as dsh
import matplotlib.lines as mlines
import random
import pandas as pd

# import cartopy.crs as ccrs
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


def get_bound_lv2():

    city_pop_df = pd.read_csv(CITY_POP)

    geo_df_lv2 = gpd.read_file(UK_SHP_ADM2)

    merge_df = pd.merge(city_pop_df, geo_df_lv2, on="ADM2_EN", how="inner")

    merge_df = gpd.GeoDataFrame(
        merge_df, crs=geo_df_lv2.crs, geometry=merge_df.geometry
    )

    return merge_df[["ADM2_EN", "Population", "geometry"]], geo_df_lv2.crs


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


def plot_train_test(lon, lat):
    lonlat = zip(lon, lat)

    list_lonlat = list(lonlat)

    train = random.sample(list_lonlat, 10000)
    test = random.sample(list_lonlat, 1000)
    crs = {"init": "epsg:4326"}
    geo_train = gpd.GeoDataFrame(crs=crs, geometry=[Point(xy) for xy in zip(train)])
    geo_test = gpd.GeoDataFrame(crs=crs, geometry=[Point(xy) for xy in zip(test)])

    fig, ax = plt.subplots(figsize=(7, 7))
    # ukr_shp.plot(ax=ax)
    geo_train.plot(ax=ax, color="green", markersize=5)
    geo_test.plot(ax=ax, color="red", markersize=5)


def plot_pred_true(ds):

    figure, axis = plt.subplots(1, 2, figsize=(16, 8))
    figure.tight_layout(pad=7.0)
    ds.test_2019.groupby("time").mean().mul(1e6)[["s5p_no2", "s5p_no2_pred"]].plot.line(
        ax=axis[0]
    )

    dsartist = dsshow(
        ds.test_2019[["s5p_no2", "s5p_no2_pred"]].mul(1e6),
        dsh.Point("s5p_no2", "s5p_no2_pred"),
        dsh.count(),
        norm="linear",
        aspect="auto",
        ax=axis[1],
    )

    plt.colorbar(dsartist)

    axis[0].set_title(
        "Time series trend of observation NO2 and Machine learning NO2 prediction"
    )
    axis[0].set_xlabel("Date")
    axis[0].set_ylabel(f"$10^{{{-6}}}$ $mol/m^2$")

    axis[1].set_title("NO2 Scatter Plot")
    axis[1].set_xlabel(f"NO2 S5P Obs $10^{{{-6}}}$ $mol/m^2$")
    axis[1].set_ylabel(f"NO2 ML predictions $10^{{{-6}}}$ $mol/m^2$")
    axis[1].annotate(
        "$R^2$ = {:.3f}".format(
            r2_score(ds.test_2019["s5p_no2"], ds.test_2019["s5p_no2_pred"])
        ),
        (10, 300),
    )
    line = mlines.Line2D([0, 1], [0, 1], color="red")
    transform = axis[1].transAxes
    line.set_transform(transform)
    axis[1].add_line(line)


# %%
