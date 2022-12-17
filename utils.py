#%%

import xarray as xr
import rioxarray as rioxr  # conflcict package seaborn conda remove seaborn
import cfgrib
import geopandas as gpd
import geopandas as gpd
import pandas as pd
import numpy as np

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


def read_tif(tif_file):

    rio_ds = rioxr.open_rasterio(tif_file)
    return rio_ds.rename({"x": "lon", "y": "lat"})


def read_grib(grib_file):

    grib_ds = cfgrib.open_dataset(grib_file)
    return grib_ds.rename({"longitude": "lon", "latitude": "lat"})


def prep_ds(org_ds, year):

    if year == 2020:
        ds = org_ds.dw_2020
    elif year == 2021:
        ds = org_ds.dw_2021
    else:
        ds = org_ds.dw_2022

    ds = ds[[S5P_PRED_COL, S5P_OBS_COL]]
    ds = ds.rio.write_crs("epsg:4326", inplace=True)
    return ds.rio.set_spatial_dims("lon", "lat", inplace=True)


def prep_s5p_ds():
    org_ds = xr.open_dataset(S5P_NO2_NC) * 1e6
    var_name = list(org_ds.keys())[0]
    org_ds = org_ds.rename(name_dict={var_name: S5P_OBS_COL})
    org_ds = org_ds.rio.write_crs("epsg:4326", inplace=True)
    org_ds = org_ds.rio.set_spatial_dims("lon", "lat", inplace=True)
    return org_ds


def prep_location_df(csv_path):

    df = pd.read_csv(csv_path)
    return gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["LONGITUDE"], df["LATITUDE"])
    )


def prep_fire_df():

    fire_war_gdf = prep_location_df(FIRE_WARTIME_CSV)
    fire_2021_gdf = prep_location_df(FIRE_2021_CSV)
    fire_2020_gdf = prep_location_df(FIRE_2020_CSV)

    fire_df = pd.concat([fire_war_gdf, fire_2021_gdf, fire_2020_gdf], ignore_index=True)

    fire_df["DATETIME"] = pd.to_datetime(fire_df["DATETIME"])

    return fire_df


def prep_conflict_df():

    battle_gdf = prep_location_df(BATTLE_CSV)
    expls_gdf = prep_location_df(EXPLOSION_CSV)

    conflict_df = pd.concat([battle_gdf, expls_gdf], ignore_index=True)
    conflict_df["DATETIME"] = pd.to_datetime(conflict_df["EVENT_DATETIME"])

    return conflict_df


def get_nday_mean(df, nday=3):
    df["time"] = df.index
    time = df["time"].groupby(np.arange(len(df)) // nday).mean()

    df = df.groupby(np.arange(len(df)) // nday).mean()
    df["time"] = time
    return df.set_index("time")


def clip_and_flat_major_city(ds, var, event="covid"):
    flat_array = np.array([])
    if event == "covid":

        bound_lv2, crs = get_bound_pop_lv2()
        col = "ADM2_EN"

        for adm2 in bound_lv2[col].values:
            geometry = bound_lv2.loc[bound_lv2[col] == adm2].geometry
            clip_arr = ds.rio.clip(geometry, crs)[var].values.reshape(-1)
            flat_array = np.concatenate((flat_array, clip_arr), axis=None)

    elif "war" in event:
        bound_lv1 = gpd.read_file(UK_SHP_ADM1)
        col = "ADM1_EN"

        if event == "war1":
            list_adm1 = [
                "Donetska",
                "Kyiv",
                "Volynska",
                "Lvivska",
                "Zakarpatska",
                "Ivano-Frankivska",
                "Chernivetska",
                "Kharkivska",
                "Zaporizka",
                "Khersonska",
                "Dnipropetrovska",
            ]
        elif event == "war2":
            list_adm1 = [
                "Donetska",
                "Kyiv",
                "Volynska",
                "Lvivska",
                "Zakarpatska",
                "Ivano-Frankivska",
                "Chernivetska",
            ]

        for adm1 in list_adm1:
            geometry = bound_lv1.loc[bound_lv1[col] == adm1].geometry
            clip_arr = ds.rio.clip(geometry, bound_lv1.crs)[var].values.reshape(-1)
            flat_array = np.concatenate((flat_array, clip_arr), axis=None)

    return flat_array

def get_boundary_cities():
    bound_lv2 = gpd.read_file(UK_SHP_ADM2)
    boundary = bound_lv2.loc[bound_lv2["ADM2_EN"].isin(LIST_BOUNDARY_CITY)]
    return boundary

def get_monthly_conflict():
    conflict_ds = prep_conflict_df()
    fire_ds = prep_fire_df()

    bound_lv2 = gpd.read_file(UK_SHP_ADM2)
    year_target = 2022
    col = "ADM2_EN"
    sd_ed = PERIOD_DICT[year_target]

    tks = list(sd_ed.keys())

    dict_conflict_monthly = {}
    dict_fire_monthly = {}

    for tk in tks:
        dict_conflict_monthly[tk] = []
        dict_fire_monthly[tk] = []
        t = sd_ed[tk]
        sd = np.datetime64(f"{year_target}-{t['sm']}-{t['sd']}{HOUR_STR}")
        ed = np.datetime64(f"{year_target}-{t['em']}-{t['ed']}{HOUR_STR}")

        mask_conflict_date = (conflict_ds["DATETIME"] > sd) & (
            conflict_ds["DATETIME"] <= ed
        )
        mask_fire_date = (fire_ds["DATETIME"] > sd) & (fire_ds["DATETIME"] <= ed)

        tk_cf_df = conflict_ds.loc[mask_conflict_date]
        tk_fire_df = fire_ds.loc[mask_fire_date]

        for adm2 in bound_lv2[col].values:
            geometry = bound_lv2.loc[bound_lv2[col] == adm2].geometry

            adm2_cflt_df = gpd.clip(tk_cf_df, geometry)
            adm2_fire_df = gpd.clip(tk_fire_df, geometry)

            dict_conflict_monthly[tk].append(len(adm2_cflt_df))
            dict_fire_monthly[tk].append(len(adm2_fire_df))

        bound_lv2[f"conflict_{tk}"] = dict_conflict_monthly[tk]
        bound_lv2[f"fire_{tk}"] = dict_fire_monthly[tk]

    return bound_lv2


# %%
