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


def get_bound_pop():

    adm_col = "ADM3_EN"
    city_pop_df = pd.read_csv(CITY_POP)

    geo_df_lv2 = gpd.read_file(UK_SHP_ADM3)

    merge_df = pd.merge(city_pop_df, geo_df_lv2, on=adm_col, how="inner")

    merge_df = gpd.GeoDataFrame(
        merge_df, crs=geo_df_lv2.crs, geometry=merge_df.geometry
    )

    return merge_df[[adm_col, "Population", "geometry"]], geo_df_lv2.crs


def read_tif(tif_file):

    rio_ds = rioxr.open_rasterio(tif_file)
    return rio_ds.rename({"x": "lon", "y": "lat"})


def read_grib(grib_file):

    grib_ds = cfgrib.open_dataset(grib_file)
    return grib_ds.rename({"longitude": "lon", "latitude": "lat"})


def prep_ds(org_ds, year):

    if year == 2019:
        ds = org_ds.dw_2019
    elif year == 2020:
        ds = org_ds.dw_2020
    elif year == 2021:
        ds = org_ds.dw_2021
    elif year == 2022:
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


def clip_and_flat_event_city(ds, var, tk, conflict_df, event):

    flat_array = np.array([])
    # # list_city = ["Donetska", "Kharkivska", "Zaporizka"]
    adm_col = "ADM2_EN"
    bound = conflict_df

    # bound, _ = get_bound_pop()
    # list_city = bound[adm_col].values
    # list_city = LIST_POP_CITY

    # coal_gdf = gpd.read_file(UK_COAL_SHP)
    # coal_gdf.crs = "EPSG:4326"
    # coal_gdf["buffer"] = coal_gdf.geometry.buffer(0.2, cap_style=3)

    list_geo = []
    list_crs = []

    # elif "war1" in event:
    #     tk1 = "02/24_02/28"
    #     event_bound = conflict_df.loc[conflict_df[f"conflict_{tk1}"] > 2]
    #     # list_city = event_bound[adm_col].to_list()
    #     list_city = event_bound[adm_col].to_list() + border_df[adm_col].to_list()
    #     # list_city = ["Kyiv"]
    if "war2" in event:
        event_bound = conflict_df.loc[
            conflict_df[f"conflict_{tk}"] > THRESHOLD_CONFLICT_POINT
        ]
        list_city = event_bound[adm_col].to_list()
        # list_city = event_bound[adm_col].to_list() + border_df[adm_col].to_list()

    for adm in list_city:
        geometry = bound.loc[bound[adm_col] == adm].geometry

        list_geo.append(geometry)
        list_crs.append(bound.crs)

    # for ppl_name in coal_gdf.name.values:
    #     geometry = coal_gdf.loc[coal_gdf["name"] == ppl_name]["buffer"].geometry
    #     list_geo.append(geometry)
    #     list_crs.append(coal_gdf.crs)

    for geometry, crs in zip(list_geo, list_crs):
        clip_arr = ds.rio.clip(geometry, crs)[var].values.reshape(-1)
        flat_array = np.concatenate((flat_array, clip_arr), axis=None)

    flat_array = flat_array[~np.isnan(flat_array)]

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
