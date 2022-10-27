#%%
import pandas as pd
import xarray as xr
import random
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

from scipy.stats import linregress
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from shapely.geometry import Point
from const import CAMS_COLS

from mypath import *
from const import *


class Dataset(object):
    def __init__(self, cams_reals_nc, cams_fc_nc, era5_nc, s5p_nc, pop_nc) -> None:

        self.cams = None
        self.era5 = None
        self.s5p = None
        self.pop = None

        self.train_geo = None
        self.test_geo = None

        self.train_2019 = pd.DataFrame()
        self.test_2019 = pd.DataFrame()

        self.non_deweather = {}
        self.deweather = {}

        self.load_data(cams_reals_nc, cams_fc_nc, era5_nc, s5p_nc, pop_nc)
        self.extract_train_test_lonlat()
        self.extract_train_test()

    def load_data(self, cams_reals_nc, cams_fc_nc, era5_nc, s5p_nc, pop_nc):

        cams_fc = xr.open_dataset(cams_fc_nc)
        cams_fc = cams_fc.rename(name_dict={list(cams_fc.keys())[0]: "no2"})
        self.cams = xr.concat([xr.open_dataset(cams_reals_nc), cams_fc], dim="time")
        self.cams = self.cams.rename(name_dict={list(self.cams.keys())[0]: "cams_no2"})

        self.era5 = xr.open_dataset(era5_nc)

        s5p = xr.open_dataset(s5p_nc)
        self.s5p = s5p.rename(name_dict={list(s5p.keys())[0]: "s5p_no2"})
        self.s5p = self.s5p.isel(band=0)

        pop = xr.open_dataset(pop_nc)
        self.pop = pop.rename(name_dict={list(pop.keys())[0]: "pop"})
        self.pop = self.pop.isel(band=0)

    def extract_train_test_lonlat(self):

        daily_ds = self.cams.isel(time=0).to_dataframe()
        daily_ds = daily_ds.dropna()

        lat = list(daily_ds.index.get_level_values(0))
        lon = list(daily_ds.index.get_level_values(1))

        list_geo = list(zip(lat, lon))

        self.test_geo = random.sample(list_geo, int(len(list_geo) * 0.05))

        [list_geo.remove(x) for x in self.test_geo]
        train = random.sample(list_geo, int(len(list_geo) * 0.25))

        self.train_geo = [x for x in train if x not in self.test_geo]

    def extract_train_test(self):

        sd = np.datetime64(f"2019-02-24T00:00:00.000000000")
        ed = np.datetime64(f"2019-07-31T00:00:00.000000000")

        cams_2019 = self.cams.sel(time=slice(sd, ed))
        era5_2019 = self.era5.sel(time=slice(sd, ed))
        s5p_2019 = self.s5p.sel(time=slice(sd, ed))

        julian_time = pd.DatetimeIndex(cams_2019.time.values).to_julian_date()
        dow = pd.DataFrame(cams_2019.time.values)[0].dt.dayofweek.values

        df_train = []
        df_test = []

        df_pop = self.pop.to_dataframe()

        for t in range(0, len(julian_time)):

            ds_cams_t = cams_2019.isel(time=t)
            ds_era5_t = era5_2019.isel(time=t)
            ds_s5p_t = s5p_2019.isel(time=t)

            df_cams_t = ds_cams_t.to_dataframe()[CAMS_COLS]
            df_era5_t = ds_era5_t.to_dataframe()[ERA5_COLS]
            df_s5p_t = ds_s5p_t.to_dataframe()[S5P_COLS]
            df_train_t = pd.concat(
                [
                    df_cams_t.loc[self.train_geo],
                    df_era5_t.loc[self.train_geo],
                    df_s5p_t.loc[self.train_geo],
                    df_pop.loc[self.train_geo],
                ],
                axis=1,
                # ignore_index=True,
            )
            # df_train_t["jdate"] = [julian_time.values[t]] * len(self.train_geo)
            df_train_t["time"] = [cams_2019.time.values[t]] * len(self.train_geo)
            df_train_t["dow"] = [dow[t]] * len(self.train_geo)
            df_train_t["lat"] = [x[0] for x in self.train_geo]
            df_train_t["lon"] = [x[1] for x in self.train_geo]

            df_train.append(df_train_t)

            df_test_t = pd.concat(
                [
                    df_cams_t.loc[self.test_geo],
                    df_era5_t.loc[self.test_geo],
                    df_s5p_t.loc[self.test_geo],
                    df_pop.loc[self.test_geo],
                ],
                axis=1,
                #     ignore_index=True,
            )

            # df_test_t["jdate"] = [julian_time.values[t]] * len(self.test_geo)
            df_test_t["time"] = [cams_2019.time.values[t]] * len(self.test_geo)
            df_test_t["dow"] = [dow[t]] * len(self.test_geo)
            df_test_t["lat"] = [x[0] for x in self.test_geo]
            df_test_t["lon"] = [x[1] for x in self.test_geo]

            df_train.append(df_train_t)
            df_test.append(df_test_t)

        self.train_2019 = pd.concat(df_train, ignore_index=True)
        self.test_2019 = pd.concat(df_test, ignore_index=True)

        self.train_2019 = self.train_2019.dropna().drop(columns=["band", "spatial_ref"])
        self.test_2019 = self.test_2019.dropna().drop(columns=["band", "spatial_ref"])


def build_deweather_model(ds):

    # model = xgb.XGBRegressor()

    train = ds.train_2019
    test = ds.test_2019

    X_train = train.drop(columns=["s5p_no2", "time"]).values
    X_test = test.drop(columns=["s5p_no2", "time"]).values

    y_train = train["s5p_no2"].values
    y_test = test["s5p_no2"].values

    np.save(X_TRAIN_NPY, X_train)
    np.save(Y_TRAIN_NPY, y_train)
    np.save(X_TEST_NPY, X_test)
    np.save(Y_TEST_NPY, y_test)

    if os.path.exists(DE_WEATHER_MODEL):
        model = pickle.load(open(DE_WEATHER_MODEL, "rb"))
    else:
        model = RandomForestRegressor()
        model = model.fit(X_train, y_train)
        pickle.dump(model, open(DE_WEATHER_MODEL, "wb"))

    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    print(f"mean_squared_error test: {mean_squared_error(y_test, y_pred)}")
    print(f"mean_squared_error train: {mean_squared_error(y_train, y_pred_train)}")

    print(f"r2 score test: {r2_score(y_test, y_pred)}")
    print(f"r2 score train: {r2_score(y_train, y_pred_train)}")

    print("-------test pcc ---------")
    print(linregress(y_pred, y_test))

    print("-------train pcc-----------")
    print(linregress(y_pred_train, y_train))

    return y_pred, y_test, y_pred_train, y_train


if __name__ == "__main__":

    ds = Dataset(CAM_REALS_NO2_NC, CAM_FC_NO2_NC, ERA5_NC, S5P_NO2_NC, POP_NC)

    y_pred, y_test, y_pred_train, y_train = build_deweather_model(ds)
    test_df = ds.test_2019
    test_df["s5p_no2_pred"] = y_pred
    time_df = test_df.groupby("time").mean()
    time_df[["s5p_no2", "s5p_no2_pred"]].plot.line()
    # cams_reals_no2 = xr.open_dataset(CAM_REALS_NO2_NC)
    # cams_fc_no2 = xr.open_dataset(CAM_FC_NO2_NC)
    # cams_fc_no2 = cams_fc_no2.rename(name_dict={list(cams_fc_no2.keys())[0]: "no2"})

    # cams_no2 = xr.concat([cams_reals_no2, cams_fc_no2], dim="time")
    # era5 = xr.open_dataset(ERA5_NC)
    # s5p_no2 = xr.open_dataset(S5P_NO2_NC)

    # pop = xr.open_dataset(POP_NC)

    # years = [2019, 2020, 2021, 2022]

    # for y in years:

    #     sd = np.datetime64(f"{y}-02-24T00:00:00.000000000")
    #     ed = np.datetime64(f"{y}-07-31T00:00:00.000000000")

    #     cams_year = cams_no2.sel(time=slice(sd, ed))
    #     era5_year = era5.sel(time=slice(sd, ed))
    #     s5p_year = s5p_no2.sel(time=slice(sd, ed))
# %%
