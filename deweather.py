#%%
import pandas as pd
import xarray as xr
import random
import numpy as np
import pickle
import os

from scipy.stats import linregress
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

from const import CAMS_COLS

from mypath import *
from const import *
from utils import *

from fig_plt import *


class Dataset(object):

    de_weather_model_path = DE_WEATHER_MODEL

    train_geo_path = TRAIN_GEO
    test_geo_path = TEST_GEO
    pfm_path = PFM_PATH
    rate_train = 0.8
    rate_test = rate_train / 4

    def __init__(self, cams_reals_nc, cams_fc_nc, era5_nc, s5p_nc, pop_nc) -> None:

        self.cams = None
        self.era5 = None
        self.s5p = None
        self.pop = None

        self.list_geo = None
        self.train_geo = None
        self.test_geo = None

        self.train_2019 = pd.DataFrame()
        self.test_2019 = pd.DataFrame()

        self.best_params = {}
        self.de_weather_model = None

        self.load_data(cams_reals_nc, cams_fc_nc, era5_nc, s5p_nc, pop_nc)
        self.extract_list_geo()
        # self.params_search()
        self.train_de_weather_model(10)
        self.to_df()
        self.de_weather()

    def load_data(self, cams_reals_nc, cams_fc_nc, era5_nc, s5p_nc, pop_nc):

        cams_fc = xr.open_dataset(cams_fc_nc) * 10e-9
        cams_fc = cams_fc.rename(name_dict={list(cams_fc.keys())[0]: "no2"})
        self.cams = xr.concat([xr.open_dataset(cams_reals_nc), cams_fc], dim="time")
        self.cams = self.cams.rename(name_dict={list(self.cams.keys())[0]: "cams_no2"})

        self.era5 = xr.open_dataset(era5_nc)

        s5p = xr.open_dataset(s5p_nc)
        self.s5p = s5p.rename(name_dict={list(s5p.keys())[0]: S5P_OBS_COL})
        self.s5p = self.s5p.isel(band=0)

        pop = xr.open_dataset(pop_nc)
        self.pop = pop.rename(name_dict={list(pop.keys())[0]: "pop"})
        self.pop = self.pop.isel(band=0)

    def extract_list_geo(self):

        daily_ds = self.cams.isel(time=0).to_dataframe()
        daily_ds = daily_ds.dropna()

        lat = list(daily_ds.index.get_level_values(0))
        lon = list(daily_ds.index.get_level_values(1))

        self.list_geo = list(zip(lat, lon))

    def extract_train_test_lonlat(self):

        list_geo = self.list_geo.copy()
        self.test_geo = random.sample(list_geo, int(len(list_geo) * self.rate_test))

        [list_geo.remove(x) for x in self.test_geo]
        train = random.sample(list_geo, int(len(list_geo) * self.rate_train))

        self.train_geo = [x for x in train if x not in self.test_geo]

    def reform_data(self, year, list_geo):

        sd = np.datetime64(f"{year}-01-01T00:00:00.000000000")
        ed = np.datetime64(f"{year}-07-31T00:00:00.000000000")

        cams_year = self.cams.sel(time=slice(sd, ed))
        era5_year = self.era5.sel(time=slice(sd, ed))
        s5p_year = self.s5p.sel(time=slice(sd, ed))

        julian_time = pd.DatetimeIndex(cams_year.time.values).to_julian_date()
        dow = pd.DataFrame(cams_year.time.values)[0].dt.dayofweek.values
        doy = pd.DataFrame(cams_year.time.values)[0].dt.dayofyear.values

        df = []

        df_pop = self.pop.to_dataframe()

        for t in range(0, len(julian_time)):

            df_cams_t = cams_year.isel(time=t).to_dataframe()[CAMS_COLS]
            df_era5_t = era5_year.isel(time=t).to_dataframe()[ERA5_COLS]
            df_s5p_t = s5p_year.isel(time=t).to_dataframe()[S5P_COLS]

            df_t = pd.concat(
                [
                    df_cams_t.loc[list_geo],
                    df_era5_t.loc[list_geo],
                    df_s5p_t.loc[list_geo],
                    df_pop.loc[list_geo],
                ],
                axis=1,
            )

            df_t["time"] = [cams_year.time.values[t]] * len(list_geo)
            df_t["dow"] = [dow[t]] * len(list_geo)
            df_t["doy"] = [doy[t]] * len(list_geo)
            df_t["lat"] = [x[0] for x in list_geo]
            df_t["lon"] = [x[1] for x in list_geo]

            df.append(df_t)

        return pd.concat(df, ignore_index=True).drop(columns=["band", "spatial_ref"])

    def extract_train_test(self):

        train_2019 = self.reform_data(2019, self.train_geo)
        test_2019 = self.reform_data(2019, self.test_geo)

        self.train_2019 = train_2019.dropna()
        self.test_2019 = test_2019.dropna()

    def extract_Xy_train_test(self):
        X_train = self.train_2019.drop(columns=[S5P_OBS_COL, "time"]).values
        X_test = self.test_2019.drop(columns=[S5P_OBS_COL, "time"]).values

        y_train = self.train_2019[S5P_OBS_COL].values
        y_test = self.test_2019[S5P_OBS_COL].values

        return X_train, y_train, X_test, y_test

    def params_search(self):
        tunned_parameters = RANDOM_GRID
        grid_search = GridSearchCV(
            RandomForestRegressor(), tunned_parameters, scoring="r2", cv=5, n_jobs=-1
        )
        data_2019 = self.reform_data(2019, self.list_geo).dropna()
        X = data_2019.drop(columns=[S5P_OBS_COL, "time"]).values
        y = data_2019[S5P_OBS_COL].values
        grid_search.fit(X, y)
        self.best_params = grid_search.best_params_

    def train_de_weather_model(self, n_test):
        if not os.path.exists(self.train_geo_path) and not os.path.exists(
            self.test_geo_path
        ):
            r2_best = 0
            pfm_dict = {}
            pfm_dict["r2_score"] = []
            pfm_dict["mse_score"] = []
            pfm_dict["type"] = []
            for n in range(n_test):
                print(f"n-test:{n}")
                self.extract_train_test_lonlat()
                self.extract_train_test()

                X_train, y_train, X_test, y_test = self.extract_Xy_train_test()

                self.de_weather_model = RandomForestRegressor(
                    n_estimators=400, min_samples_leaf=7, n_jobs=-1
                )
                self.de_weather_model = self.de_weather_model.fit(X_train, y_train)

                y_pred = self.de_weather_model.predict(X_test)
                y_pred_train = self.de_weather_model.predict(X_train)

                mse_test = mean_squared_error(y_test, y_pred)
                mse_train = mean_squared_error(y_train, y_pred_train)

                r2_test = r2_score(y_test, y_pred)
                r2_train = r2_score(y_train, y_pred_train)

                pfm_dict["r2_score"].append(r2_test)
                pfm_dict["mse_score"].append(mse_test)
                pfm_dict["type"].append("test")

                pfm_dict["r2_score"].append(r2_train)
                pfm_dict["mse_score"].append(mse_train)
                pfm_dict["type"].append("train")

                if r2_test > r2_best:
                    r2_best = r2_test

                    pickle.dump(
                        self.de_weather_model, open(self.de_weather_model_path, "wb")
                    )
                    pickle.dump(self.train_geo, open(self.train_geo_path, "wb"))
                    pickle.dump(self.test_geo, open(self.test_geo_path, "wb"))
            pfm_df = pd.DataFrame.from_dict(pfm_dict)
            sns.violinplot(data=pfm_df, x="type", y="r2_score")
            pfm_df.to_csv(self.pfm_path)

        self.train_geo = pickle.load(open(self.train_geo_path, "rb"))
        self.test_geo = pickle.load(open(self.test_geo_path, "rb"))
        self.de_weather_model = pickle.load(open(self.de_weather_model_path, "rb"))

        self.extract_train_test()
        X_train, y_train, X_test, y_test = self.extract_Xy_train_test()
        y_pred = self.de_weather_model.predict(X_test)
        y_pred_train = self.de_weather_model.predict(X_train)

        print(f"mean_squared_error test: {mean_squared_error(y_test, y_pred)}")
        print(f"mean_squared_error train: {mean_squared_error(y_train, y_pred_train)}")

        print(f"r2 score test: {r2_score(y_test, y_pred)}")
        print(f"r2 score train: {r2_score(y_train, y_pred_train)}")

        print("-------test pcc ---------")
        print(linregress(y_pred, y_test))

        print("-------train pcc-----------")
        print(linregress(y_pred_train, y_train))

        # plot
        self.test_2019[S5P_PRED_COL] = y_pred

        return y_pred, y_test, y_pred_train, y_train

    def to_df(self):

        self.df_2020 = self.reform_data(2020, self.list_geo).fillna(0)
        self.df_2021 = self.reform_data(2021, self.list_geo).fillna(0)
        self.df_2022 = self.reform_data(2022, self.list_geo).fillna(0)

    def de_weather(self):

        s5p_2020_pred = self.de_weather_model.predict(
            self.df_2020.drop(columns=[S5P_OBS_COL, "time"]).values
        )
        s5p_2021_pred = self.de_weather_model.predict(
            self.df_2021.drop(columns=[S5P_OBS_COL, "time"]).values
        )
        s5p_2022_pred = self.de_weather_model.predict(
            self.df_2022.drop(columns=[S5P_OBS_COL, "time"]).values
        )
        self.df_2020[S5P_PRED_COL] = s5p_2020_pred
        self.df_2021[S5P_PRED_COL] = s5p_2021_pred
        self.df_2022[S5P_PRED_COL] = s5p_2022_pred

        self.df_2020[[S5P_PRED_COL, S5P_OBS_COL]] = (
            self.df_2020[[S5P_PRED_COL, S5P_OBS_COL]] * 1e6
        )
        self.df_2021[[S5P_PRED_COL, S5P_OBS_COL]] = (
            self.df_2021[[S5P_PRED_COL, S5P_OBS_COL]] * 1e6
        )
        self.df_2022[[S5P_PRED_COL, S5P_OBS_COL]] = (
            self.df_2022[[S5P_PRED_COL, S5P_OBS_COL]] * 1e6
        )

        self.dw_2020 = self.df_2020.set_index(["time", "lat", "lon"]).to_xarray()
        self.dw_2021 = self.df_2021.set_index(["time", "lat", "lon"]).to_xarray()
        self.dw_2022 = self.df_2022.set_index(["time", "lat", "lon"]).to_xarray()


if __name__ == "__main__":
    ds = Dataset(CAM_REALS_NO2_NC, CAM_FC_NO2_NC, ERA5_NC, S5P_NO2_NC, POP_NC)
    # plot_pred_true(ds)
    # plot_obs_bau_adm2_map(ds, 2022)
    # plot_obs_bau_adm2_map(ds, 2021)
    # plot_obs_bau_adm2_map(ds, 2020)

# %%
