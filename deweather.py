#%%
import pandas as pd
import xarray as xr
import random
import numpy as np
import pickle
import os
import math
import lightgbm as lgbm

from scipy.stats import linregress
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from flaml.model import LGBMEstimator
from flaml import AutoML
from flaml.default import LGBMRegressor
from flaml import tune

from mypath import *
from const import *
from utils import *
from fig_plt import *


class GPULGBM(LGBMEstimator):
    def __init__(self, **config):
        super().__init__(device="gpu", **config)


class Dataset(object):

    de_weather_model_train_path = DE_WEATHER_MODEL_TRAIN
    de_weather_model_pred_path = DE_WEATHER_MODEL_PRED

    train_geo_path = TRAIN_GEO
    test_geo_path = TEST_GEO
    pfm_path = PFM_PATH
    rate_train = 0.8
    rate_test = 0.2

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
        self.de_weather_model_train = None
        self.de_weather_model_pred = None

        self.load_data(cams_reals_nc, cams_fc_nc, era5_nc, s5p_nc, pop_nc)
        self.cal_rh()
        self.extract_list_geo()

        # self.extract_train_test_lonlat()
        # self.extract_train_test()

        # self.params_search()
        self.train_de_weather_model()

        # self.build_deweather_pred()
        # self.to_df()
        # self.de_weather()

    def load_data(self, cams_reals_nc, cams_fc_nc, era5_nc, s5p_nc, pop_nc):

        cams_fc = xr.open_dataset(cams_fc_nc) * 1e9
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

    def cal_rh(self):
        beta = 17.625
        lamda = 243.04
        e = math.e

        dp = self.era5["d2m"] - 272.15
        t = self.era5["t2m"] - 272.15

        self.era5["relative humidity"] = 100 * (
            (e ** ((beta * dp) / (lamda + dp))) / (e ** ((beta * t) / (lamda + t)))
        )

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
        # train = random.sample(list_geo, int(len(list_geo) * self.rate_train))

        # self.train_geo = [x for x in train if x not in self.test_geo]
        self.train_geo = list_geo
        print("train/test samples: ", len(self.train_geo), len(self.test_geo))

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

        org_data = pd.concat(df, ignore_index=True).drop(
            columns=["band", "spatial_ref"]
        )

        # feb29 = np.datetime64(f"2020-02-29T00:00:00.000000000")
        # org_data = org_data[org_data["time"] != feb29]

        # cols = ["dow", "doy", "lat", "lon"]
        # oh_data = pd.get_dummies(org_data, columns=cols)
        # return org_data, oh_data

        return org_data

    def extract_train_test(self):

        train_2019 = self.reform_data(2019, self.train_geo)
        test_2019 = self.reform_data(2019, self.test_geo)

        self.train_2019 = train_2019.dropna()
        self.test_2019 = test_2019.dropna()

        # self.train_2019, self.test_2019 = drop_outlier(
        #     train_2019, test_2019, [S5P_OBS_COL]
        # )

    def extract_Xy_train_test(self):
        X_train = self.train_2019.drop(columns=[S5P_OBS_COL, "time"]).values
        X_test = self.test_2019.drop(columns=[S5P_OBS_COL, "time"]).values

        y_train = self.train_2019[S5P_OBS_COL].values
        y_test = self.test_2019[S5P_OBS_COL].values

        return X_train, y_train, X_test, y_test

    def params_search(self):
        tunned_parameters = LGBM_HYP_PARAMS
        # self.auto_model = flaml.default.LGBMRegressor(
        #     categorical_feature=[8, 9, 10, 11], device="gpu"
        # )

        automl = AutoML()
        settings = {
            "time_budget": 60 * 720,  # total running time in seconds
            "metric": "rmse",  # primary metrics for regression can be chosen from: ['mae','mse','r2']
            # "estimator_list": ['lgbm'],  # list of ML learners; we tune lightgbm in this example
            "task": "regression",  # task type
            # "log_file_name": "houses_experiment.log",  # flaml log file
            "seed": 7654321,  # random seed
            "custom_hp": {
                "gpu_lgbm": {
                    "log_max_bin": {
                        "domain": tune.lograndint(lower=3, upper=7),
                        "init_value": 5,
                    },
                }
            },
        }
        automl.add_learner(learner_name="gpu_lgbm", learner_class=GPULGBM)
        settings["estimator_list"] = ["gpu_lgbm"]  # change the estimator list

        data_2019 = self.reform_data(2019, self.list_geo)
        data_2019 = data_2019.dropna()

        X = data_2019.drop(columns=[S5P_OBS_COL, "time"])
        y = data_2019[S5P_OBS_COL]

        #
        automl.fit(X_train=X, y_train=y, **settings)
        print("Best hyperparmeter config:", automl.best_config)
        print("Best r2 on validation data: {0:.4g}".format(1 - automl.best_loss))
        print(
            "Training duration of best run: {0:.4g} s".format(
                automl.best_config_train_time
            )
        )
        print(automl.model.estimator)
        plt.barh(automl.feature_names_in_, automl.feature_importances_)
        self.auto_model = automl
        # self.best_params = self.auto_model.best_config

        # print(self.best_params)
        # plt.barh(
        #     self.auto_model.feature_names_in_, self.auto_model.feature_importances_
        # )

    def train_de_weather_model(self):
        # if not os.path.exists(self.train_geo_path) and not os.path.exists(
        #     self.test_geo_path
        # ):

        self.extract_train_test_lonlat()
        self.extract_train_test()

        X_train, y_train, X_test, y_test = self.extract_Xy_train_test()

        # self.de_weather_model_train = RandomForestRegressor(
        #     n_estimators=400, min_samples_leaf=7, n_jobs=-1
        # )
        model = lgbm.LGBMRegressor(**LGBM_HYP_PARAMS)
        self.de_weather_model_train = model.fit(
            X_train,
            y_train,
        )

        y_pred = self.de_weather_model_train.predict(X_test)
        y_pred_train = self.de_weather_model_train.predict(X_train)

        self.test_2019[S5P_PRED_COL] = y_pred
        self.train_2019[S5P_PRED_COL] = y_pred_train

        rmse_test = mean_squared_error(y_test, y_pred, squared=False)
        rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)

        r2_test = r2_score(y_test, y_pred)
        r2_train = r2_score(y_train, y_pred_train)

        norm_rmse_test_2 = np.sum(np.square(y_test - y_pred)) / np.sum(np.square(y_test))
        norm_rmse_train_2 = np.sum(np.square(y_train - y_pred_train)) / np.sum(np.square(y_train))
        
        norm_rmse_test = norm_rmse_test_2, rmse_test / np.amax(y_test), rmse_test / (np.amax(y_test) - np.amin(y_test))
        norm_rmse_train = norm_rmse_train_2, rmse_train / np.amax(y_train), rmse_train / (np.amax(y_train) - np.amin(y_train))

        MBE_test = np.mean(y_pred - y_test)
        MBE_train = np.mean(y_pred_train - y_train)

        norm_MBE_test = np.sum(y_pred - y_test) / np.sum(y_test)
        norm_MBE_train = np.sum(y_pred_train - y_train) / np.sum(y_train)

        print(f"rmse test: {rmse_test}")
        print(f"rmse train: {rmse_train}")

        print(f"norm rmse test: {norm_rmse_test}")
        print(f"norm rmse train: {norm_rmse_train}")

        print(f"mbe test: {MBE_test}")
        print(f"mbe train: {MBE_train}")

        print(f"norm mbe test: {norm_MBE_test}")
        print(f"norm mbe train: {norm_MBE_train}")

        print(f"r2 score test: {r2_test}")
        print(f"r2 score train: {r2_train}")

        # print("-------test pcc ---------")
        # print(linregress(y_pred, y_test))

        # print("-------train pcc-----------")
        # print(linregress(y_pred_train, y_train))

    def build_deweather_pred(self):
        if not os.path.exists(self.de_weather_model_pred_path):

            # old model
            # self.de_weather_model_pred = RandomForestRegressor(
            #     n_estimators=800, min_samples_leaf=10, n_jobs=-1
            # )
            model = lgbm.LGBMRegressor(**LGBM_HYP_PARAMS)
            self.de_weather_model_pred = model
            train = self.reform_data(2019, self.list_geo)
            train = train.dropna()

            X_train = train.drop(columns=[S5P_OBS_COL, "time"])
            y_train = train[S5P_OBS_COL]
            self.de_weather_model_pred = self.de_weather_model_pred.fit(
                X_train, y_train
            )
            # pickle.dump(
            #     self.de_weather_model_pred, open(self.de_weather_model_pred_path, "wb")
            # )
            return
        # self.de_weather_model_pred = pickle.load(
        #     open(self.de_weather_model_pred_path, "rb")
        # )

    def to_df(self):

        self.df_2019 = self.reform_data(2019, self.list_geo)
        self.df_2020 = self.reform_data(2020, self.list_geo)
        self.df_2021 = self.reform_data(2021, self.list_geo)
        self.df_2022 = self.reform_data(2022, self.list_geo)

    # def prep_oh(self):
    #     self.oh_2019 = self.oh_2019.fillna(0)
    #     self.oh_2020 = self.oh_2020.fillna(0)
    #     self.oh_2021 = self.oh_2021.fillna(0)
    #     self.oh_2022 = self.oh_2022.fillna(0)

    #     for col in NONE_OH_COLS:
    #         max = self.df_2019[col].max()
    #         min = self.df_2019[col].min()
    #         print(col, min, max)

    #         self.oh_2019[col] = (self.oh_2019[col] - min) / (max - min)
    #         self.oh_2020[col] = (self.oh_2020[col] - min) / (max - min)
    #         self.oh_2021[col] = (self.oh_2021[col] - min) / (max - min)
    #         self.oh_2022[col] = (self.oh_2022[col] - min) / (max - min)

    def de_weather(self):

        # norm
        s5p_2019_pred = self.de_weather_model_pred.predict(
            self.df_2019.drop(columns=[S5P_OBS_COL, "time"]).values
        )
        s5p_2020_pred = self.de_weather_model_pred.predict(
            self.df_2020.drop(columns=[S5P_OBS_COL, "time"]).values
        )
        s5p_2021_pred = self.de_weather_model_pred.predict(
            self.df_2021.drop(columns=[S5P_OBS_COL, "time"]).values
        )
        s5p_2022_pred = self.de_weather_model_pred.predict(
            self.df_2022.drop(columns=[S5P_OBS_COL, "time"]).values
        )

        self.df_2019[S5P_PRED_COL] = s5p_2019_pred
        self.df_2020[S5P_PRED_COL] = s5p_2020_pred
        self.df_2021[S5P_PRED_COL] = s5p_2021_pred
        self.df_2022[S5P_PRED_COL] = s5p_2022_pred

        self.df_2019[[S5P_PRED_COL, S5P_OBS_COL]] = (
            self.df_2019[[S5P_PRED_COL, S5P_OBS_COL]] * 1e6
        )
        self.df_2020[[S5P_PRED_COL, S5P_OBS_COL]] = (
            self.df_2020[[S5P_PRED_COL, S5P_OBS_COL]] * 1e6
        )
        self.df_2021[[S5P_PRED_COL, S5P_OBS_COL]] = (
            self.df_2021[[S5P_PRED_COL, S5P_OBS_COL]] * 1e6
        )
        self.df_2022[[S5P_PRED_COL, S5P_OBS_COL]] = (
            self.df_2022[[S5P_PRED_COL, S5P_OBS_COL]] * 1e6
        )

        self.dw_2019 = self.df_2019.set_index(["time", "lat", "lon"]).to_xarray()
        self.dw_2020 = self.df_2020.set_index(["time", "lat", "lon"]).to_xarray()
        self.dw_2021 = self.df_2021.set_index(["time", "lat", "lon"]).to_xarray()
        self.dw_2022 = self.df_2022.set_index(["time", "lat", "lon"]).to_xarray()


# if __name__ == "__main__":
#     ds = Dataset(CAM_REALS_NO2_NC, CAM_FC_NO2_NC, ERA5_NC, S5P_NO2_NC, POP_NC)


# plot_ppl_obs_bau_line_mlt(ds)
# plot_obs_bau_pop_line_mlt(ds)
# plot_obs_bau_bubble(ds, year)
# plot_obs_bubble("covid")
# plot_weather_params(ds, event="covid")
# plot_obs_bau_adm2(ds, 2022, "2_no2_bau")

# %%
