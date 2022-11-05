#%%
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


from utils import *
from const import *


def prep_ds(org_ds, year):

    if year == 2020:
        ds = org_ds.dw_2020
    elif year == 2021:
        ds = org_ds.dw_2021
    else:
        ds = org_ds.dw_2022

    ds = ds.rio.write_crs("epsg:4326", inplace=True)
    return ds.rio.set_spatial_dims("lon", "lat", inplace=True)


def prep_s5p_ds():
    org_ds = xr.open_dataset(S5P_NO2_NC)
    var_name = list(org_ds.keys())[0]
    org_ds = org_ds.rename(name_dict={var_name: "s5p_no2"})
    org_ds = org_ds.rio.write_crs("epsg:4326", inplace=True)
    org_ds = org_ds.rio.set_spatial_dims("lon", "lat", inplace=True)
    return org_ds


def plot_trend_line(ds, title):

    import matplotlib.dates as mdates

    month_day_fmt = mdates.DateFormatter("%b %d")
    figure, ax = plt.subplots(figsize=(16, 8))

    years = [2020, 2021, 2022]

    sd = np.datetime64(f"2019-01-01T00:00:00.000000000")
    ed = np.datetime64(f"2019-08-01T00:00:00.000000000")
    ds_2019 = ds.sel(time=slice(sd, ed)).mean(dim=["lat", "lon"])["s5p_no2"]
    ax.plot(ds_2019.time, ds_2019.values, label="2019")
    x = ds_2019.time
    for year in years:
        sd = np.datetime64(f"{year}-01-01T00:00:00.000000000")
        ed = np.datetime64(f"{year}-08-01T00:00:00.000000000")
        ds_year = ds.sel(time=slice(sd, ed)).mean(dim=["lat", "lon"])["s5p_no2"]

        y = ds_year.values[:-1] if year == 2020 else ds_year.values
        ax.plot(x, y, label=year)
    ax.legend()
    ax.set_title(title)
    ax.xaxis.set_major_formatter(month_day_fmt)


def plot_obs_pop_line():

    org_ds = prep_s5p_ds()

    bound_lv2, crs = get_bound_pop_lv2()
    list_city = bound_lv2["ADM2_EN"].values

    for city in list_city:
        geometry = bound_lv2.loc[bound_lv2["ADM2_EN"] == city].geometry
        ds_city = org_ds.rio.clip(geometry, crs)
        plot_trend_line(ds_city, city)


def plot_change_bubble(geo_df, cols):
    cmap = "bwr"
    for col in cols:

        figure, ax = plt.subplots(figsize=(16, 8))
        bound_lv1 = gpd.read_file(UK_SHP_ADM1)
        bound_lv1.plot(ax=ax, facecolor="white", edgecolor="black", lw=0.7)

        g = sns.scatterplot(
            data=geo_df,
            x=geo_df.centroid.x,
            y=geo_df.centroid.y,
            hue=col,
            # hue_norm=(-20, 20),
            size="Population",
            sizes=(150, 500),
            palette=cmap,
            ax=ax,
        )

        g.legend(bbox_to_anchor=(1.0, 1.0), ncol=1)

        # norm = plt.Normalize(-30, 30)
        norm = plt.Normalize(geo_df[col].min(), geo_df[col].max())
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        clb = g.figure.colorbar(sm)
        clb.ax.set_ylabel(r"NO$_{2}$ col. change (%)")
        clb.ax.yaxis.set_label_position("right")
        g.set(title=col)

        h, l = g.get_legend_handles_labels()
        plt.legend(
            h[-7:],
            l[-7:],
            bbox_to_anchor=(1, 1),
            loc="upper right",
            borderaxespad=0.0,
            fontsize=13,
        )


def plot_obs_pop_bubble():

    org_ds = prep_s5p_ds()
    bound_lv2, crs = get_bound_pop_lv2()
    adm_col = "ADM2_EN"
    list_city = bound_lv2[adm_col].values

    sd_ed = PERIOD_DICT[2022]

    years = [2019, 2020, 2021, 2022]

    obs_dict_year = {}

    for year in years:
        for tk in sd_ed.keys():
            obs_dict_year[f"{year}_{tk}"] = []
            t = sd_ed[tk]
            sd = np.datetime64(f"{year}-{t['sm']}-{t['sd']}T00:00:00.000000000")
            ed = np.datetime64(f"{year}-{t['em']}-{t['ed']}T00:00:00.000000000")

            for city in list_city:
                geometry = bound_lv2.loc[bound_lv2[adm_col] == city].geometry
                adm_ds = (
                    org_ds.rio.clip(geometry, crs)
                    .mean(dim=["lat", "lon"])
                    .sel(time=slice(sd, ed))
                    .mean("time")[["s5p_no2"]]
                )
                obs_dict_year[f"{year}_{tk}"].append(adm_ds["s5p_no2"].item())
            bound_lv2[f"{year}_{tk}"] = obs_dict_year[f"{year}_{tk}"]

    change_dict = {}
    # for year in years[:-1]:
    for tk in sd_ed.keys():
        change_dict[f"2022_2021_{tk}"] = (
            bound_lv2[f"2022_{tk}"] - bound_lv2[f"2021_{tk}"]
        )

    df_no2_change = pd.DataFrame.from_dict(change_dict)
    df_no2_change["Population"] = bound_lv2["Population"].values
    geo_df = gpd.GeoDataFrame(
        df_no2_change, crs=crs, geometry=bound_lv2.geometry.centroid
    )

    plot_change_bubble(geo_df, list(change_dict.keys()))
    return geo_df


def plot_obs_adm2_map():
    org_ds = prep_s5p_ds()

    bound_lv2 = gpd.read_file(UK_SHP_ADM2)
    return


def plot_obs_bau_adm2_map(org_ds, year):

    ds = prep_ds(org_ds, year)

    bound_lv2 = gpd.read_file(UK_SHP_ADM2)
    sd_ed = PERIOD_DICT[year]
    adm_col = "ADM2_EN"

    dict_no2_change = {}
    list_adm1 = bound_lv2[adm_col].values

    for tk in sd_ed.keys():
        dict_no2_change[tk] = []

        t = sd_ed[tk]
        sd = np.datetime64(f"{year}-{t['sm']}-{t['sd']}T00:00:00.000000000")
        ed = np.datetime64(f"{year}-{t['em']}-{t['ed']}T00:00:00.000000000")

        for adm1 in list_adm1:
            geometry = bound_lv2.loc[bound_lv2[adm_col] == adm1].geometry
            adm_ds = (
                ds.rio.clip(geometry, bound_lv2.crs)
                .mean(dim=["lat", "lon"])
                .sel(time=slice(sd, ed))
                .mean("time")[["s5p_no2_pred", "s5p_no2"]]
            )
            dict_no2_change[tk].append(
                (adm_ds["s5p_no2"].item() - adm_ds["s5p_no2_pred"].item())
                * 100
                / adm_ds["s5p_no2_pred"].item()
            )
        bound_lv2[tk] = dict_no2_change[tk]
    for tk in sd_ed.keys():
        figure, ax = plt.subplots(figsize=(16, 8))
        bound_lv2.plot(
            column=tk,
            ax=ax,
            legend=True,
            cmap="bwr",
            vmin=-20,
            vmax=20,
            legend_kwds={"label": r"NO$_{2}$ col. change (%)"},
        )
        plt.title(tk, fontsize=18)


def plot_obs_bau_pop_bubble(org_ds, year):

    ds = prep_ds(org_ds)

    bound_lv1 = gpd.read_file(UK_SHP_ADM1)
    bound_lv2, crs = get_bound_pop_lv2()

    list_city = bound_lv2["ADM2_EN"].values
    sd_ed = PERIOD_DICT[year]

    dict_no2_change = {}

    for tk in sd_ed.keys():

        dict_no2_change[tk] = []

        t = sd_ed[tk]
        sd = np.datetime64(f"{year}-{t['sm']}-{t['sd']}T00:00:00.000000000")
        ed = np.datetime64(f"{year}-{t['em']}-{t['ed']}T00:00:00.000000000")

        for city in list_city:

            geometry = bound_lv2.loc[bound_lv2["ADM2_EN"] == city].geometry
            city_ds = (
                ds.rio.clip(geometry, crs)
                .mean(dim=["lat", "lon"])
                .sel(time=slice(sd, ed))
                .mean("time")[["s5p_no2_pred", "s5p_no2"]]
            )
            dict_no2_change[tk].append(
                (city_ds["s5p_no2"].item() - city_ds["s5p_no2_pred"].item())
                * 100
                / city_ds["s5p_no2_pred"].item()
            )

    df_no2_change = pd.DataFrame.from_dict(dict_no2_change)
    df_no2_change["Population"] = bound_lv2["Population"].values
    geo_df = gpd.GeoDataFrame(
        df_no2_change, crs=crs, geometry=bound_lv2.geometry.centroid
    )
    cmap = "bwr"
    for col in sd_ed.keys():

        figure, ax = plt.subplots(figsize=(16, 8))
        bound_lv1 = gpd.read_file(UK_SHP_ADM1)
        bound_lv1.plot(ax=ax, facecolor="white", edgecolor="black", lw=0.7)

        g = sns.scatterplot(
            data=geo_df,
            x=geo_df.centroid.x,
            y=geo_df.centroid.y,
            hue=col,
            hue_norm=(-20, 20),
            size="Population",
            sizes=(150, 500),
            palette=cmap,
            ax=ax,
        )

        g.legend(bbox_to_anchor=(1.0, 1.0), ncol=1)

        norm = plt.Normalize(-30, 30)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        clb = g.figure.colorbar(sm)
        clb.ax.set_ylabel(r"NO$_{2}$ col. change (%)")
        clb.ax.yaxis.set_label_position("right")
        g.set(title=rf"NO$_{2}$_OBS - NO$_{2}$_BAU {col} - {year}")

        h, l = g.get_legend_handles_labels()
        plt.legend(
            h[-7:],
            l[-7:],
            bbox_to_anchor=(1, 1),
            loc="upper right",
            borderaxespad=0.0,
            fontsize=13,
        )

    return geo_df


def plot_obs_bau_pop_line(ds):
    ds = ds.rio.write_crs("epsg:4326", inplace=True)
    ds = ds.rio.set_spatial_dims("lon", "lat", inplace=True)
    # bound_lv2, crs = get_bound_pop_lv2()
    bound_lv2 = gpd.read_file(UK_SHP_ADM2)
    city_no2 = {}
    for i, city in enumerate(bound_lv2["ADM2_EN"].values):
        fig = plt.figure(1 + i, figsize=(16, 8))
        ax = plt.subplot(1, 1, 1)
        geometry = bound_lv2.loc[bound_lv2["ADM2_EN"] == city].geometry
        ds_clip = ds.rio.clip(geometry, bound_lv2.crs).mean(dim=["lat", "lon"])[
            ["s5p_no2_pred", "s5p_no2"]
        ]
        city_no2[city] = ds_clip
        df = ds_clip.to_dataframe()
        df[["s5p_no2_pred", "s5p_no2"]].plot.line(ax=ax)
        ax.set_title(f"{city}")


def plot_obs_year():
    org_ds = prep_s5p_ds()

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