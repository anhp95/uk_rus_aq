#%%
import xarray as xr
import geopandas as gpd

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm

import seaborn as sns
import numpy as np
import metpy.calc as mpcalc
import datashader as dsh

from metpy.units import units
from datashader.mpl_ext import dsshow
from windrose import WindroseAxes

from utils import *
from const import *

# Bubble plot
def plot_obs_change_map(event="war"):

    bound_lv1 = gpd.read_file(UK_SHP_ADM1)
    coal_gdf = gpd.read_file(UK_COAL_SHP)

    org_ds = prep_s5p_ds()

    year_target = 2022 if event == "war" else 2020
    sd_ed = PERIOD_DICT[year_target]
    year_srcs = [i for i in range(2019, year_target)]

    tks = list(sd_ed.keys())

    pixel_change_dict = {}

    for i, year in enumerate(year_srcs + [year_target]):

        for tk in tks:
            t = sd_ed[tk]
            sd = np.datetime64(f"{year}-{t['sm']}-{t['sd']}{HOUR_STR}")
            ed = np.datetime64(f"{year}-{t['em']}-{t['ed']}{HOUR_STR}")
            pixel_change_dict[f"{year}_{tk}"] = org_ds.sel(time=slice(sd, ed)).mean(
                "time"
            )[[S5P_OBS_COL]]

    # Plot self changes for pixel
    nrow = int((len(year_srcs) + 1) / 2)
    self_figure, self_ax = plt.subplots(
        nrow, 2, figsize=(10, 4 * nrow), layout="constrained"
    )
    j = 0
    for i, year in enumerate(year_srcs + [year_target]):
        i = int(i / 2)
        j = 0 if j > 1 else j
        ds_change = (
            (
                pixel_change_dict[f"{year}_{tks[1]}"]
                - pixel_change_dict[f"{year}_{tks[0]}"]
            )
            * 100
            / pixel_change_dict[f"{year}_{tks[0]}"]
        )[S5P_OBS_COL]

        pcm = ds_change.plot(
            ax=self_ax[i][j], cmap="coolwarm", vmin=-70, vmax=70, add_colorbar=False
        )
        coal_gdf.plot(
            ax=self_ax[i][j], color="green", markersize=30, label="Coal power plant"
        )
        bound_lv1.plot(ax=self_ax[i][j], facecolor="None", edgecolor="black", lw=0.2)
        # plt.title(f"OBS_2022_{y}_{tk}", fontsize=18)
        self_ax[i][j].set_xlabel("")
        self_ax[i][j].set_ylabel("")
        self_ax[i][j].legend(
            bbox_to_anchor=(0, 0),
            loc="lower left",
        )
        self_ax[i][j].set_title(year)

        self_ax[i][j].set_xlim([22, 41])
        self_ax[i][j].set_ylim([44, 53])
        j += 1

    self_figure.colorbar(
        pcm,
        ax=self_ax[:, :],
        orientation="horizontal",
        extend="both",
        label="NO$_{2}$ col. change (%)",
        location="bottom",
        shrink=0.4,
    )
    plt.suptitle(rf"Observed_NO$_{2}$_Difference_{year_target}", fontsize=18)
    # plt.tight_layout()

    # Plot inter changes
    nrows = len(tks)
    ncols = 3
    inter_figure, inter_ax = plt.subplots(
        nrows, ncols, figsize=(6 * ncols, 4 * nrows), layout="constrained"
    )
    for i, tk in enumerate(tks):
        for j, year in enumerate(year_srcs):

            ds_change = (
                (
                    pixel_change_dict[f"{year_target}_{tk}"]
                    - pixel_change_dict[f"{year}_{tk}"]
                )
                * 100
                / pixel_change_dict[f"{year}_{tk}"]
            )[S5P_OBS_COL]

            pcm = ds_change.plot(
                ax=inter_ax[i][j],
                cmap="coolwarm",
                vmin=-70,
                vmax=70,
                add_colorbar=False,
            )
            coal_gdf.plot(
                ax=inter_ax[i][j],
                color="green",
                markersize=30,
                label="Coal power plant",
            )
            bound_lv1.plot(
                ax=inter_ax[i][j], facecolor="None", edgecolor="black", lw=0.2
            )
            # plt.title(f"OBS_2022_{y}_{tk}", fontsize=18)
            inter_ax[i][j].set_xlabel("")
            inter_ax[i][j].set_ylabel("")
            inter_ax[i][j].legend(
                bbox_to_anchor=(0, 0),
                loc="lower left",
            )
            inter_ax[i][j].set_title(f"{year}_{tk}")

            inter_ax[i][j].set_xlim([22, 41])
            inter_ax[i][j].set_ylim([44, 53])

    inter_figure.colorbar(
        pcm,
        ax=inter_ax[:, :],
        orientation="horizontal",
        extend="both",
        label="NO$_{2}$ col. change (%)",
        location="bottom",
        shrink=0.4,
    )


#%%
# Px plot
def plot_obs_bau_map(org_ds, year):

    ds = prep_ds(org_ds, year)
    coal_gdf = gpd.read_file(UK_COAL_SHP)
    sd_ed = PERIOD_DICT[2022]
    adm_col = "ADM2_EN"

    tks = list(sd_ed.keys())[1:]
    nrow = int(len(tks) / 2)
    ncols = 2
    figure, ax = plt.subplots(
        nrow, ncols, figsize=(6 * ncols, 4 * nrow), layout="constrained"
    )
    j = 0
    for i, tk in enumerate(tks):

        i = int(i / 2)
        j = 0 if j > 1 else j

        t = sd_ed[tk]
        sd = np.datetime64(f"{year}-{t['sm']}-{t['sd']}T00:00:00.000000000")
        ed = np.datetime64(f"{year}-{t['em']}-{t['ed']}T00:00:00.000000000")

        tk_ds = ds.sel(time=slice(sd, ed)).mean("time")[[S5P_PRED_COL, S5P_OBS_COL]]
        change_ds = (
            (tk_ds[S5P_OBS_COL] - tk_ds[S5P_PRED_COL]) * 100 / tk_ds[S5P_PRED_COL]
        )

        bound_lv1 = gpd.read_file(UK_SHP_ADM1)
        bound_lv0 = gpd.read_file(UK_SHP_ADM0)

        pcm = change_ds.plot(
            ax=ax[i][j],
            cmap="bwr",
            vmin=-40,
            vmax=40,
            add_colorbar=False
            # legend=False
            # cbar_kwargs={
            #     "label": r"NO$_{2}$ col. change (%)",
            #     "orientation": "horizontal",
            #     "fraction": 0.047,
            #     "extend": "both",
            # },
        )
        # bound_lv0.plot(ax=ax[i][j], facecolor="None", edgecolor="black", lw=2)
        coal_gdf.plot(
            ax=ax[i][j], color="green", markersize=30, label="Coal power plant"
        )
        bound_lv1.plot(ax=ax[i][j], facecolor="None", edgecolor="black", lw=0.2)
        ax[i][j].set_xlabel("")
        ax[i][j].set_ylabel("")
        ax[i][j].legend(
            bbox_to_anchor=(0, 0),
            loc="lower left",
        )
        ax[i][j].set_title(tk)

        ax[i][j].set_xlim([22, 41])
        ax[i][j].set_ylim([44, 53])
        j += 1
        # plt.title(tk, fontsize=18)
    figure.colorbar(
        pcm,
        ax=ax[:, :],
        orientation="horizontal",
        extend="both",
        label="NO$_{2}$ col. change (%)",
        location="bottom",
        shrink=0.4,
    )
    plt.suptitle(
        rf"Observed_Deweathered_NO$_{2}$_Difference_{year} (Pixel level)", fontsize=18
    )


# %%
def plot_trend_line(ds, title):

    import matplotlib.dates as mdates

    month_day_fmt = mdates.DateFormatter("%b %d")
    figure, ax = plt.subplots(figsize=(6, 4))

    years = [2020, 2021, 2022]

    sd = np.datetime64(f"2019-01-01T00:00:00.000000000")
    ed = np.datetime64(f"2019-07-31T00:00:00.000000000")
    ds_2019 = ds.sel(time=slice(sd, ed)).mean(dim=["lat", "lon"])[S5P_OBS_COL]
    # ax.plot(ds_2019.time, ds_2019.values, label="2019")
    x = ds_2019.time.values
    print(x)

    change_dict = {}
    change_dict["time"] = x
    change_dict[2019] = ds_2019.values.flatten()

    for year in years:
        sd = np.datetime64(f"{year}-01-01T00:00:00.000000000")
        ed = np.datetime64(f"{year}-07-31T00:00:00.000000000")
        ds_year = ds.sel(time=slice(sd, ed)).mean(dim=["lat", "lon"])[S5P_OBS_COL]
        y = ds_year.values[:-1] if year == 2020 else ds_year.values
        change_dict[year] = y.flatten()
        # ax.plot(x, y, label=year)

    df = pd.DataFrame.from_dict(change_dict).fillna(0)
    df.set_index("time")
    df = get_nday_mean(df, nday=3)
    df["2019-2021"] = (df[2019] + df[2020] + df[2021]) / 3
    df[[2019, 2022]].plot.line(ax=ax)
    ax.legend()
    ax.set_title(title)
    ax.xaxis.set_major_formatter(month_day_fmt)
    return df


def plot_obs_pop_line():

    org_ds = prep_s5p_ds()
    adm_col = "ADM2_EN"
    bound_pop, crs = get_bound_pop()
    list_city = bound_pop[adm_col].values

    for city in list_city:
        geometry = bound_pop.loc[bound_pop[adm_col] == city].geometry
        ds_city = org_ds.rio.clip(geometry, crs)
        df = plot_trend_line(ds_city, city)


def plot_obs_year():
    org_ds = prep_s5p_ds()
    bound_lv1 = gpd.read_file(UK_SHP_ADM1)

    years = [2019, 2020, 2021, 2022]

    figure, ax = plt.subplots(2, 2, figsize=(16, 6.5 * 2))
    for i, y in enumerate(years):
        s5p_no2_year = org_ds.sel(time=org_ds.time.dt.year.isin([y]))
        s5p_mean = s5p_no2_year.mean("time")[S5P_OBS_COL]
        s5p_mean = s5p_mean
        s5p_mean.plot(
            cmap="YlOrRd",
            vmin=70,
            vmax=100,
            cbar_kwargs={"label": NO2_UNIT},
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
    ds.test_2019.groupby("time").mean().mul(1e6)[[S5P_OBS_COL, S5P_PRED_COL]].plot.line(
        ax=axis[0]
    )

    dsartist = dsshow(
        ds.test_2019[[S5P_OBS_COL, S5P_PRED_COL]].mul(1e6),
        dsh.Point(S5P_OBS_COL, S5P_PRED_COL),
        dsh.count(),
        norm="linear",
        aspect="auto",
        ax=axis[1],
    )

    plt.colorbar(dsartist)

    axis[0].set_title(r"a) Time series trend of OBS_NO$_{2}$ and BAU_NO$_{2}$")
    axis[0].set_xlabel("Date")
    axis[0].set_ylabel(f"$10^{{{-6}}}$ $mol/m^2$")

    axis[1].set_title(f"b) Scatter Plot of OBS_NO$_{2}$ and BAU_NO$_{2}$")
    axis[1].set_xlabel(r"OBS_NO$_{2}$ ($10^{{{-6}}}$ $mol/m^2$)")
    axis[1].set_ylabel(r"BAU_NO$_{2}$ ($10^{{{-6}}}$ $mol/m^2$)")
    axis[1].annotate(
        "$R^2$ = {:.3f}".format(
            r2_score(ds.test_2019[S5P_OBS_COL], ds.test_2019[S5P_PRED_COL])
        ),
        (10, 750),
    )
    line = mlines.Line2D([0, 1], [0, 1], color="red", label="1:1 line")
    transform = axis[1].transAxes
    line.set_transform(transform)
    axis[1].add_line(line)
    axis[1].legend()


#%%
# Plot fire location and conflict point


def plot_fire_conflict():

    # data_df = prep_fire_df() if data_type == "Fire Spot" else prep_conflict_df()
    # color = "orange" if data_type == "Fire Spot" else "red"
    color_fire = "red"
    color_conflict = "orange"
    label_fire = "Fire Spot"
    label_conflict = "Conflict Spot"
    label_coal = "Coal power plant"

    years = [2020, 2021, 2022]

    coal_gdf = gpd.read_file(UK_COAL_SHP)

    bound_lv1 = gpd.read_file(UK_SHP_ADM1)
    bound_lv0 = gpd.read_file(UK_SHP_ADM0)

    fire_df = gpd.clip(prep_fire_df(), bound_lv0.geometry)
    conflict_df = gpd.clip(prep_conflict_df(), bound_lv0.geometry)

    # Plot fire locations
    figure_fire, ax_fire = plt.subplots(6, 3, figsize=(24, 5 * 6), layout="constrained")
    for j, y in enumerate(years):

        sd_ed = PERIOD_DICT[y]
        tks = list(sd_ed.keys())
        for i, tk in enumerate(tks):

            t = sd_ed[tk]

            sd = np.datetime64(f"{y}-{t['sm']}-{t['sd']}T00:00:00.000000000")
            ed = np.datetime64(f"{y}-{t['em']}-{t['ed']}T00:00:00.000000000")

            mask = (fire_df["DATETIME"] > sd) & (fire_df["DATETIME"] <= ed)
            df = fire_df.loc[mask]

            coal_gdf.plot(
                ax=ax_fire[i][j], color="green", markersize=20, label=label_coal
            )
            df.plot(ax=ax_fire[i][j], color=color_fire, markersize=2, label=label_fire)
            bound_lv1.plot(
                ax=ax_fire[i][j], facecolor="None", edgecolor="black", lw=0.2
            )
            ax_fire[i][j].legend(
                bbox_to_anchor=(0, 0),
                loc="lower left",
            )
            ax_fire[i][j].set_title(f"{y}[{tk}]", fontsize=25)
    figure_fire.suptitle(
        f"Satellite-captured fire spots from the end of Febuary to July in 2020, 2021, 2022",
        fontsize=28,
    )
    # Plot conflict locations
    figure_conflict, ax_conflict = plt.subplots(
        4, 3, figsize=(24, 5 * 4), layout="constrained"
    )
    sd_ed = PERIOD_DICT[2022]

    tks = list(sd_ed.keys())

    j = 0
    for i, tk in enumerate(tks):

        i = int(i / 3)

        i = i if i == 0 else i + 1
        j = j if j < 3 else j - 3
        t = sd_ed[tk]
        sd = np.datetime64(f"2022-{t['sm']}-{t['sd']}T00:00:00.000000000")
        ed = np.datetime64(f"2022-{t['em']}-{t['ed']}T00:00:00.000000000")

        mask = (fire_df["DATETIME"] > sd) & (fire_df["DATETIME"] <= ed)
        masked_fire_df = fire_df.loc[mask]

        mask = (conflict_df["DATETIME"] > sd) & (conflict_df["DATETIME"] <= ed)
        masked_conflict_df = conflict_df.loc[mask]

        coal_gdf.plot(
            ax=ax_conflict[i][j], color="green", markersize=20, label=label_coal
        )
        masked_fire_df.plot(
            ax=ax_conflict[i][j],
            color=color_fire,
            markersize=2,
            label=label_fire,
        )
        bound_lv1.plot(
            ax=ax_conflict[i][j], facecolor="None", edgecolor="black", lw=0.2
        )
        ax_conflict[i][j].legend(
            bbox_to_anchor=(0, 0),
            loc="lower left",
        )
        ax_conflict[i][j].set_title(f"2022[{tk}]", fontsize=25)

        coal_gdf.plot(
            ax=ax_conflict[i + 1][j], color="green", markersize=20, label=label_coal
        )
        masked_conflict_df.plot(
            ax=ax_conflict[i + 1][j],
            color=color_conflict,
            markersize=2,
            label=label_conflict,
        )
        bound_lv1.plot(
            ax=ax_conflict[i + 1][j], facecolor="None", edgecolor="black", lw=0.2
        )

        ax_conflict[i + 1][j].legend(bbox_to_anchor=(0, 0), loc="lower left")
        ax_conflict[i + 1][j].set_title(f"2022[{tk}]", fontsize=25)

        j = j + 1

    figure_conflict.suptitle(
        f"Satellite-captured fire spots and Statistic conflict locations from Febuary to July in 2022",
        fontsize=28,
    )


# %%
def plot_ax_line(
    ds,
    geometry,
    location,
    gdf,
    ax,
    year,
    set_ylabel=False,
    get_table=False,
    event="covid",
):

    vl_covid_clr = "#33a02c"
    vl_war_clr = "#1f78b4"
    label_war = "War start date"
    label_covid = "Lockdown period"

    ls_covid = "dashed"
    ls_war = "dashed"
    lw = 2.5

    pred_truth_diff = ["cyan", "red", "#feb24c"]

    sd = np.datetime64(f"{year}-02-01T00:00:00.000000000")
    ed = np.datetime64(f"{year}-07-31T00:00:00.000000000")
    ds_clip = ds.rio.clip(geometry, gdf.crs).sel(time=slice(sd, ed))[
        [S5P_PRED_COL, S5P_OBS_COL]
    ]

    ds_clip_plot = ds_clip.mean(dim=["lat", "lon"], skipna=True)

    # calculate covid stats
    sd_cv19 = np.datetime64(f"{year}-04-18T00:00:00.000000000")
    ed_cv19 = np.datetime64(f"{year}-05-08T00:00:00.000000000")
    if event == "war":
        sd_cv19 = np.datetime64(f"{year}-02-25T00:00:00.000000000")
        ed_cv19 = np.datetime64(f"{year}-03-23T00:00:00.000000000")
        if year == 2020:
            ed_cv19 = np.datetime64(f"{year}-03-22T00:00:00.000000000")
    ds_clip_covid = ds_clip.sel(time=slice(sd_cv19, ed_cv19)).mean(dim=["lat", "lon"])
    obs_bau = (
        (ds_clip_covid[S5P_OBS_COL] - ds_clip_covid[S5P_PRED_COL])
        * 100
        / ds_clip_covid[S5P_PRED_COL]
    )
    t = obs_bau.values.shape
    obs_bau_std = np.nanstd(np.average(obs_bau.values.reshape(3, -1), axis=0))
    obs_bau_mean = obs_bau.mean(dim=["time"], skipna=True).item()

    # ploting
    org_df = ds_clip_plot.to_dataframe()
    df = get_nday_mean(org_df, nday=3)

    df[OBS_PRED_CHNAGE] = df[S5P_OBS_COL] - df[S5P_PRED_COL]
    df[[S5P_PRED_COL, S5P_OBS_COL]].plot.line(
        ax=ax, color=pred_truth_diff, legend=False
    )
    # ax.set_ylim([-40, 200])

    if set_ylabel:
        ax.set_ylabel(NO2_UNIT)
    ax.grid(color="#d9d9d9")
    ax.set_title(f"{location}-{year}", fontsize=18)
    handles, labels = ax.get_legend_handles_labels()

    # ax.axhline(
    #     y=0,
    #     color="black",
    #     linewidth=1,
    #     linestyle="--",
    # )

    # if year == 2020:
    #     ax.axvline(
    #         x=np.datetime64(f"{year}-03-25T00:00:00.000000000"),
    #         color=vl_covid_clr,
    #         linewidth=lw,
    #         linestyle=ls_covid,
    #         label=label_covid,
    #     )
    #     ax.axvline(
    #         x=np.datetime64(f"{year}-05-11T00:00:00.000000000"),
    #         color=vl_covid_clr,
    #         linewidth=lw,
    #         linestyle=ls_covid,
    #     )
    # elif year == 2021:
    if year != 2019:
        covid_line = ax.axvline(
            x=np.datetime64(f"{year}-03-25T00:00:00.000000000"),
            color=vl_covid_clr,
            linewidth=lw,
            linestyle=ls_covid,
        )
        ax.axvline(
            x=np.datetime64(f"{year}-05-11T00:00:00.000000000"),
            color=vl_covid_clr,
            linewidth=lw,
            linestyle=ls_covid,
        )
        war_line = ax.axvline(
            x=np.datetime64(f"{year}-02-24T00:00:00.000000000"),
            color=vl_war_clr,
            linewidth=lw,
            linestyle=ls_war,
        )

        handles = handles + [war_line]
        labels = labels + [label_war]

        handles = handles + [covid_line]
        labels = labels + [label_covid]
        return (
            obs_bau_mean,
            obs_bau_std,
            handles,
            labels if get_table else handles,
            labels,
        )

    return obs_bau_mean, obs_bau_std if get_table else 1

    # elif year == 2022:
    #     ax.axvline(
    #         x=np.datetime64(f"{year}-02-24T00:00:00.000000000"),
    #         color=vl_war_clr,
    #         linewidth=lw,
    #         linestyle=ls_war,
    #         label=label_war,
    #     )


def plot_ppl_obs_bau_line_mlt(org_ds):

    ds_2019 = prep_ds(org_ds, 2019)
    ds_2020 = prep_ds(org_ds, 2020)
    ds_2021 = prep_ds(org_ds, 2021)
    ds_2022 = prep_ds(org_ds, 2022)

    coal_gdf = gpd.read_file(UK_COAL_SHP)
    coal_gdf.crs = "EPSG:4326"
    coal_gdf["buffer"] = coal_gdf.geometry.buffer(0.2, cap_style=3)

    for i, ppl_name in enumerate(coal_gdf.name.values):

        geometry = coal_gdf.loc[coal_gdf["name"] == ppl_name]["buffer"].geometry
        # geometry = coal_gdf.loc[coal_gdf["name"] == ppl_name].geometry
        fig, ax = plt.subplots(1, 4, figsize=(20, 4))

        plot_ax_line(
            ds_2019, geometry, ppl_name, coal_gdf, ax[0], 2019, set_ylabel=True
        )
        plot_ax_line(
            ds_2020,
            geometry,
            ppl_name,
            coal_gdf,
            ax[1],
            2020,
        )
        plot_ax_line(ds_2021, geometry, ppl_name, coal_gdf, ax[2], 2021)
        plot_ax_line(ds_2022, geometry, ppl_name, coal_gdf, ax[3], 2022)

        # fig.legend(
        #     handles,
        #     labels,
        #     ncol=5,
        #     loc="upper center",
        #     bbox_to_anchor=(0.5, -0.01),
        #     fontsize=18,
        # )


def plot_obs_bau_pop_line_mlt(org_ds, event="covid"):

    ds_2019 = prep_ds(org_ds, 2019)
    ds_2020 = prep_ds(org_ds, 2020)
    ds_2021 = prep_ds(org_ds, 2021)
    ds_2022 = prep_ds(org_ds, 2022)

    col = "ADM3_EN"
    bound_pop, crs = get_bound_pop()
    # col = "ADM2_EN"
    list_city = LIST_POP_CITY
    # list_city = [
    #     "Kyiv",
    #     "Kharkivska",
    #     "Donetska",
    #     "Lvivska",
    #     "Dniprovska",
    #     "Odeska",
    #     "Zaporizka",
    #     "Kryvorizka",
    # ]
    nrows, ncols = len(list_city), 4
    fig, ax = plt.subplots(
        nrows, ncols, figsize=(12 * ncols, 6 * nrows), layout="constrained"
    )
    year_col = [i for i in range(2020, 2023)]
    name_col = ["mean", "var"]
    table_dict = {}

    for y in year_col:
        for n in name_col:
            table_dict[f"{y}_{n}"] = []

    table_dict["city"] = []
    for i, city in enumerate(list_city):

        geometry = bound_pop.loc[bound_pop[col] == city].geometry
        # fig, ax = plt.subplots(1, 3, figsize=(16, 4))
        plot_ax_line(
            ds_2019,
            geometry,
            city,
            bound_pop,
            ax[i][0],
            2019,
            set_ylabel=True,
            event=event,
        )
        obs_bau_2020 = plot_ax_line(
            ds_2020,
            geometry,
            city,
            bound_pop,
            ax[i][1],
            2020,
            get_table=True,
            event=event,
        )
        # with handles and labels
        obs_bau_2021 = plot_ax_line(
            ds_2021,
            geometry,
            city,
            bound_pop,
            ax[i][2],
            2021,
            get_table=True,
            event=event,
        )
        obs_bau_2022 = plot_ax_line(
            ds_2022,
            geometry,
            city,
            bound_pop,
            ax[i][3],
            2022,
            get_table=False,
            event=event,
        )

        table_dict["2022_mean"].append(obs_bau_2022[0])
        table_dict["2022_var"].append(obs_bau_2022[1])

        table_dict["2021_mean"].append(obs_bau_2021[0])
        table_dict["2021_var"].append(obs_bau_2021[1])

        table_dict["2020_mean"].append(obs_bau_2020[0])
        table_dict["2020_var"].append(obs_bau_2020[1])

        table_dict["city"].append(city)

    fig.legend(
        obs_bau_2021[2],
        obs_bau_2021[3],
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.01),
        fontsize=40,
    )
    return pd.DataFrame.from_dict(table_dict).round(1)


def plot_obs_bau_pop_line_sgl(org_ds, year):
    ds = prep_ds(org_ds, year)
    bound_lv2, crs = get_bound_pop()
    adm_col = "ADM2_EN"
    # bound_lv2 = gpd.read_file(UK_SHP_ADM2)
    city_no2 = {}

    sd = np.datetime64(f"{year}-02-01T00:00:00.000000000")
    ed = np.datetime64(f"{year}-07-31T00:00:00.000000000")

    for i, city in enumerate(bound_lv2[adm_col].values):
        fig = plt.figure(1 + i, figsize=(6, 4))
        ax = plt.subplot(1, 1, 1)
        geometry = bound_lv2.loc[bound_lv2[adm_col] == city].geometry
        ds_clip = (
            ds.rio.clip(geometry, crs)
            .sel(time=slice(sd, ed))
            .mean(dim=["lat", "lon"])[[S5P_PRED_COL, S5P_OBS_COL]]
        )
        city_no2[city] = ds_clip
        df = ds_clip.to_dataframe()
        df = get_nday_mean(df, nday=3)
        df[OBS_PRED_CHNAGE] = df[S5P_OBS_COL] - df[S5P_PRED_COL]
        df[[S5P_PRED_COL, S5P_OBS_COL, OBS_PRED_CHNAGE]].plot.line(ax=ax)
        if year in [2020, 2021]:
            ax.axvline(
                x=np.datetime64(f"{year}-03-25T00:00:00.000000000"),
                color="r",
                linewidth=1,
                linestyle="--",
            )
            ax.axvline(
                x=np.datetime64(f"{year}-05-11T00:00:00.000000000"),
                color="r",
                linewidth=1,
                linestyle="--",
            )
        if year == 2022:
            ax.axvline(
                x=np.datetime64(f"{year}-02-24T00:00:00.000000000"),
                color="r",
                linewidth=1,
                linestyle="--",
            )
        ax.axhline(
            y=0,
            color="black",
            linewidth=1,
            linestyle="--",
        )
        ax.grid()
        ax.set_title(f"{city}")


def plot_ppl_obs_bau_line_sgl(org_ds, year):

    ds = prep_ds(org_ds, year)
    coal_gdf = gpd.read_file(UK_COAL_SHP)
    # coal_gdf["buffer"] = coal_gdf.geometry.buffer(0.15, cap_style=3).to_crs(
    #     coal_gdf.crs
    # )

    sd = np.datetime64(f"{year}-02-01T00:00:00.000000000")
    ed = np.datetime64(f"{year}-07-31T00:00:00.000000000")

    ppl_no2 = {}

    for i, ppl_name in enumerate(coal_gdf.name.values):

        geometry = coal_gdf.loc[coal_gdf["name"] == ppl_name].geometry

        fig = plt.figure(1 + i, figsize=(6, 4))
        ax = plt.subplot(1, 1, 1)

        ds_clip = (
            ds.rio.clip(geometry, coal_gdf.crs)
            .sel(time=slice(sd, ed))
            .mean(dim=["lat", "lon"])[[S5P_PRED_COL, S5P_OBS_COL]]
        )
        ppl_no2[ppl_name] = ds_clip
        df = ds_clip.to_dataframe()
        df = get_nday_mean(org_df, nday=3)
        df[OBS_PRED_CHNAGE] = df[S5P_OBS_COL] - df[S5P_PRED_COL]
        df[[S5P_PRED_COL, S5P_OBS_COL, OBS_PRED_CHNAGE]].plot.line(ax=ax)
        ax.axhline(
            y=0,
            color="black",
            linewidth=1,
            linestyle="--",
        )
        if year == 2022:
            ax.axvline(
                x=np.datetime64(f"{year}-02-24T00:00:00.000000000"),
                color="r",
                linewidth=1,
                linestyle="--",
            )
        ax.grid()
        ax.set_title(f"{ppl_name}-{year}")


def plot_ppl_obs_line():

    org_ds = prep_s5p_ds()
    years = [2019, 2020, 2021, 2022]
    coal_gdf = gpd.read_file(UK_COAL_SHP)
    # coal_gdf["buffer"] = coal_gdf.geometry.buffer(0.15, cap_style=3).to_crs(
    #     coal_gdf.crs
    # )
    for i, ppl_name in enumerate(coal_gdf.name.values):
        geometry = coal_gdf.loc[coal_gdf["name"] == ppl_name].geometry
        ds_clip = org_ds.rio.clip(geometry, coal_gdf.crs).mean(dim=["lat", "lon"])[
            S5P_OBS_COL
        ]
        ds_city = org_ds.rio.clip(geometry, coal_gdf.crs)
        df = plot_trend_line(ds_city, ppl_name)


def plot_wind_speed_direction(ds, year):
    pass


def plot_weather_params(ds, event="covid"):

    # border_df = get_boundary_cities()
    # conflict_df = get_monthly_conflict()

    u10 = ds.era5["u10"]
    v10 = ds.era5["v10"]
    ds.era5["wind"] = np.sqrt(u10**2 + v10**2)

    year_target = 2022
    list_color = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]

    if event == "covid":
        year_target = 2020
        list_color = ["#1b9e77", "#d95f02"]

    years = [i for i in range(2019, year_target + 1)]
    sd_ed = PERIOD_DICT[year_target]

    tks = list(sd_ed.keys())

    var_label_dict = {
        # "wind": "Wind speed (m/s)",
        "blh": "Boundary layer height (m)",
        # "z": "Geopotential (m\u00b2/s\u00b22)",
    }
    # x_range_dict = {"wind": [2, 4.5], "blh": [400, 900], "t2m": [275, 285]}
    list_var = list(var_label_dict.keys())

    nrows = 1 if len(tks) == 2 else 2
    ncols = int(len(tks) / nrows)
    figure, ax = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4 * nrows), layout="constrained"
    )
    ylabel = "Relative Frequency (%)"

    k = 0
    j = 0
    for i, tk in enumerate(tks):
        i = int(i / ncols)
        j = j if j < ncols else 0
        t = sd_ed[tk]
        ax_sub = ax[i][j] if nrows > 1 else ax[j]
        for var in list_var:
            for year, color in zip(years, list_color):
                sd = np.datetime64(f"{year}-{t['sm']}-{t['sd']}{HOUR_STR}")
                ed = np.datetime64(f"{year}-{t['em']}-{t['ed']}{HOUR_STR}")
                # sel_ds = ds.era5.sel(time=slice(sd, ed)).mean("time")
                sel_ds = ds.era5.sel(time=slice(sd, ed))
                ts_data = sel_ds[var].values.reshape(-1)
                ts_data = ts_data[~np.isnan(ts_data)]
                # ts_data = clip_and_flat_event_city(sel_ds, var, tk, conflict_df, event)

                ws = np.ones_like(ts_data) * 100 / ts_data.size
                ax_sub.hist(
                    ts_data,
                    bins=15,
                    weights=ws,
                    ec=color,
                    fc="None",
                    histtype="step",
                    label=f"{year}",
                    linewidth=3 if year == 2022 or year == 2019 else 1,
                )
                ax_sub.legend()
                ax_sub.set_title(f"{INDEX_FIG[k]}) {tk}", fontsize=20)
                ax_sub.set_xlabel(var_label_dict[var])
                # ax_sub.set_xlim(x_range_dict[var])
                ax_sub.set_ylabel(ylabel)
                ax_sub.legend(loc="upper right")
            k += 1
            j += 1


def plot_obs_bau_bubble(org_ds, year):

    ds = prep_ds(org_ds, year)

    bound_lv1 = gpd.read_file(UK_SHP_ADM1)
    adm_col = "ADM3_EN"
    bound_pop, crs = get_bound_pop()

    list_city = LIST_POP_CITY
    sd_ed = PERIOD_DICT[year]

    dict_no2_change = {}

    tks = list(sd_ed.keys())

    for tk in tks:

        dict_no2_change[tk] = []

        t = sd_ed[tk]
        sd = np.datetime64(f"{year}-{t['sm']}-{t['sd']}T00:00:00.000000000")
        ed = np.datetime64(f"{year}-{t['em']}-{t['ed']}T00:00:00.000000000")

        for city in list_city:

            geometry = bound_pop.loc[bound_pop[adm_col] == city].geometry
            city_ds = (
                ds.rio.clip(geometry, crs)
                .mean(dim=["lat", "lon"])
                .sel(time=slice(sd, ed))
                .mean("time")[[S5P_PRED_COL, S5P_OBS_COL]]
            )
            dict_no2_change[tk].append(
                (city_ds[S5P_OBS_COL].item() - city_ds[S5P_PRED_COL].item())
                * 100
                / city_ds[S5P_PRED_COL].item()
            )

    df_no2_change = pd.DataFrame.from_dict(dict_no2_change)
    df_no2_change["Population"] = bound_pop["Population"].values
    geo_df = gpd.GeoDataFrame(
        df_no2_change, crs=crs, geometry=bound_pop.geometry.centroid
    )

    nrows = int(len(tks) / 2)
    ncols = 2
    figure, ax = plt.subplots(
        nrows, ncols, figsize=(6 * ncols, 5 * nrows), layout="constrained"
    )

    j = 0
    for i, col in enumerate(tks):

        i = int(i / 2)
        j = 0 if j > 1 else j
        sub_ax = ax[i][j] if nrows > 1 else ax[j]
        bound_lv1 = gpd.read_file(UK_SHP_ADM1)
        bound_lv1.plot(ax=sub_ax, facecolor="white", edgecolor="black", lw=0.7)
        norm_val = 15
        g = sns.scatterplot(
            data=geo_df,
            x=geo_df.centroid.x,
            y=geo_df.centroid.y,
            hue=col,
            hue_norm=(-15, 15),
            size="Population",
            sizes=(50, 500),
            palette=CMAP_NO2,
            ax=sub_ax,
            edgecolor="black",
            linewidth=1,
        )

        # g.legend(
        #     bbox_to_anchor=(1.0, 1.0),
        #     ncol=1,
        #     bbox_transform=sub_ax.transAxes,
        # )

        norm = plt.Normalize(-15, 15)
        sm = plt.cm.ScalarMappable(cmap=CMAP_NO2, norm=norm)
        sm.set_array([])

        # clb = g.figure.colorbar(
        #     sm,
        #     ax=ax[i][j - 1],
        #     fraction=0.047,
        #     orientation="horizontal",
        #     extend="both",
        #     label="NO$_{2}$ col. change (%)",
        # )

        g.set(title=rf"{INDEX_FIG[j]}) {year}_OBS[{col}] - {year}_BAU[{col}]")

        h, l = g.get_legend_handles_labels()
        l = [f"{li}M" if li != "Population" else li for li in l]
        legend = sub_ax.legend(
            h[-6:],
            l[-6:],
            bbox_to_anchor=(0, 0),
            loc="lower left",
            borderaxespad=0.0,
            # fontsize=13,
            edgecolor="black",
        )
        legend.get_frame().set_alpha(None)
        # legend.get_frame().set_facecolor((0, 0, 1, 0.1))
        j += 1
    figure.colorbar(
        sm,
        ax=ax[:, :] if nrows > 1 else ax[:],
        # ax=ax[:,1],
        # fraction=0.47,
        orientation="horizontal",
        extend="both",
        label="NO$_{2}$ col. change (%)",
        location="bottom",
        shrink=0.4,
    )
    plt.suptitle(rf"OBS_NO$_{2}$ - BAU_NO$_{2}$ difference (Major cities)", fontsize=18)
    return geo_df


def plot_obs_bubble(event="war"):

    bound_lv1 = gpd.read_file(UK_SHP_ADM1)

    org_ds = prep_s5p_ds()
    bound_pop, crs = get_bound_pop()
    adm_col = "ADM3_EN"
    list_city = bound_pop[adm_col].values

    year_target = 2022 if event == "war" else 2020
    sd_ed = PERIOD_DICT[year_target]
    tks = list(sd_ed.keys())
    year_srcs = [i for i in range(2019, year_target)]

    obs_dict_year = {}

    nrows = int((len(year_srcs) + 1) / 2)
    ncols = 2

    self_figure, self_ax = plt.subplots(
        nrows, 2, figsize=(6 * ncols, 5 * nrows), layout="constrained"
    )

    j = 0
    mean_std_dict = {}
    for i, year in enumerate(year_srcs + [year_target]):
        i = int(i / 2)
        j = 0 if j > 1 else j
        for tk in tks[:2]:
            obs_dict_year[f"{year}[{tk}]"] = []
            t = sd_ed[tk]
            sd = np.datetime64(f"{year}-{t['sm']}-{t['sd']}{HOUR_STR}")
            ed = np.datetime64(f"{year}-{t['em']}-{t['ed']}{HOUR_STR}")

            for city in list_city:
                geometry = bound_pop.loc[bound_pop[adm_col] == city].geometry
                adm_ds = (
                    org_ds.rio.clip(geometry, crs)
                    .mean(dim=["lat", "lon"])
                    .sel(time=slice(sd, ed))
                    .mean("time")[[S5P_OBS_COL]]
                )
                obs_dict_year[f"{year}[{tk}]"].append(adm_ds[S5P_OBS_COL].item())
            bound_pop[f"{year}[{tk}]"] = obs_dict_year[f"{year}[{tk}]"]

        col = f"{year}[{tks[1]}] - {year}[{tks[0]}]"
        bound_pop[col] = (
            (bound_pop[f"{year}[{tks[1]}]"] - bound_pop[f"{year}[{tks[0]}]"])
            * 100
            / bound_pop[f"{year}[{tks[0]}]"]
        )
        mean_std_dict[f"mean_{col}"] = np.mean(bound_pop[col].values)
        mean_std_dict[f"std_{col}"] = np.nanstd(bound_pop[col].values)
        # Plot
        ax = self_ax[i][j] if nrows > 1 else self_ax[j]
        bound_lv1.plot(ax=ax, facecolor="white", edgecolor="black", lw=0.7)
        norm_val = 15
        g = sns.scatterplot(
            data=bound_pop,
            x=bound_pop.centroid.x,
            y=bound_pop.centroid.y,
            hue=col,
            hue_norm=(-1 * norm_val, norm_val),
            size="Population",
            sizes=(50, 500),
            palette=CMAP_NO2,
            ax=ax,
            edgecolor="black",
            linewidth=1,
        )
        norm = plt.Normalize(-1 * norm_val, norm_val)
        sm = plt.cm.ScalarMappable(cmap=CMAP_NO2, norm=norm)
        sm.set_array([])

        g.set(title=rf"{INDEX_FIG[j]}) {col}")

        h, l = g.get_legend_handles_labels()
        l = [f"{li}M" if li != "Population" else li for li in l]
        legend = ax.legend(
            h[-6:],
            l[-6:],
            bbox_to_anchor=(0, 0),
            loc="lower left",
            borderaxespad=0.0,
            # fontsize=13,
            edgecolor="black",
        )
        legend.get_frame().set_alpha(None)
        j += 1

    self_figure.colorbar(
        sm,
        ax=self_ax[:, :] if nrows > 1 else self_ax[:],
        # ax=ax[:,1],
        # fraction=0.47,
        orientation="horizontal",
        extend="both",
        label="NO$_{2}$ col. change (%)",
        location="bottom",
        shrink=0.4,
    )
    plt.suptitle(rf'OBS "before-during" change estimates (Major cities)', fontsize=18)

    # inter bf af change
    nrows = len(year_srcs)
    ncols = 2
    inter_figure, inter_ax = plt.subplots(
        nrows, ncols, figsize=(6 * ncols, 5 * nrows), layout="constrained"
    )
    j = 0

    for i, year_src in enumerate(year_srcs):
        for j, tk in enumerate(tks[:2]):

            col = f"{year_target}[{tk}] - {year_src}[{tk}]"
            bound_pop[col] = (
                (bound_pop[f"{year_target}[{tk}]"] - bound_pop[f"{year_src}[{tk}]"])
                * 100
                / bound_pop[f"{year_src}[{tk}]"]
            )
            mean_std_dict[f"mean_{col}"] = np.mean(bound_pop[col].values)
            mean_std_dict[f"std_{col}"] = np.nanstd(bound_pop[col].values)
            ax = inter_ax[i][j] if nrows > 1 else inter_ax[j]
            bound_lv1.plot(ax=ax, facecolor="white", edgecolor="black", lw=0.7)
            g = sns.scatterplot(
                data=bound_pop,
                x=bound_pop.centroid.x,
                y=bound_pop.centroid.y,
                hue=col,
                hue_norm=(-1 * norm_val, norm_val),
                size="Population",
                sizes=(50, 500),
                palette=CMAP_NO2,
                ax=ax,
                edgecolor="black",
                linewidth=1,
            )
            norm = plt.Normalize(-1 * norm_val, norm_val)
            sm = plt.cm.ScalarMappable(cmap=CMAP_NO2, norm=norm)
            sm.set_array([])

            g.set(title=rf"{INDEX_FIG[j]}) {col}")

            h, l = g.get_legend_handles_labels()
            l = [f"{li}M" if li != "Population" else li for li in l]
            legend = ax.legend(
                h[-6:],
                l[-6:],
                bbox_to_anchor=(0, 0),
                loc="lower left",
                borderaxespad=0.0,
                # fontsize=13,
                edgecolor="black",
            )
            legend.get_frame().set_alpha(None)
    inter_figure.colorbar(
        sm,
        ax=inter_ax[:, :] if nrows > 1 else inter_ax[:],
        # ax=ax[:,1],
        # fraction=0.47,
        orientation="horizontal",
        extend="both",
        label="NO$_{2}$ col. change (%)",
        location="bottom",
        shrink=0.4,
    )
    plt.suptitle(rf'OBS "year-to-year" change estimates (Major cities)', fontsize=18)

    return bound_pop, mean_std_dict


def plot_obs_change_adm2():

    adm_col = "ADM2_EN"
    bound_lv2 = gpd.read_file(UK_SHP_ADM2)
    coal_gdf = gpd.read_file(UK_COAL_SHP)
    boundary = get_boundary_cities()
    org_ds = prep_s5p_ds()

    conflict_ds = get_monthly_conflict()

    year_target = 2022
    sd_ed = PERIOD_DICT[year_target]
    year_srcs = [i for i in range(2019, year_target)]

    tks = list(sd_ed.keys())

    pixel_change_dict = {}
    for year in year_srcs + [year_target]:

        for tk in tks:
            t = sd_ed[tk]
            sd = np.datetime64(f"{year}-{t['sm']}-{t['sd']}{HOUR_STR}")
            ed = np.datetime64(f"{year}-{t['em']}-{t['ed']}{HOUR_STR}")
            pixel_change_dict[f"{year}_{tk}"] = org_ds.sel(time=slice(sd, ed)).mean(
                "time"
            )[[S5P_OBS_COL]]

        pixel_change_dict[f"self_{year}"] = (
            (
                pixel_change_dict[f"{year}_{tks[1]}"]
                - pixel_change_dict[f"{year}_{tks[0]}"]
            )
            * 100
            / pixel_change_dict[f"{year}_{tks[0]}"]
        )
        self_year_change = []

        for adm2 in bound_lv2[adm_col].values:
            geometry = bound_lv2.loc[bound_lv2[adm_col] == adm2].geometry
            self_year_change.append(
                pixel_change_dict[f"self_{year}"]
                .rio.clip(geometry, bound_lv2.crs)
                .mean(dim=["lat", "lon"])[S5P_OBS_COL]
                .item()
            )
        bound_lv2[f"self_{year}"] = self_year_change

    for tk in tks:
        for year in year_srcs:

            pixel_change_year = (
                (
                    pixel_change_dict[f"{year_target}_{tk}"]
                    - pixel_change_dict[f"{year}_{tk}"]
                )
                * 100
                / pixel_change_dict[f"{year}_{tk}"]
            )
            year_tk_items = []
            for adm2 in bound_lv2[adm_col].values:
                geometry = bound_lv2.loc[bound_lv2[adm_col] == adm2].geometry

                # cal obs deweather no2 ds
                change_adm_no2_ds = (
                    pixel_change_year.rio.clip(geometry, bound_lv2.crs)
                    .mean(dim=["lat", "lon"])[S5P_OBS_COL]
                    .item()
                )
                year_tk_items.append(change_adm_no2_ds)
            bound_lv2[f"inter_{year}_{tk}"] = year_tk_items

    # self_ref_fig, self_ref_ax = plt.subplots(
    #     2, 2, figsize=(6 * 2, 5.2 * 2), layout="constrained"
    # )
    # inter_ref_fig, inter_ref_ax = plt.subplots(
    #     2, 3, figsize=(6 * 3, 5.2 * 2), layout="constrained"
    # )
    ncols = 3
    nrows = len(tks)
    inter_war_fig, inter_war_ax = plt.subplots(
        nrows, ncols, figsize=(6 * ncols, 5 * nrows), layout="constrained"
    )

    # self refugee change
    # j = 0
    # k = 0
    # for i, year in enumerate(year_srcs + [year_target]):
    #     i = int(i / 2)
    #     j = 0 if j > 1 else j
    #     col = f"self_{year}"
    #     bound_lv2.plot(
    #         column=col,
    #         ax=self_ref_ax[i][j],
    #         cmap=CMAP_NO2,
    #         vmin=-70,
    #         vmax=70,
    #         legend=False,
    #     )
    #     self_ref_ax[i][j].set_title(
    #         rf"{INDEX_FIG[k]}) {year}[{tks[1]}] - {year}[{tks[0]}]", fontsize=14
    #     )
    #     bound_lv2.plot(
    #         ax=self_ref_ax[i][j], facecolor="None", edgecolor="black", lw=0.2
    #     )
    #     self_ref_conflict = conflict_ds.loc[conflict_ds[f"conflict_{tks[1]}"] > 2]
    #     self_ref_conflict.plot(
    #         ax=self_ref_ax[i][j],
    #         facecolor="None",
    #         edgecolor=EDGE_COLOR_CONFLICT,
    #         lw=1,
    #     )
    #     coal_gdf.plot(
    #         ax=self_ref_ax[i][j],
    #         color=COAL_COLOR,
    #         markersize=20,
    #         label="CPP",
    #     )
    #     boundary.plot(
    #         ax=self_ref_ax[i][j],
    #         facecolor="None",
    #         edgecolor=EDGE_COLOR_BORDER,
    #         lw=1,
    #     )
    #     handles, _ = self_ref_ax[i][j].get_legend_handles_labels()
    #     self_ref_ax[i][j].legend(
    #         handles=[*LG_CONFLICT, *LG_BORDER, *handles], loc="lower left"
    #     )
    #     j += 1
    #     k += 1
    # norm = plt.Normalize(-70, 70)
    # sm = plt.cm.ScalarMappable(cmap=CMAP_NO2, norm=norm)
    # sm.set_array([])
    # self_ref_fig.colorbar(
    #     sm,
    #     ax=self_ref_ax[:, :],
    #     # ax=ax[:,1],
    #     # fraction=0.47,
    #     orientation="horizontal",
    #     extend="both",
    #     label="NO$_{2}$ col. change (%)",
    #     location="bottom",
    #     shrink=0.4,
    # )
    # self_ref_fig.suptitle(
    #     rf'OBS "before-during" change estimates (City level)', fontsize=18
    # )
    # # inter refugee change
    # for i, tk in enumerate(tks[:1]):
    #     for j, year in enumerate(year_srcs):
    #         col = f"inter_{year}_{tk}"
    #         bound_lv2.plot(
    #             column=col,
    #             ax=inter_ref_ax[i][j],
    #             cmap=CMAP_NO2,
    #             vmin=-70,
    #             vmax=70,
    #             legend=False,
    #         )
    #         inter_ref_ax[i][j].set_title(
    #             f"{year_target}[{tk}] - {year}[{tk}]", fontsize=16
    #         )
    #         bound_lv2.plot(
    #             ax=inter_ref_ax[i][j], facecolor="None", edgecolor="black", lw=0.2
    #         )
    #         if i > 0:
    #             inter_ref_conflict = conflict_ds.loc[
    #                 conflict_ds[f"conflict_{tks[1]}"] > 2
    #             ]
    #             inter_ref_conflict.plot(
    #                 ax=inter_ref_ax[i][j],
    #                 facecolor="None",
    #                 edgecolor=EDGE_COLOR_CONFLICT,
    #                 lw=1,
    #             )
    #             coal_gdf.plot(
    #                 ax=inter_ref_ax[i][j],
    #                 color=COAL_COLOR,
    #                 markersize=20,
    #                 label="CPP",
    #             )
    #             boundary.plot(
    #                 ax=inter_ref_ax[i][j],
    #                 facecolor="None",
    #                 edgecolor=EDGE_COLOR_BORDER,
    #                 lw=1,
    #             )
    #             handles, _ = inter_ref_ax[i][j].get_legend_handles_labels()
    #             inter_ref_ax[i][j].legend(
    #                 handles=[*LG_CONFLICT, *LG_BORDER, *handles], loc="lower left"
    #             )

    # norm = plt.Normalize(-70, 70)
    # sm = plt.cm.ScalarMappable(cmap=CMAP_NO2, norm=norm)
    # sm.set_array([])
    # inter_ref_fig.colorbar(
    #     sm,
    #     ax=inter_ref_ax[:, :],
    #     # ax=ax[:,1],
    #     # fraction=0.47,
    #     orientation="horizontal",
    #     extend="both",
    #     label="NO$_{2}$ col. change (%)",
    #     location="bottom",
    #     shrink=0.4,
    # )
    # inter_ref_fig.suptitle(
    #     rf'OBS "year-to-year" change estimates (City level)', fontsize=20
    # )

    # inter war change
    for i, tk in enumerate(tks):
        for j, year in enumerate(year_srcs):
            col = f"inter_{year}_{tk}"
            bound_lv2.plot(
                column=col,
                ax=inter_war_ax[i][j],
                cmap=CMAP_NO2,
                vmin=-30,
                vmax=30,
                legend=False,
            )
            inter_war_ax[i][j].set_title(
                f"{year_target}[{tk}] - {year}[{tk}]", fontsize=18
            )
            bound_lv2.plot(
                ax=inter_war_ax[i][j], facecolor="None", edgecolor="black", lw=0.2
            )
            inter_war_conflict = conflict_ds.loc[conflict_ds[f"conflict_{tk}"] > 10]
            inter_war_conflict.plot(
                ax=inter_war_ax[i][j],
                facecolor="None",
                edgecolor=EDGE_COLOR_CONFLICT,
                lw=1,
            )
            coal_gdf.plot(
                ax=inter_war_ax[i][j],
                color=COAL_COLOR,
                markersize=20,
                label="CPP",
            )
            boundary.plot(
                ax=inter_war_ax[i][j],
                facecolor="None",
                edgecolor=EDGE_COLOR_BORDER,
                lw=1,
            )
            handles, _ = inter_war_ax[i][j].get_legend_handles_labels()
            inter_war_ax[i][j].legend(
                handles=[*LG_CONFLICT, *LG_BORDER, *handles], loc="lower left"
            )

    norm = plt.Normalize(-30, 30)
    sm = plt.cm.ScalarMappable(cmap=CMAP_NO2, norm=norm)
    sm.set_array([])
    inter_war_fig.colorbar(
        sm,
        ax=inter_war_ax[:, :],
        # ax=ax[:,1],
        # fraction=0.47,
        orientation="horizontal",
        extend="both",
        label="NO$_{2}$ col. change (%)",
        location="bottom",
        shrink=0.4,
    )
    inter_war_fig.suptitle(
        rf'OBS "year-to-year" change estimates (City level)', fontsize=22
    )

    return bound_lv2


def plot_obs_bau_adm2(org_ds, year_ref, mode="3_cf_no2_bau"):
    year_war = 2022
    border_df = get_boundary_cities()
    conflict_df = get_monthly_conflict()
    coal_gdf = gpd.read_file(UK_COAL_SHP)
    # mode =["2_cf", "2_no2_bau", "3_cf_no2_bau"]
    ds = prep_ds(org_ds, year_ref)

    conflict_ds = prep_conflict_df()
    fire_ds = prep_fire_df()

    bound_lv2 = gpd.read_file(UK_SHP_ADM2)
    sd_ed = PERIOD_DICT[year_war]

    adm_col = "ADM2_EN"

    dict_no2_change = {}
    dict_conflict_change = {}
    dict_fire_change = {}

    list_adm = bound_lv2[adm_col].values

    tks = list(sd_ed.keys())
    for tk in tks:

        dict_no2_change[tk] = []
        dict_conflict_change[tk] = []
        dict_fire_change[tk] = []

        t = sd_ed[tk]
        sd = np.datetime64(f"{year_ref}-{t['sm']}-{t['sd']}T00:00:00.000000000")
        ed = np.datetime64(f"{year_ref}-{t['em']}-{t['ed']}T00:00:00.000000000")

        for adm in list_adm:
            geometry = bound_lv2.loc[bound_lv2[adm_col] == adm].geometry

            # cal obs deweather no2 ds
            adm_no2_ds = (
                ds.rio.clip(geometry, bound_lv2.crs)
                .mean(dim=["lat", "lon"])
                .sel(time=slice(sd, ed))
                .mean("time")[[S5P_PRED_COL, S5P_OBS_COL]]
            )

            dict_no2_change[tk].append(
                (adm_no2_ds[S5P_OBS_COL].item() - adm_no2_ds[S5P_PRED_COL].item())
                * 100
                / adm_no2_ds[S5P_PRED_COL].item()
            )

            # cal conflict_ds
            mask_date = (conflict_ds["DATETIME"] > sd) & (conflict_ds["DATETIME"] <= ed)
            amd_cflt_ds = conflict_ds.loc[mask_date]
            amd_cflt_ds = gpd.clip(amd_cflt_ds, geometry)
            dict_conflict_change[tk].append(len(amd_cflt_ds))

            # cal fire ds
            mask_date = (fire_ds["DATETIME"] > sd) & (fire_ds["DATETIME"] <= ed)
            amd_fire_ds = fire_ds.loc[mask_date]
            amd_fire_ds = gpd.clip(amd_fire_ds, geometry)
            dict_fire_change[tk].append(len(amd_fire_ds))

        bound_lv2[f"war_{tk}"] = dict_no2_change[tk]
        bound_lv2[f"conflict_{tk}"] = dict_conflict_change[tk]
        bound_lv2[f"fire_{tk}"] = dict_fire_change[tk]

    # nrows = 1 if mode[0] == "2" else len(tks[2:])

    # mode =["2_cf", "2_no2_bau", "3_cf_no2_bau"]
    nrows = len(tks)
    ncols = 3
    figure, ax = plt.subplots(
        nrows, ncols, figsize=(6 * ncols, 5 * nrows), layout="constrained"
    )
    # if mode == "2_cf":
    #     t = tks[1]
    #     bound_lv2.plot(
    #         column=f"conflict_{t}",
    #         ax=ax[0],
    #         legend=True,
    #         cmap=CMAP_CONFLICT,
    #         legend_kwds={
    #             "label": "Number of conflict spots",
    #             "orientation": "horizontal",
    #             "extend": "both",
    #             "fraction": 0.8,
    #         },
    #     )

    #     bound_lv2.plot(
    #         column=f"fire_{t}",
    #         ax=ax[1],
    #         legend=True,
    #         cmap=CMAP_FIRE,
    #         legend_kwds={
    #             "label": "Number of fire spots",
    #             "orientation": "horizontal",
    #             "extend": "both",
    #             "fraction": 0.8,
    #         },
    #     )
    #     ax[0].set_title(f"a) Conflict locations", fontsize=14)
    #     ax[1].set_title(f"b) Fire spots", fontsize=14)
    #     for i in range(len(ax)):
    #         bound_lv2.plot(ax=ax[i], facecolor="None", edgecolor="black", lw=0.2)
    #     plt.suptitle(
    #         rf"Conflict locations and Fire spots in {year_ref}[{t}]", fontsize=18
    #     )
    # elif mode == "2_no2_bau":
    #     for i in [0, 1]:
    #         bound_lv2.plot(
    #             column=f"war_{tks[i]}",
    #             ax=ax[i],
    #             legend=True,
    #             cmap=CMAP_NO2,
    #             vmin=-40,
    #             vmax=40,
    #             legend_kwds={
    #                 "label": r"NO$_{2}$ col. change (%)",
    #                 "orientation": "horizontal",
    #                 "extend": "both",
    #                 "shrink": 0.8,
    #             },
    #         )
    #         ax[i].set_title(
    #             f"{INDEX_FIG[i]}) {year_ref}_OBS[{tks[i]}] - {year_ref}_BAU[{tks[i]}]",
    #             fontsize=14,
    #         )
    #         coal_gdf.plot(
    #             ax=ax[i],
    #             color=COAL_COLOR,
    #             markersize=20,
    #             label="CPP",
    #         )
    #         bound_lv2.plot(ax=ax[i], facecolor="None", edgecolor="black", lw=0.2)
    #     event_bound = conflict_df.loc[conflict_df[f"conflict_{tks[i]}"] > 2]
    #     event_bound.plot(
    #         ax=ax[i], facecolor="None", edgecolor=EDGE_COLOR_CONFLICT, lw=1
    #     )
    #     border_df.plot(ax=ax[i], facecolor="None", edgecolor=EDGE_COLOR_BORDER, lw=1)
    #     handles, _ = ax[i].get_legend_handles_labels()
    #     ax[i].legend(handles=[*LG_CONFLICT, *LG_BORDER, *handles], loc="lower left")
    #     plt.suptitle(
    #         rf"OBS_NO$_{2}$ - BAU_NO$_{2}$ difference (City level)", fontsize=18
    #     )
    # elif mode == "3_cf_no2_bau":
    for i, tk in enumerate(tks):
        legend = False if i < 4 else True
        bound_lv2.plot(
            column=f"war_{tk}",
            ax=ax[i][0],
            legend=legend,
            cmap=CMAP_NO2,
            vmin=-20,
            vmax=20,
            legend_kwds={
                "label": r"NO$_{2}$ col. change (%)",
                "orientation": "horizontal",
                "extend": "both",
                "shrink": 0.8,
            },
        )
        bound_lv2.plot(
            column=f"conflict_{tk}",
            ax=ax[i][1],
            legend=legend,
            cmap=CMAP_CONFLICT,
            vmin=0,
            vmax=300,
            legend_kwds={
                "label": "Number of conflict spots",
                "orientation": "horizontal",
                "extend": "both",
                "shrink": 0.8,
            },
        )

        bound_lv2.plot(
            column=f"fire_{tk}",
            ax=ax[i][2],
            legend=legend,
            cmap=CMAP_FIRE,
            vmin=0,
            vmax=600,
            legend_kwds={
                "label": "Number of fire spots",
                "orientation": "horizontal",
                "extend": "both",
                "shrink": 0.8,
            },
        )

        event_bound = conflict_df.loc[
            conflict_df[f"conflict_{tk}"] > THRESHOLD_CONFLICT_POINT
        ]
        event_bound.plot(
            ax=ax[i][0], facecolor="None", edgecolor=EDGE_COLOR_CONFLICT, lw=1
        )
        border_df.plot(ax=ax[i][0], facecolor="None", edgecolor=EDGE_COLOR_BORDER, lw=1)

        for j in range(len(ax[i])):
            coal_gdf.plot(
                ax=ax[i][j],
                color=COAL_COLOR,
                markersize=20,
                label="CPP",
            )
            bound_lv2.plot(ax=ax[i][j], facecolor="None", edgecolor="black", lw=0.2)
            ax[i][j].legend(loc="lower left")

        handles, _ = ax[i][0].get_legend_handles_labels()
        ax[i][0].legend(handles=[*LG_CONFLICT, *LG_BORDER, *handles], loc="lower left")

        ax[i][0].set_title(rf"{year_ref}_OBS[{tk}] - {year_ref}_BAU[{tk}]", fontsize=14)
        ax[i][1].set_title(f"Conflict Locations {year_ref}[{tk}]", fontsize=14)
        ax[i][2].set_title(f"Fire Locations {year_ref}[{tk}]", fontsize=14)
    plt.suptitle(
        rf"OBS_NO$_{2}$ -  BAU_NO$_{2}$ difference , Conflict locations, and Fire spots 2022[Mar - Jul] (City level)",
        fontsize=18,
    )

    return bound_lv2


def plot_conflict_refugee_adm2():

    conflict_df = get_monthly_conflict()
    coal_gdf = gpd.read_file(UK_COAL_SHP)
    boundary = get_boundary_cities()
    bound_lv2 = gpd.read_file(UK_SHP_ADM2)

    sd_ed = PERIOD_DICT[2022]
    tks = list(sd_ed.keys())

    lg = [mpatches.Patch(facecolor="w", edgecolor="green", label="Boder City")]

    # refugee
    ref_figure, ref_ax = plt.subplots(1, 1, figsize=(6, 5), layout="constrained")

    conflict_df.plot(
        column=f"conflict_{tks[1]}",
        ax=ref_ax,
        legend=True,
        cmap=CMAP_CONFLICT,
        legend_kwds={
            "label": "Number of the conflict spots",
            "orientation": "vertical",
            "extend": "both",
            "shrink": 0.5,
        },
    )

    boundary.plot(ax=ref_ax, facecolor="None", edgecolor=BOUNDARY_COLOR, lw=1)
    coal_gdf.plot(
        ax=ref_ax,
        color=COAL_COLOR,
        markersize=20,
        label="CPP",
    )
    bound_lv2.plot(ax=ref_ax, facecolor="None", edgecolor="black", lw=0.2)
    handles, _ = ref_ax.get_legend_handles_labels()
    ref_ax.legend(handles=[*lg, *handles], loc="lower left")
    ref_ax.set_title(f"2022[{tks[1]}]")


def plot_conflict_war_adm2():
    conflict_df = get_monthly_conflict()
    coal_gdf = gpd.read_file(UK_COAL_SHP)
    boundary = get_boundary_cities()
    bound_lv2 = gpd.read_file(UK_SHP_ADM2)

    sd_ed = PERIOD_DICT[2022]
    tks = list(sd_ed.keys())[2:]

    gs = gridspec.GridSpec(2, 6)
    fig = plt.figure(figsize=(18, 10), constrained_layout=True)

    lg = [mpatches.Patch(facecolor="w", edgecolor="green", label="Border City")]

    axs = []
    for i in range(0, 5):
        if i < 3:
            ax = fig.add_subplot(gs[0, i * 2 : i * 2 + 2])
        elif i > 2:
            ax = fig.add_subplot(gs[1, (i - 2) * 2 - 1 : (i - 2) * 2 + 1])
        conflict_df.plot(
            column=f"conflict_{tks[i]}",
            ax=ax,
            vmin=0,
            vmax=300,
            legend=False,
            cmap=CMAP_CONFLICT,
        )

        boundary.plot(ax=ax, facecolor="None", edgecolor=BOUNDARY_COLOR, lw=1)
        coal_gdf.plot(
            ax=ax,
            color=COAL_COLOR,
            markersize=20,
            label="CPP",
        )
        bound_lv2.plot(ax=ax, facecolor="None", edgecolor="black", lw=0.2)
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles=[*lg, *handles], loc="lower left")
        ax.set_title(f"2022[{tks[i]}]")
        axs += [ax]
    norm = plt.Normalize(0, 300)
    sm = plt.cm.ScalarMappable(cmap=CMAP_CONFLICT, norm=norm)
    sm.set_array([])
    fig.colorbar(
        sm,
        ax=axs,
        # ax=ax[:,1],
        # fraction=0.47,
        orientation="horizontal",
        extend="both",
        label="Number of the conflict spots",
        location="bottom",
        shrink=0.4,
    )


def plot_wind_rose(ds, year_src, event="covid"):

    # border_df = get_boundary_cities()
    # conflict_df = get_monthly_conflict()
    adm2_col = "ADM2_EN"
    wind_var = "wind"
    u10_var = "u10"
    v10_var = "v10"

    year_target = 2020 if event == "covid" else 2022
    # fig = (
    #     plt.figure(figsize=(10, 5), constrained_layout=True)
    #     if event == "covid"
    #     else plt.figure(figsize=(15, 10), constrained_layout=True)
    # )

    bins = [i for i in range(0, 7)]

    u10 = ds.era5[u10_var]
    v10 = ds.era5[v10_var]
    ds.era5[wind_var] = np.sqrt(u10**2 + v10**2)
    sd_ed = PERIOD_DICT[year_target]
    tks = list(sd_ed.keys())
    
    ncols = 2
    nrows = int(len(tks)/2)

    fig = plt.figure(figsize=(ncols * 5, nrows*5), constrained_layout=True)
    fig.suptitle(f"Wind speed and direction in {year_src}")

    for i, tk in enumerate(tks):
        t = sd_ed[tk]
        sd = np.datetime64(f"{year_src}-{t['sm']}-{t['sd']}{HOUR_STR}")
        ed = np.datetime64(f"{year_src}-{t['em']}-{t['ed']}{HOUR_STR}")
        july_wind_ds = ds.era5.sel(time=slice(sd, ed))[[wind_var, u10_var, v10_var]]

        # threshold_point = 2 if i == 0 else THRESHOLD_CONFLICT_POINT
        # event_bound = conflict_df.loc[conflict_df[f"conflict_{tk}"] > threshold_point]

        # if event == "border":
        #     event_bound = border_df

        wind_flat = []
        u10_flat = []
        v10_flat = []
        # for adm2 in event_bound[adm2_col].values:
        #     geometry = event_bound.loc[event_bound[adm2_col] == adm2].geometry

        #     clip_ds = july_wind_ds.rio.clip(geometry, event_bound.crs)

        wind_flat = np.concatenate(
            (wind_flat, july_wind_ds[wind_var].values.reshape(-1)), axis=None
        )
        u10_flat = np.concatenate(
            (u10_flat, july_wind_ds[u10_var].values.reshape(-1)), axis=None
        )
        v10_flat = np.concatenate(
            (v10_flat, july_wind_ds[v10_var].values.reshape(-1)), axis=None
        )

        wind_flat = wind_flat[~np.isnan(wind_flat)]
        u10_flat = u10_flat[~np.isnan(u10_flat)]
        v10_flat = v10_flat[~np.isnan(v10_flat)]

        u10_flat = units.Quantity(u10_flat, "m/s")
        v10_flat = units.Quantity(v10_flat, "m/s")

        wind_d = mpcalc.wind_direction(u10_flat, v10_flat)
        # ax = (
        #     fig.add_subplot(1, 2, i + 1, projection="windrose")
        #     if event == "covid"
        #     else fig.add_subplot(2, 3, i + 1, projection="windrose")
        # )
        ax = fig.add_subplot(nrows, ncols, i+1, projection="windrose")

        # ax_w = WindroseAxes.from_ax(ax[i])
        ax.contourf(
            wind_d.magnitude,
            wind_flat,
            normed=True,
            bins=bins,
            lw=3,
            cmap=cm.Spectral_r,
        )
        ax.contour(wind_d.magnitude, wind_flat, normed=True, bins=bins, colors="black")
        # ax.bar(wind_d.magnitude, wind_flat, bins=bins, normed=True, opening=0.8, edgecolor='white')
        ax.set_legend(title="Windspeed: m/s")
        ax.set_title(f"{tk}")

    return wind_d, wind_flat


# %%
