#%%
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from utils import *
from const import *

#%%
# Bubble plot
def plot_change_bubble(geo_df, cols):
    cmap = "coolwarm"
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


def plot_obs_bubble():

    org_ds = prep_s5p_ds()
    bound_lv2, crs = get_bound_pop_lv2()
    adm_col = "ADM2_EN"
    list_city = bound_lv2[adm_col].values

    # sd_ed = PERIOD_DICT[2022]
    sd_ed = PERIOD_DICT[2020]

    # years = [2019, 2020, 2021, 2022]
    years = [2019, 2020]

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
                    .mean("time")[[S5P_OBS_COL]]
                )
                obs_dict_year[f"{year}_{tk}"].append(adm_ds[S5P_OBS_COL].item())
            bound_lv2[f"{year}_{tk}"] = obs_dict_year[f"{year}_{tk}"]

    change_dict = {}
    for y in years[:-1]:
        for tk in sd_ed.keys():
            change_dict[f"OBS_2020_{y}_{tk}"] = (
                (bound_lv2[f"2020_{tk}"] - bound_lv2[f"{y}_{tk}"])
                * 100
                / bound_lv2[f"{y}_{tk}"]
            )

    df_no2_change = pd.DataFrame.from_dict(change_dict)
    df_no2_change["Population"] = bound_lv2["Population"].values
    geo_df = gpd.GeoDataFrame(
        df_no2_change, crs=crs, geometry=bound_lv2.geometry.centroid
    )

    plot_change_bubble(geo_df, list(change_dict.keys()))
    return geo_df


def plot_obs_bau_bubble(org_ds, year):

    ds = prep_ds(org_ds, year)

    bound_lv1 = gpd.read_file(UK_SHP_ADM1)
    bound_lv2, crs = get_bound_pop_lv2()

    list_city = bound_lv2["ADM2_EN"].values
    sd_ed = PERIOD_DICT[year]

    dict_no2_change = {}

    tks = list(sd_ed.keys())

    for tk in tks:

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
                .mean("time")[[S5P_PRED_COL, S5P_OBS_COL]]
            )
            dict_no2_change[tk].append(
                (city_ds[S5P_OBS_COL].item() - city_ds[S5P_PRED_COL].item())
                * 100
                / city_ds[S5P_PRED_COL].item()
            )

    df_no2_change = pd.DataFrame.from_dict(dict_no2_change)
    df_no2_change["Population"] = bound_lv2["Population"].values
    geo_df = gpd.GeoDataFrame(
        df_no2_change, crs=crs, geometry=bound_lv2.geometry.centroid
    )

    cmap = "seismic"
    nrow = int(len(tks) / 2)
    # figure, ax = plt.subplots(nrow, 2, figsize=(22, 13))
    figure, ax = plt.subplots(nrow, 2, figsize=(22, 6.5 * nrow))

    j = 0
    for i, col in enumerate(tks):

        i = int(i / 2)
        j = 0 if j > 1 else j

        bound_lv1 = gpd.read_file(UK_SHP_ADM1)
        bound_lv1.plot(ax=ax[i][j], facecolor="white", edgecolor="black", lw=0.7)

        g = sns.scatterplot(
            data=geo_df,
            x=geo_df.centroid.x,
            y=geo_df.centroid.y,
            hue=col,
            hue_norm=(-20, 20),
            size="Population",
            sizes=(150, 500),
            palette=cmap,
            ax=ax[i][j],
        )

        g.legend(bbox_to_anchor=(1.0, 1.0), ncol=1, bbox_transform=ax[i][j].transAxes)

        norm = plt.Normalize(-20, 20)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        clb = g.figure.colorbar(sm, ax=ax[i][j])
        clb.ax.set_ylabel(r"NO$_{2}$ col. change (%)")
        clb.ax.yaxis.set_label_position("right")
        g.set(title=rf"Observered_NO$_{2}$ - Deweathered_NO$_{2}$ {col} - {year}")

        h, l = g.get_legend_handles_labels()
        l = [f"{li}M" for li in l]
        legend = ax[i][j].legend(
            h[-7:],
            l[-7:],
            bbox_to_anchor=(0, 0),
            loc="lower left",
            borderaxespad=0.0,
            fontsize=13,
            edgecolor="black"
            # bbox_transform=ax[i][j].transAxes,
        )
        legend.get_frame().set_alpha(None)
        # legend.get_frame().set_facecolor((0, 0, 1, 0.1))
        j += 1

    # return geo_df


#%%
# Px plot
def plot_obs_bau_map(org_ds, year):

    ds = prep_ds(org_ds, year)
    coal_gdf = gpd.read_file(UK_COAL_SHP)
    sd_ed = PERIOD_DICT[year]
    adm_col = "ADM2_EN"

    tks = list(sd_ed.keys())
    nrow = int(len(tks) / 2)
    figure, ax = plt.subplots(nrow, 2, figsize=(16, 6.5 * nrow))
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

        change_ds.plot(
            ax=ax[i][j],
            cmap="seismic",
            vmin=-100,
            vmax=100,
            cbar_kwargs={
                "label": r"NO$_{2}$ col. change (%)",
                "orientation": "horizontal",
                "fraction": 0.047,
                "extend": "both",
            },
        )
        # bound_lv0.plot(ax=ax[i][j], facecolor="None", edgecolor="black", lw=2)
        coal_gdf.plot(
            ax=ax[i][j], color="green", markersize=30, label="Coal power plant"
        )
        bound_lv1.plot(ax=ax[i][j], facecolor="None", edgecolor="black", lw=0.2)
        ax[i][j].set_xlabel("longitude")
        ax[i][j].set_ylabel("latitude")
        ax[i][j].legend()
        ax[i][j].set_title(tk)

        ax[i][j].set_xlim([22, 41])
        ax[i][j].set_ylim([44, 53])
        j += 1
        # plt.title(tk, fontsize=18)
    plt.suptitle(rf"Observed_Deweathered_NO$_{2}$_Difference_{year}", fontsize=18)
    plt.tight_layout()
    plt.subplots_adjust(top=0.98)


def plot_obs_change_map():

    bound_lv1 = gpd.read_file(UK_SHP_ADM1)

    org_ds = prep_s5p_ds()

    sd_ed = PERIOD_DICT[2020]
    years = [2019]
    # sd_ed = PERIOD_DICT[2022]
    # years = [2019, 2020, 2021]
    tks = list(sd_ed.keys())
    nrow = int(len(tks) / 2)
    figure, ax = plt.subplots(nrow, 2, figsize=(16, 6.5 * nrow))

    for y in years:
        j = 0
        for i, tk in enumerate(tks):

            i = int(i / 2)
            j = 0 if j > 1 else j

            t = sd_ed[tk]

            sd_event = np.datetime64(f"2020-{t['sm']}-{t['sd']}T00:00:00.000000000")
            ed_event = np.datetime64(f"2020-{t['em']}-{t['ed']}T00:00:00.000000000")

            sd_y = np.datetime64(f"{y}-{t['sm']}-{t['sd']}T00:00:00.000000000")
            ed_y = np.datetime64(f"{y}-{t['em']}-{t['ed']}T00:00:00.000000000")

            ds_event = org_ds.sel(time=slice(sd_event, ed_event)).mean("time")[
                [S5P_OBS_COL]
            ]
            ds_y = org_ds.sel(time=slice(sd_y, ed_y)).mean("time")[[S5P_OBS_COL]]

            ds_change = (
                (ds_event[S5P_OBS_COL] - ds_y[S5P_OBS_COL]) * 100 / ds_y[S5P_OBS_COL]
            )

            ds_change.plot(
                ax=ax[i][j],
                cmap="seismic",
                vmin=-70,
                vmax=70,
                cbar_kwargs={
                    "label": r"NO$_{2}$ col. change (%)",
                    "orientation": "horizontal",
                    "fraction": 0.047,
                    "extend": "both",
                },
            )
            bound_lv1.plot(ax=ax[i][j], facecolor="None", edgecolor="black", lw=0.2)
            # plt.title(f"OBS_2022_{y}_{tk}", fontsize=18)
            ax[i][j].set_xlabel("longitude")
            ax[i][j].set_ylabel("latitude")
            ax[i][j].set_title(tk)

            ax[i][j].set_xlim([22, 41])
            ax[i][j].set_ylim([44, 53])
            j += 1
    plt.suptitle(rf"Observed_NO$_{2}$_Difference_2020_{y}", fontsize=18)
    # plt.tight_layout()
    # plt.subplots_adjust(top=0.9)


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

    bound_lv2, crs = get_bound_pop_lv2()
    list_city = bound_lv2["ADM2_EN"].values

    for city in list_city:
        geometry = bound_lv2.loc[bound_lv2["ADM2_EN"] == city].geometry
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

    axis[0].set_title(
        "Time series trend of observation NO2 and Machine learning NO2 prediction"
    )
    axis[0].set_ylabel("Date")
    axis[0].set_ylabel(f"$10^{{{-6}}}$ $mol/m^2$")

    axis[1].set_title("NO2 Scatter Plot")
    axis[1].set_ylabel(f"NO2 S5P Obs $10^{{{-6}}}$ $mol/m^2$")
    axis[1].set_ylabel(f"NO2 ML predictions $10^{{{-6}}}$ $mol/m^2$")
    axis[1].annotate(
        "$R^2$ = {:.3f}".format(
            r2_score(ds.test_2019[S5P_OBS_COL], ds.test_2019[S5P_PRED_COL])
        ),
        (10, 300),
    )
    line = mlines.Line2D([0, 1], [0, 1], color="red")
    transform = axis[1].transAxes
    line.set_transform(transform)
    axis[1].add_line(line)


#%%
# Plot fire location and conflict point


def plot_fire_conflict(year, data_type="Fire Spot"):

    data_df = prep_fire_df() if data_type == "Fire Spot" else prep_conflict_df()
    color = "orange" if data_type == "Fire Spot" else "red"

    nucl_gdf = gpd.read_file(UK_NUC_SHP)
    coal_gdf = gpd.read_file(UK_COAL_SHP)

    bound_lv0 = gpd.read_file(UK_SHP_ADM0)
    bound_lv1 = gpd.read_file(UK_SHP_ADM1)

    sd_ed = PERIOD_DICT[year]

    for tk in sd_ed.keys():

        t = sd_ed[tk]
        figure, ax = plt.subplots(figsize=(16, 8))

        sd = np.datetime64(f"{year}-{t['sm']}-{t['sd']}T00:00:00.000000000")
        ed = np.datetime64(f"{year}-{t['em']}-{t['ed']}T00:00:00.000000000")

        mask = (data_df["DATETIME"] > sd) & (data_df["DATETIME"] <= ed)
        df = data_df.loc[mask]

        df.plot(ax=ax, color=color, markersize=10, label=data_type)
        nucl_gdf.plot(ax=ax, color="green", markersize=100, label="Nuclear")
        coal_gdf.plot(ax=ax, color="blue", markersize=100, label="Coal")
        bound_lv1.plot(ax=ax, facecolor="None", edgecolor="black", lw=0.4)
        bound_lv0.plot(ax=ax, facecolor="None", edgecolor="black", lw=2)
        ax.legend()
        plt.title(tk, fontsize=18)


# %%
def plot_ax_line(ds, geometry, ppl_name, gdf, ax, year, set_ylabel=False):

    vl_covid_clr = "#252525"
    vl_war_clr = "#252525"
    label_war = "War start date"
    label_covid = "Lockdown"

    ls_covid = "dashed"
    ls_war = "solid"
    lw = 1.5

    pred_truth_diff = ["cyan", "red", "#feb24c"]

    sd = np.datetime64(f"{year}-02-01T00:00:00.000000000")
    ed = np.datetime64(f"{year}-07-31T00:00:00.000000000")
    ds_clip = (
        ds.rio.clip(geometry, gdf.crs)
        .sel(time=slice(sd, ed))
        .mean(dim=["lat", "lon"])[[S5P_PRED_COL, S5P_OBS_COL]]
    )
    org_df = ds_clip.to_dataframe()
    df = get_nday_mean(org_df, nday=3)

    df[OBS_PRED_CHNAGE] = df[S5P_OBS_COL] - df[S5P_PRED_COL]
    df[[S5P_PRED_COL, S5P_OBS_COL, OBS_PRED_CHNAGE]].plot.line(
        ax=ax, color=pred_truth_diff, legend=False
    )

    if set_ylabel:
        ax.set_ylabel(NO2_UNIT)
    ax.grid(color="#d9d9d9")
    ax.set_title(f"{ppl_name}-{year}")
    handles, labels = ax.get_legend_handles_labels()

    ax.axhline(
        y=0,
        color="black",
        linewidth=1,
        linestyle="--",
    )

    if year == 2020:
        ax.axvline(
            x=np.datetime64(f"{year}-03-25T00:00:00.000000000"),
            color=vl_covid_clr,
            linewidth=lw,
            linestyle=ls_covid,
            label=label_covid,
        )
        ax.axvline(
            x=np.datetime64(f"{year}-05-11T00:00:00.000000000"),
            color=vl_covid_clr,
            linewidth=lw,
            linestyle=ls_covid,
        )

    elif year == 2021:
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

    elif year == 2022:
        ax.axvline(
            x=np.datetime64(f"{year}-02-24T00:00:00.000000000"),
            color=vl_war_clr,
            linewidth=lw,
            linestyle=ls_war,
            label=label_war,
        )

    return handles, labels


def plot_ppl_obs_bau_line_mlt(org_ds):

    ds_2020 = prep_ds(org_ds, 2020)
    ds_2021 = prep_ds(org_ds, 2021)
    ds_2022 = prep_ds(org_ds, 2022)

    coal_gdf = gpd.read_file(UK_COAL_SHP)
    # coal_gdf["buffer"] = coal_gdf.geometry.buffer(0.15, cap_style=3).to_crs(
    #     coal_gdf.crs
    # )

    for i, ppl_name in enumerate(coal_gdf.name.values):

        geometry = coal_gdf.loc[coal_gdf["name"] == ppl_name].geometry
        fig, ax = plt.subplots(1, 3, figsize=(16, 4))

        plot_ax_line(
            ds_2020, geometry, ppl_name, coal_gdf, ax[0], 2020, set_ylabel="True"
        )
        handles, labels = plot_ax_line(
            ds_2021, geometry, ppl_name, coal_gdf, ax[1], 2021
        )
        plot_ax_line(ds_2022, geometry, ppl_name, coal_gdf, ax[2], 2022)

        fig.legend(
            handles, labels, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.01)
        )


def plot_obs_bau_pop_line_mlt(org_ds):

    ds_2020 = prep_ds(org_ds, 2020)
    ds_2021 = prep_ds(org_ds, 2021)
    ds_2022 = prep_ds(org_ds, 2022)

    bound_lv2, crs = get_bound_pop_lv2()

    for i, city in enumerate(bound_lv2["ADM2_EN"].values):

        geometry = bound_lv2.loc[bound_lv2["ADM2_EN"] == city].geometry
        fig, ax = plt.subplots(1, 3, figsize=(16, 4))

        plot_ax_line(ds_2020, geometry, city, bound_lv2, ax[0], 2020, set_ylabel=True)
        handles, labels = plot_ax_line(ds_2021, geometry, city, bound_lv2, ax[1], 2021)
        plot_ax_line(ds_2022, geometry, city, bound_lv2, ax[2], 2022)

        fig.legend(
            handles, labels, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.01)
        )


def plot_obs_bau_pop_line_sgl(org_ds, year):
    ds = prep_ds(org_ds, year)
    bound_lv2, crs = get_bound_pop_lv2()
    # bound_lv2 = gpd.read_file(UK_SHP_ADM2)
    city_no2 = {}

    sd = np.datetime64(f"{year}-02-01T00:00:00.000000000")
    ed = np.datetime64(f"{year}-07-31T00:00:00.000000000")

    for i, city in enumerate(bound_lv2["ADM2_EN"].values):
        fig = plt.figure(1 + i, figsize=(6, 4))
        ax = plt.subplot(1, 1, 1)
        geometry = bound_lv2.loc[bound_lv2["ADM2_EN"] == city].geometry
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


def plot_weather_params(ds):
    year = 2019
    sd = np.datetime64(f"{year}-01-01T00:00:00.000000000")
    ed = np.datetime64(f"{year}-03-31T00:00:00.000000000")
    ts = ds.era5.sel(time=slice(sd, ed))

    year = 2022
    sd = np.datetime64(f"{year}-01-01T00:00:00.000000000")
    ed = np.datetime64(f"{year}-03-31T00:00:00.000000000")
    ts_2022 = ds.era5.sel(time=slice(sd, ed))

    figure, ax = plt.subplots(figsize=(10, 6))
    xr.plot.hist(ts_2022["d2m"], ec="b", fc="None", histtype="step")
    xr.plot.hist(ts["d2m"], ec="r", fc="None", histtype="step")


def plot_obs_bau_adm2(org_ds, year):

    ds = prep_ds(org_ds, year)

    conflict_ds = prep_conflict_df()
    fire_ds = prep_fire_df()

    bound_lv2 = gpd.read_file(UK_SHP_ADM2)
    sd_ed = PERIOD_DICT[year]

    adm_col = "ADM2_EN"

    dict_no2_change = {}
    dict_conflict_change = {}
    dict_fire_change = {}

    list_adm = bound_lv2[adm_col].values

    tks = list(sd_ed.keys())[2:]
    for tk in tks:

        dict_no2_change[tk] = []
        dict_conflict_change[tk] = []
        dict_fire_change[tk] = []

        t = sd_ed[tk]
        sd = np.datetime64(f"{year}-{t['sm']}-{t['sd']}T00:00:00.000000000")
        ed = np.datetime64(f"{year}-{t['em']}-{t['ed']}T00:00:00.000000000")

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

    figure, ax = plt.subplots(len(tks), 3, figsize=(30, 7 * len(tks)))

    for i, tk in enumerate(tks):
        bound_lv2.plot(
            column=f"war_{tk}",
            ax=ax[i][0],
            legend=True,
            cmap="coolwarm",
            vmin=-20,
            vmax=20,
            legend_kwds={
                "label": r"NO$_{2}$ col. change (%)",
                "orientation": "horizontal",
            },
        )
        bound_lv2.plot(
            column=f"conflict_{tk}",
            ax=ax[i][1],
            legend=True,
            cmap="Reds",
            legend_kwds={
                "label": "Number of conflict spots",
                "orientation": "horizontal",
            },
        )

        bound_lv2.plot(
            column=f"fire_{tk}",
            ax=ax[i][2],
            legend=True,
            cmap="OrRd",
            legend_kwds={"label": "Number of fire spots", "orientation": "horizontal"},
        )

        for j in range(len(ax[i])):
            bound_lv2.plot(ax=ax[i][j], facecolor="None", edgecolor="black", lw=0.2)
            ax[i][j].legend(loc="upper center", bbox_to_anchor=(0.5, -0.01))

        ax[i][0].set_title(f"Observation and Deweather Difference -{tk}-{year}")
        ax[i][1].set_title(f"Conflict Locations-{tk}-{year}")
        ax[i][2].set_title(f"Fire Locations-{tk}-{year}")
    # plt.title(tk, fontsize=18)
