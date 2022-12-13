#%%
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from utils import *
from const import *

# Bubble plot
def plot_obs_bubble(event="war"):

    bound_lv1 = gpd.read_file(UK_SHP_ADM1)

    org_ds = prep_s5p_ds()
    bound_lv2, crs = get_bound_pop_lv2()
    adm_col = "ADM2_EN"
    list_city = bound_lv2[adm_col].values

    year_target = 2022 if event == "war" else 2020
    sd_ed = PERIOD_DICT[year_target]
    tks = list(sd_ed.keys())
    year_srcs = [i for i in range(2019, year_target)]

    obs_dict_year = {}

    cmap = "RdYlBu_r"
    nrows = int((len(year_srcs) + 1) / 2)
    ncols = 2

    self_figure, self_ax = plt.subplots(
        nrows, 2, figsize=(6 * ncols, 5 * nrows), layout="constrained"
    )

    j = 0
    for i, year in enumerate(year_srcs + [year_target]):
        i = int(i / 2)
        j = 0 if j > 1 else j
        for tk in tks[:2]:
            obs_dict_year[f"{year}_{tk}"] = []
            t = sd_ed[tk]
            sd = np.datetime64(f"{year}-{t['sm']}-{t['sd']}{HOUR_STR}")
            ed = np.datetime64(f"{year}-{t['em']}-{t['ed']}{HOUR_STR}")

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

        col = f"{year} {tks[1]} - {tks[0]} Difference"
        bound_lv2[col] = (
            (bound_lv2[f"{year}_{tks[1]}"] - bound_lv2[f"{year}_{tks[0]}"])
            * 100
            / bound_lv2[f"{year}_{tks[0]}"]
        )
        # Plot
        ax = self_ax[i][j] if nrows > 1 else self_ax[j]
        bound_lv1.plot(ax=ax, facecolor="white", edgecolor="black", lw=0.7)
        norm_val = 20
        g = sns.scatterplot(
            data=bound_lv2,
            x=bound_lv2.centroid.x,
            y=bound_lv2.centroid.y,
            hue=col,
            hue_norm=(-1 * norm_val, norm_val),
            size="Population",
            sizes=(150, 500),
            palette=cmap,
            ax=ax,
        )
        norm = plt.Normalize(-1 * norm_val, norm_val)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        g.set(title=rf"{col}")

        h, l = g.get_legend_handles_labels()
        l = [f"{li}M" if li != "Population" else li for li in l]
        legend = ax.legend(
            h[-7:],
            l[-7:],
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
    plt.suptitle(rf"Annual Before-During changes", fontsize=18)

    # inter bf af change
    nrows = len(year_srcs)
    ncols = 2
    inter_figure, inter_ax = plt.subplots(
        nrows, ncols, figsize=(6 * ncols, 5 * nrows), layout="constrained"
    )
    j = 0

    for i, year_src in enumerate(year_srcs):
        for j, tk in enumerate(tks[:2]):

            col = f"{year_target}_{tk} - {year_src}_{tk} Difference"
            bound_lv2[col] = (
                (bound_lv2[f"{year_target}_{tk}"] - bound_lv2[f"{year_src}_{tk}"])
                * 100
                / bound_lv2[f"{year_src}_{tk}"]
            )
            ax = inter_ax[i][j] if nrows > 1 else inter_ax[j]
            bound_lv1.plot(ax=ax, facecolor="white", edgecolor="black", lw=0.7)
            g = sns.scatterplot(
                data=bound_lv2,
                x=bound_lv2.centroid.x,
                y=bound_lv2.centroid.y,
                hue=col,
                hue_norm=(-1 * norm_val, norm_val),
                size="Population",
                sizes=(150, 500),
                palette=cmap,
                ax=ax,
            )
            norm = plt.Normalize(-1 * norm_val, norm_val)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])

            g.set(title=rf"{col}")

            h, l = g.get_legend_handles_labels()
            l = [f"{li}M" if li != "Population" else li for li in l]
            legend = ax.legend(
                h[-7:],
                l[-7:],
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
    plt.suptitle(rf"Interannual changes", fontsize=18)
    # plt.tight_layout()


def plot_obs_change_adm2():
    bound_lv2 = gpd.read_file(UK_SHP_ADM2)
    adm_col = "ADM2_EN"
    coal_gdf = gpd.read_file(UK_COAL_SHP)
    org_ds = prep_s5p_ds()

    year_target = 2022
    sd_ed = PERIOD_DICT[year_target]
    year_srcs = [i for i in range(2019, year_target)]

    tks = list(sd_ed.keys())

    pixel_change_dict = {}

    for year in (year_srcs + [year_target]):

        for tk in tks:
            t = sd_ed[tk]
            sd = np.datetime64(f"{year}-{t['sm']}-{t['sd']}{HOUR_STR}")
            ed = np.datetime64(f"{year}-{t['em']}-{t['ed']}{HOUR_STR}")
            pixel_change_dict[f"{year}_{tk}"] = org_ds.sel(time=slice(sd, ed)).mean("time")[
                [S5P_OBS_COL]
            ]
        pixel_change_dict[f"self_{year}"] = (pixel_change_dict[f"{year}_{tks[1]}"] - pixel_change_dict[f"{year}_{tks[0]}"])*100/pixel_change_dict[f"{year}_{tks[0]}"]
        self_year_change = []
        for adm2 in bound_lv2[adm_col].values:
            geometry = bound_lv2.loc[bound_lv2[adm_col] == adm2].geometry
            self_year_change.append(pixel_change_dict[f"self_{year}"].rio.clip(geometry, bound_lv2.crs).mean(dim=["lat", "lon"])[S5P_OBS_COL].item())
        bound_lv2[f"self_{year}"] = self_year_change


    for tk in tks:
        for year in year_srcs:

            pixel_change_year = (pixel_change_dict[f"{year_target}_{tk}"] - pixel_change_dict[f"{year}_{tk}"])*100/pixel_change_dict[f"{year}_{tk}"]
            year_tk_items = []
            for adm2 in bound_lv2[adm_col].values:
                geometry = bound_lv2.loc[bound_lv2[adm_col] == adm2].geometry

                    # cal obs deweather no2 ds
                change_adm_no2_ds = (
                    pixel_change_year.rio.clip(geometry, bound_lv2.crs)
                    .mean(dim=["lat", "lon"])[S5P_OBS_COL].item()
                )
                year_tk_items.append(change_adm_no2_ds)
            bound_lv2[f"inter_{year}_{tk}"] = year_tk_items

    self_ref_fig, self_ref_ax = plt.subplots(2, 2, figsize=(6*2, 5 * 2), layout="constrained")
    inter_ref_fig, inter_ref_ax = plt.subplots(2, 3, figsize=(6*3, 5 * 2), layout="constrained")

    inter_war_fig, inter_war_ax = plt.subplots(len(tks[2:]), 3, figsize=(6 * 3, 5*len(tks[2:])), layout="constrained")

    #self refugee change
    j=0
    for i, year in enumerate(year_srcs + [year_target]):
        i = int(i/2)
        j = 0 if j > 1 else j
        col = f"self_{year}"
        bound_lv2.plot(
            column=col, ax=self_ref_ax[i][j], cmap="coolwarm", vmin=-70, vmax=70, legend=True, legend_kwds={
                        "label": "Number of fire spots",
                        "orientation": "horizontal",
                    },
        )
        self_ref_ax[i][j].set_title(col, fontsize=14)
        bound_lv2.plot(ax=self_ref_ax[i][j], facecolor="None", edgecolor="black", lw=0.2)
        self_ref_ax[i][j].legend(loc="upper center", bbox_to_anchor=(0.5, -0.01))
        j += 1
    #inter refugee change
    for i, tk in enumerate(tks[:2]):
        for j, year in enumerate(year_srcs):
            col = f"inter_{year}_{tk}"
            bound_lv2.plot(
                column=col, ax=inter_ref_ax[i][j], cmap="coolwarm", vmin=-70, vmax=70, legend=True, legend_kwds={
                            "label": "Number of fire spots",
                            "orientation": "horizontal",
                        },
            )
            inter_ref_ax[i][j].set_title(col, fontsize=14)
            bound_lv2.plot(ax=inter_ref_ax[i][j], facecolor="None", edgecolor="black", lw=0.2)
            inter_ref_ax[i][j].legend(loc="upper center", bbox_to_anchor=(0.5, -0.01))
    #inter war change
    for i, tk in enumerate(tks[2:]):
        for j, year in enumerate(year_srcs):
            col = f"inter_{year}_{tk}"
            bound_lv2.plot(
                column=col, ax=inter_war_ax[i][j], cmap="coolwarm", vmin=-30, vmax=30, legend=True, legend_kwds={
                            "label": "Number of fire spots",
                            "orientation": "horizontal",
                        },
            )
            inter_war_ax[i][j].set_title(col, fontsize=14)
            bound_lv2.plot(ax=inter_war_ax[i][j], facecolor="None", edgecolor="black", lw=0.2)
            inter_war_ax[i][j].legend(loc="upper center", bbox_to_anchor=(0.5, -0.01))


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

    cmap = "RdYlBu_r"
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
            sizes=(150, 500),
            palette=cmap,
            ax=sub_ax,
        )

        # g.legend(
        #     bbox_to_anchor=(1.0, 1.0),
        #     ncol=1,
        #     bbox_transform=sub_ax.transAxes,
        # )

        norm = plt.Normalize(-15, 15)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        # clb = g.figure.colorbar(
        #     sm,
        #     ax=ax[i][j - 1],
        #     fraction=0.047,
        #     orientation="horizontal",
        #     extend="both",
        #     label="NO$_{2}$ col. change (%)",
        # )

        g.set(title=rf"{col}")

        h, l = g.get_legend_handles_labels()
        l = [f"{li}M" if li != "Population" else li for li in l]
        legend = sub_ax.legend(
            h[-7:],
            l[-7:],
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
    plt.suptitle(
        rf"Observed_Deweathered_NO$_{2}$_Difference_{year} (Major cities)", fontsize=18
    )
    # plt.subplots_adjust(top=0.95)
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
            cmap="coolwarm",
            vmin=-100,
            vmax=100,
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
    # plt.tight_layout()
    # plt.subplots_adjust(top=0.95)


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
    figure_fire, ax_fire = plt.subplots(6, 3, figsize=(24, 5 * 6))
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
            ax_fire[i][j].set_title(f"{tk} - {y}", fontsize=18)

    # Plot conflict locations
    figure_conflict, ax_conflict = plt.subplots(4, 3, figsize=(24, 5 * 4))
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
            ax=ax_conflict[i][j], color="blue", markersize=20, label=label_coal
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
        ax_conflict[i][j].set_title(f"{tk} - 2022", fontsize=18)

        coal_gdf.plot(
            ax=ax_conflict[i + 1][j], color="blue", markersize=20, label=label_coal
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
        ax_conflict[i + 1][j].set_title(f"{tk} - 2022", fontsize=18)

        j = j + 1


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
    ax.set_ylim([-40, 200])

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


def plot_weather_params(ds, y_src=2019, y_tgrt=2020, var="wind"):
    # war start date before 02/09-24 - after 02/24 - 03/11
    # s_src_date = "02-09T00:00:00.000000000"
    # e_src_date = "02-24T00:00:00.000000000"

    # covid before 03/10-25 after 03/25-04/09
    s_src_date = "02-25T00:00:00.000000000"
    e_src_date = "03-25T00:00:00.000000000"
    title = "Feb-25 to Mar-25"

    ylabel = "Relative Frequency (%)"

    if var == "wind":
        xlabel = "Wind speed (m/s)"

        u10 = ds.era5["u10"]
        v10 = ds.era5["v10"]
        ds.era5[var] = np.sqrt(u10**2 + v10**2)
    elif var == "t2m":
        xlabel = "Temperature (K)"
    elif var == "blh":
        xlabel = "Boundary layer height (m)"

    sd = np.datetime64(f"{y_src}-{s_src_date}")
    ed = np.datetime64(f"{y_src}-{e_src_date}")
    ts_src = ds.era5.sel(time=slice(sd, ed)).mean("time")[var].values.reshape(-1)
    w_src = np.ones_like(ts_src) * 100 / ts_src.size

    sd = np.datetime64(f"{y_tgrt}-{s_src_date}")
    ed = np.datetime64(f"{y_tgrt}-{e_src_date}")
    t_tgrt = ds.era5.sel(time=slice(sd, ed)).mean("time")[var].values.reshape(-1)
    w_tgrt = np.ones_like(t_tgrt) * 100 / t_tgrt.size

    print(t_tgrt.shape, w_tgrt.shape)
    print(ts_src.shape, w_src.shape)

    figure, ax = plt.subplots(figsize=(4, 4))
    ax.hist(t_tgrt, weights=w_tgrt, ec="b", fc="None", histtype="step", label=y_tgrt)
    ax.hist(ts_src, weights=w_src, ec="r", fc="None", histtype="step", label=y_src)
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def plot_obs_bau_adm2(org_ds, year, mode="3_cf_no2_bau"):

    # mode =["2_cf", "2_no2_bau", "3_cf_no2_bau"]
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

    tks = list(sd_ed.keys())
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

    nrows = 1 if mode[0] == "2" else len(tks[2:])

    # mode =["2_cf", "2_no2_bau", "3_cf_no2_bau"]
    ncols = int(mode[0])
    figure, ax = plt.subplots(
        nrows, ncols, figsize=(6 * ncols, 5 * nrows), layout="constrained"
    )
    if mode == "2_cf":
        t = tks[1]
        bound_lv2.plot(
            column=f"conflict_{t}",
            ax=ax[0],
            legend=True,
            cmap="Reds",
            legend_kwds={
                "label": "Number of conflict spots",
                "orientation": "horizontal",
            },
        )

        bound_lv2.plot(
            column=f"fire_{t}",
            ax=ax[1],
            legend=True,
            cmap="OrRd",
            legend_kwds={
                "label": "Number of fire spots",
                "orientation": "horizontal",
            },
        )
        ax[0].set_title(f"Conflict Locations", fontsize=14)
        ax[1].set_title(f"Fire Locations", fontsize=14)
        for i in range(len(ax)):
            bound_lv2.plot(ax=ax[i], facecolor="None", edgecolor="black", lw=0.2)
            ax[i].legend(loc="upper center", bbox_to_anchor=(0.5, -0.01))
        plt.suptitle(rf"{year} ({t})", fontsize=18)
    elif mode == "2_no2_bau":
        for i in [0, 1]:
            bound_lv2.plot(
                column=f"war_{tks[i]}",
                ax=ax[i],
                legend=True,
                cmap="coolwarm",
                vmin=-20,
                vmax=20,
                legend_kwds={
                    "label": r"NO$_{2}$ col. change (%)",
                    "orientation": "horizontal",
                },
            )
            ax[i].set_title(f"({year} {tks[i]})", fontsize=14)
            bound_lv2.plot(ax=ax[i], facecolor="None", edgecolor="black", lw=0.2)
            ax[i].legend(loc="upper center", bbox_to_anchor=(0.5, -0.01))
        plt.suptitle(rf"OBS - BAU NO$_{2}$", fontsize=18)
    elif mode == "3_cf_no2_bau":
        for i, tk in enumerate(tks[2:]):
            legend = False if i < 4 else True
            bound_lv2.plot(
                column=f"war_{tk}",
                ax=ax[i][0],
                legend=legend,
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
                legend=legend,
                cmap="Reds",
                vmin=0,
                vmax=300,
                legend_kwds={
                    "label": "Number of conflict spots",
                    "orientation": "horizontal",
                },
            )

            bound_lv2.plot(
                column=f"fire_{tk}",
                ax=ax[i][2],
                legend=legend,
                cmap="OrRd",
                vmin=0,
                vmax=600,
                legend_kwds={
                    "label": "Number of fire spots",
                    "orientation": "horizontal",
                },
            )

            for j in range(len(ax[i])):
                bound_lv2.plot(ax=ax[i][j], facecolor="None", edgecolor="black", lw=0.2)
                ax[i][j].legend(loc="upper center", bbox_to_anchor=(0.5, -0.01))

            ax[i][0].set_title(rf"OBS - BAU NO$_{2}$ ({tk})", fontsize=14)
            ax[i][1].set_title(f"Conflict Locations ({tk})", fontsize=14)
            ax[i][2].set_title(f"Fire Locations ({tk})", fontsize=14)
        plt.suptitle(
            rf"OBS - BAU NO$_{2}$ , Conflict and Fire Locations ({year})",
            fontsize=18,
        )
    # plt.title(tk, fontsize=18)
