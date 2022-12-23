import matplotlib.patches as mpatches

UK_BOUND = {
    "min_lat": 44.386411,
    "max_lat": 52.379291,
    "min_lon": 22.136912,
    "max_lon": 40.227883,
}

HOUR_STR = "T00:00:00.000000000"

PERIOD_DICT = {
    2022: {
        "02/16_02/15": {"sd": "01", "sm": "02", "ed": "15", "em": "02"},
        "02/24_02/28": {"sd": "15", "sm": "02", "ed": "08", "em": "03"},
        # "Feb 24-28": {"sd": "01", "sm": "02", "ed": "01", "em": "03"},
        "March": {"sd": "01", "sm": "03", "ed": "01", "em": "04"},
        "April": {"sd": "01", "sm": "04", "ed": "01", "em": "05"},
        "May": {"sd": "01", "sm": "05", "ed": "01", "em": "06"},
        "June": {"sd": "01", "sm": "06", "ed": "01", "em": "07"},
        "July": {"sd": "01", "sm": "07", "ed": "01", "em": "08"},
    },
    2021: {
        "Feb 24-28": {"sd": "01", "sm": "02", "ed": "01", "em": "03"},
        "March": {"sd": "01", "sm": "03", "ed": "01", "em": "04"},
        "April": {"sd": "01", "sm": "04", "ed": "01", "em": "05"},
        "May": {"sd": "01", "sm": "05", "ed": "01", "em": "06"},
        "June": {"sd": "01", "sm": "06", "ed": "01", "em": "07"},
        "July": {"sd": "01", "sm": "07", "ed": "01", "em": "08"},
    },
    2020: {
        "02/08_03/25": {"sd": "08", "sm": "02", "ed": 25, "em": "03"},
        "03/25_05/11": {"sd": 25, "sm": "03", "ed": 11, "em": "05"},
        # "Feb 24-28": {"sd": "01", "sm": "02", "ed": "01", "em": "03"},
        # "March": {"sd": "01", "sm": "03", "ed": "01", "em": "04"},
        # "April": {"sd": "01", "sm": "04", "ed": "01", "em": "05"},
        # "May": {"sd": "01", "sm": "05", "ed": "01", "em": "06"},
        # "June": {"sd": "01", "sm": "06", "ed": "01", "em": "07"},
        # "July": {"sd": "01", "sm": "07", "ed": "01", "em": "08"},
    },
}

LIST_WAR_CITY = [
    "Mykolaivska",
    "Khersonska",
    "Zaporizka",
    "Donetska",
    "Luhanska",
    "Kharkivska",
    "Sumska",
    "Chernihivska",
    "Kyivska",
    "Kyiv",
]

LIST_BOUNDARY_CITY = [
    "Kovelskyi",
    "Volodymyr-Volynskyi",
    "Chervonohradskyi",
    "Lvivskyi",
    "Yavorivskyi",
    "Sambirskyi",
    "Uzhhorodskyi",
    "Berehivskyi",
    "Khustskyi",
    "Tiachivskyi",
    "Rakhivskyi",
    "Verkhovynskyi",
    "Vyzhnytskyi",
    "Chernivetskyi",
    "Dnistrovskyi",
    "Mohyliv-Podilskyi",
    "Tulchynskyi",
    "Podilskyi",
    "Rozdilnianskyi",
    "Odeskyi",
    "Bilhorod-Dnistrovskyi",
    "Bolhradskyi",
    "Bilhorod-Dnistrovskyi",
    "Izmailskyi",
]

RANDOM_GRID = {
    "min_samples_leaf": [1, 3, 5, 7, 10],
    "n_estimators": [200, 400, 600, 800],
}

NO2_UNIT = f"$10^{{{-6}}}$ $mol/m^2$"

S5P_OBS_COL = r"OBS_S5P"
S5P_PRED_COL = r"BAU_S5P"
OBS_PRED_CHNAGE = r"OBS_BAU_Difference"

ERA5_COLS = ["u10", "v10", "d2m", "t2m", "blh", "z"]
POP_COLS = ["pop"]
S5P_COLS = [S5P_OBS_COL]
CAMS_COLS = ["cams_no2"]

INDEX_FIG = [
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
]

COAL_COLOR = "#542788"
BOUNDARY_COLOR = "green"
CMAP_FIRE = "Reds"
CMAP_CONFLICT = "OrRd"
# CMAP_NO2 = "RdYlBu_r"
CMAP_NO2 = "bwr"
EDGE_COLOR_CONFLICT = "red"
EDGE_COLOR_BORDER = "green"
CMAP_WIND = "Spectral_r"

LG_CONFLICT = [
    mpatches.Patch(
        facecolor="w", edgecolor=EDGE_COLOR_CONFLICT, label="Reported Conflict"
    )
]
LG_BORDER = [
    mpatches.Patch(facecolor="w", edgecolor=EDGE_COLOR_BORDER, label="Border City")
]

THRESHOLD_CONFLICT_POINT = 70

NONE_OH_COLS = [
    "cams_no2",
    "u10",
    "v10",
    "d2m",
    "t2m",
    "blh",
    "z",
    "pop",
]
