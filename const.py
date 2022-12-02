UK_BOUND = {
    "min_lat": 44.386411,
    "max_lat": 52.379291,
    "min_lon": 22.136912,
    "max_lon": 40.227883,
}

PERIOD_DICT = {
    2022: {
        # "January": {"sd": "01", "sm": "01", "ed": "01", "em": "02"},
        # "Febuary": {"sd": "01", "sm": "02", "ed": "01", "em": "03"},
        "Febuary_before_war": {"sd": "01", "sm": "02", "ed": 24, "em": "02"},
        "Febuary_war": {"sd": 24, "sm": "02", "ed": "01", "em": "03"},
        "Refugee_time": {"sd": "01", "sm": "03", "ed": "09", "em": "03"},
        # "Lockdown": {"sd": 25, "sm": "03", "ed": 25, "em": "04"},
        "March": {"sd": "01", "sm": "03", "ed": "01", "em": "04"},
        "April": {"sd": "01", "sm": "04", "ed": "01", "em": "05"},
        "May": {"sd": "01", "sm": "05", "ed": "01", "em": "06"},
        "June": {"sd": "01", "sm": "06", "ed": "01", "em": "07"},
        "July": {"sd": "01", "sm": "07", "ed": "01", "em": "08"},
    },
    2021: {
        # "January": {"sd": "01", "sm": "01", "ed": "01", "em": "02"},
        # "Febuary": {"sd": "01", "sm": "02", "ed": "01", "em": "03"},
        # "Febuary_before_war": {"sd": "01", "sm": "02", "ed": 24, "em": "02"},
        "Febuary_war": {"sd": 24, "sm": "02", "ed": "01", "em": "03"},
        # "Refugee_time": {"sd": "01", "sm": "03", "ed": "09", "em": "03"},
        "March": {"sd": "01", "sm": "03", "ed": "01", "em": "04"},
        "April": {"sd": "01", "sm": "04", "ed": "01", "em": "05"},
        "May": {"sd": "01", "sm": "05", "ed": "01", "em": "06"},
        "June": {"sd": "01", "sm": "06", "ed": "01", "em": "07"},
        "July": {"sd": "01", "sm": "07", "ed": "01", "em": "08"},
    },
    2020: {
        # "January": {"sd": "01", "sm": "01", "ed": "01", "em": "02"},
        # "Febuary": {"sd": "01", "sm": "02", "ed": "01", "em": "03"},
        # "Febuary_bf_war": {"sd": "01", "sm": "02", "ed": 24, "em": "02"},
        "Febuary_war": {"sd": 24, "sm": "02", "ed": "01", "em": "03"},
        # "March_before_Lockdown": {"sd": "01", "sm": "03", "ed": 25, "em": "03"},
        # "Lockdown": {"sd": 25, "sm": "03", "ed": 25, "em": "04"},
        # "Lockdown": {"sd": 25, "sm": "03", "ed": 11, "em": "05"},
        "March": {"sd": "01", "sm": "03", "ed": "01", "em": "04"},
        "April": {"sd": "01", "sm": "04", "ed": "01", "em": "05"},
        "May": {"sd": "01", "sm": "05", "ed": "01", "em": "06"},
        "June": {"sd": "01", "sm": "06", "ed": "01", "em": "07"},
        "July": {"sd": "01", "sm": "07", "ed": "01", "em": "08"},
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

RANDOM_GRID = {
    "min_samples_leaf": [1, 3, 5, 7, 10],
    "n_estimators": [200, 400, 600, 800],
}

NO2_UNIT = f"$10^{{{-6}}}$ $mol/m^2$"

S5P_OBS_COL = r"Observed_S5P"
S5P_PRED_COL = r"Deweathered_S5P"
OBS_PRED_CHNAGE = r"Observed_Deweathered_Difference"

ERA5_COLS = ["u10", "v10", "d2m", "t2m", "blh", "z"]
POP_COLS = ["pop"]
S5P_COLS = [S5P_OBS_COL]
CAMS_COLS = ["cams_no2"]
