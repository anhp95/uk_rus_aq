UK_BOUND = {
    "min_lat": 44.386411,
    "max_lat": 52.379291,
    "min_lon": 22.136912,
    "max_lon": 40.227883,
}

ERA5_COLS = ["u10", "v10", "d2m", "t2m", "blh", "z"]
POP_COLS = ["pop"]
S5P_COLS = ["s5p_no2"]
CAMS_COLS = ["cams_no2"]

PERIOD_DICT = {
    2022: {
        "Jan": {"sd": "01", "sm": "01", "ed": "01", "em": "02"},
        "Feb": {"sd": "01", "sm": "02", "ed": "01", "em": "03"},
        "Feb_bf_war": {"sd": "01", "sm": "02", "ed": 24, "em": "02"},
        "Feb_war": {"sd": 24, "sm": "02", "ed": "01", "em": "03"},
        "Refugee_time": {"sd": "01", "sm": "03", "ed": "09", "em": "03"},
        "Lockdown": {"sd": 25, "sm": "03", "ed": 25, "em": "04"},
        "Mar": {"sd": "01", "sm": "03", "ed": "01", "em": "04"},
        "Apr": {"sd": "01", "sm": "04", "ed": "01", "em": "05"},
        "May": {"sd": "01", "sm": "05", "ed": "01", "em": "06"},
        "Jun": {"sd": "01", "sm": "06", "ed": "01", "em": "07"},
        "Jul": {"sd": "01", "sm": "07", "ed": "01", "em": "08"},
    },
    2021: {
        "Jan": {"sd": "01", "sm": "01", "ed": "01", "em": "02"},
        "Feb": {"sd": "01", "sm": "02", "ed": "01", "em": "03"},
        "Feb_bf_war": {"sd": "01", "sm": "02", "ed": 24, "em": "02"},
        "Feb_war": {"sd": 24, "sm": "02", "ed": "01", "em": "03"},
        "Refugee_time": {"sd": "01", "sm": "03", "ed": "09", "em": "03"},
        "Mar": {"sd": "01", "sm": "03", "ed": "01", "em": "04"},
        "Apr": {"sd": "01", "sm": "04", "ed": "01", "em": "05"},
        "May": {"sd": "01", "sm": "05", "ed": "01", "em": "06"},
        "Jun": {"sd": "01", "sm": "06", "ed": "01", "em": "07"},
        "Jul": {"sd": "01", "sm": "07", "ed": "01", "em": "08"},
    },
    2020: {
        "Jan": {"sd": "01", "sm": "01", "ed": "01", "em": "02"},
        "Feb": {"sd": "01", "sm": "02", "ed": "01", "em": "03"},
        "Feb_bf_war": {"sd": "01", "sm": "02", "ed": 24, "em": "02"},
        "Feb_war": {"sd": 24, "sm": "02", "ed": "01", "em": "03"},
        "Mar_bf_lockdown": {"sd": "01", "sm": "03", "ed": 25, "em": "03"},
        "Lockdown": {"sd": 25, "sm": "03", "ed": 25, "em": "04"},
        "May": {"sd": "01", "sm": "05", "ed": "01", "em": "06"},
        "Jun": {"sd": "01", "sm": "06", "ed": "01", "em": "07"},
        "Jul": {"sd": "01", "sm": "07", "ed": "01", "em": "08"},
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
