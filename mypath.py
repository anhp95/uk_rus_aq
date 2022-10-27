#%%
import os
import glob

BASE_DIR = "data"

VIIR_DIR = os.path.join(BASE_DIR, "DL_FIRE_J1V-C2_302419")

FIRE_2020_SHP = os.path.join(VIIR_DIR, "fire_nrt_J1V-C2_302419-b4war-2020.shp")
FIRE_2021_SHP = os.path.join(VIIR_DIR, "fire_nrt_J1V-C2_302419-b4war-2021.shp")
FIRE_WARTIME_SHP = os.path.join(VIIR_DIR, "fire_nrt_J1V-C2_302419-wartime.shp")

CONFLICT_DIR = os.path.join(BASE_DIR, "conflict_location")
CONFLICT_XLS = os.path.join(CONFLICT_DIR, "Ukraine_Black_Sea_2020_2022_Oct07.xlsx")

BOUND_DIR = os.path.join(BASE_DIR, "uk_bound", "ukr_adm_sspe_20221005")
UK_SHP_ADM0 = os.path.join(BOUND_DIR, "ukr_admbnda_adm0_sspe_20221005.shp")

POWERPLANT_DIR = os.path.join(BASE_DIR, "ukraine_powerplant")
UK_NUC_SHP = os.path.join(POWERPLANT_DIR, "nuclear.shp")
UK_COAL_SHP = os.path.join(POWERPLANT_DIR, "coal.shp")

# CLIMATE
CLIMATE_DIR = os.path.join(BASE_DIR, "climate")
CLIMATE_FILE = glob.glob(os.path.join(CLIMATE_DIR, "era5", "*.grib"))[0]

# NO2
NO2_DIR = os.path.join(BASE_DIR, "no2")

SF_REALS_NO2_2019_FILES = glob.glob(os.path.join(NO2_DIR, "cams-reals", "2019", "*.nc"))

SF_FC_NO2_2020_2022_FILES = glob.glob(os.path.join(NO2_DIR, "cams-fc", "*.grib"))

CL_NO2_FILES = glob.glob(os.path.join(NO2_DIR, "UK_RUS", "*.tif"))

# POP
POP_FILE = os.path.join(BASE_DIR, "pop", "uk_pop_2020.tif")

# LAT LON DIR
ERA5_LAT_FILE = os.path.join(BASE_DIR, "interp_latlon", "era5_lat.npy")
ERA5_LON_FILE = os.path.join(BASE_DIR, "interp_latlon", "era5_lon.npy")
POP_LAT_FILE = os.path.join(BASE_DIR, "interp_latlon", "pop_lat.npy")
POP_LON_FILE = os.path.join(BASE_DIR, "interp_latlon", "pop_lon.npy")
CAMS_REALS_LAT_FILE = os.path.join(BASE_DIR, "interp_latlon", "cams_reals_lat.npy")
CAMS_REALS_LON_FILE = os.path.join(BASE_DIR, "interp_latlon", "cams_reals_lon.npy")

# PREPROCESS DIR

PREPROCESS_DIR = os.path.join(BASE_DIR, "preprocessed")

CAM_REALS_NO2_NC = os.path.join(PREPROCESS_DIR, "cams_reals_no2.nc")
CAM_FC_NO2_NC = os.path.join(PREPROCESS_DIR, "cams_fc_no2.nc")
ERA5_NC = os.path.join(PREPROCESS_DIR, "era5.nc")
S5P_NO2_NC = os.path.join(PREPROCESS_DIR, "s5p_no2.nc")
POP_NC = os.path.join(PREPROCESS_DIR, "pop.nc")

DE_WEATHER_MODEL = os.path.join("de_weather_model", "model.sav")
# %%
