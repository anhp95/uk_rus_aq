#%%

import ee

ee.Authenticate()
ee.Initialize()

#%%

import pandas as pd
from calendar import monthrange

UK_BOUND = ee.Geometry.Polygon(
    [
        22.136912,
        52.379291,
        40.227883,
        52.379291,
        40.227883,
        44.386411,
        22.136912,
        44.386411,
    ]
)
SCALE = 1113.2
FOLDER = "UK_RUS"

YEARS = [2019, 2020, 2021, 2022]
MONTHS = [2, 3, 4, 5, 6, 7]

MONTHS_BF_WAR = [1, 2]

NO2_BAND = "NO2_column_number_density"


def export2drive(img, id):
    print(id)
    task = ee.batch.Export.image.toDrive(
        **{
            "image": img,
            "description": id,
            "folder": FOLDER,
            "scale": SCALE,
            "region": UK_BOUND,
        }
    )
    task.start()


def to_julian_date(year, month, day):
    ts = pd.Timestamp(year=year, month=month, day=day)
    return ts.to_julian_date()


def download_no2_war():
    for y in YEARS:
        for m in MONTHS:
            d = 1 if m != 2 else 24
            _, ed = monthrange(y, m)
            for dix in range(d, ed + 1):
                jd = to_julian_date(y, m, dix)
                print(jd)
                img = (
                    ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_NO2")
                    .filter(ee.Filter.eq("TIME_REFERENCE_JULIAN_DAY", jd))
                    .mosaic()
                    .select(NO2_BAND)
                    .clip(UK_BOUND)
                )
                export2drive(img, str(jd))


def download_no2_bf_war():

    for y in YEARS:
        for m in MONTHS_BF_WAR:
            ed = monthrange(y, m)[1] if m == 1 else 23
            for dix in range(1, ed + 1):
                jd = to_julian_date(y, m, dix)
                print(jd)
                img = (
                    ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_NO2")
                    .filter(ee.Filter.eq("TIME_REFERENCE_JULIAN_DAY", jd))
                    .mosaic()
                    .select(NO2_BAND)
                    .clip(UK_BOUND)
                )
                export2drive(img, str(jd))


def download_pop():
    pop = (
        ee.ImageCollection("WorldPop/GP/100m/pop")
        .filter(ee.Filter.eq("country", "UKR"))
        .filter(ee.Filter.eq("year", 2020))
        .mosaic()
    )
    export2drive(dataset, "uk_pop_2020")


# %%
