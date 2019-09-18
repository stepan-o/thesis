import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from time import time


def df_from_csv(csv_path, parse_dates=None, low_memory=False):
    # load DataFrame with Teranet records
    t = time()
    df = pd.read_csv(csv_path, parse_dates=parse_dates, low_memory=low_memory)
    elapsed = time() - t
    print("----- DataFrame loaded"
          "\nin {0:.2f} seconds".format(elapsed) +
          "\nwith {0:,} rows\nand {1:,} columns"
          .format(df.shape[0], df.shape[1]) +
          "\n-- Column names:\n", df.columns)
    return df


def add_geom(df, crs, crs_name=True, x_col='x', y_col='y'):
    # combine values in columns 'x' and 'y' into a POINT geometry object
    t = time()
    geometry = [Point(xy) for xy in zip(df[x_col], df[y_col])]
    # generate a new GeoDataFrame by adding point geometry to data frame 'teranet_sales_data'
    sgdf = gpd.GeoDataFrame(df, geometry=geometry)
    elapsed = time() - t
    print("\n----- Geometry generated from 'X' and 'Y' pairs, GeoDataFrame created!"
          "\nin {0:.2f} seconds ({1:.2f} minutes)".format(elapsed, elapsed / 60) +
          "\nwith {0:,} rows\nand {1:,} columns"
          .format(sgdf.shape[0], sgdf.shape[1]) +
          "\n-- Column names:\n", sgdf.columns)

    # add CRS for WGS84 (lat-long) to GeoDataFrame with Teranet records
    sgdf.crs = crs
    if crs_name:
        print("\n----- CRS dictionary for {0} added to GeoDataFrame!".format(crs['init']))

    return sgdf
