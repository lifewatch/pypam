"""
Module: geolocation.py
Authors: Clea Parcerisas
Institution: VLIZ (Vlaams Institute voor de Zee)
"""

import datetime
import pathlib
import sqlite3

import contextily as ctx
import geopandas
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import shapely
from geopy import distance
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Apply the default theme
sns.set_theme()


class SurveyLocation:
    def __init__(self, geofile=None, **kwargs):
        """
        Location of all the survey points
        Parameters
        ----------
        geofile : str or Path
            Where the data is stored. can be a csv, gpx, pickle or sqlite3
        """
        if geofile is None:
            self.geotrackpoints = None
        else:
            if type(geofile) == str:
                geofile = pathlib.Path(geofile)
            extension = geofile.suffix
            if extension == '.gpx':
                geotrackpoints = self.read_gpx(geofile)
            elif extension == '.pkl':
                geotrackpoints = self.read_pickle(geofile, **kwargs)
            elif extension == '.csv':
                geotrackpoints = self.read_csv(geofile, **kwargs)
            elif extension == '.sqlite3':
                geotrackpoints = self.read_sqlite3(geofile, **kwargs)
            else:
                raise Exception('Extension %s is not implemented!' % extension)

            self.geotrackpoints = geotrackpoints
            self.geofile = geofile

    def read_gpx(self, geofile):
        """
        Read a GPX from GARMIN
        Parameters
        ----------
        geofile : str or Path
            GPX file
        Returns
        -------
        Geopandas DataFrame
        """
        geotrackpoints = geopandas.read_file(geofile, layer='track_points')
        geotrackpoints.drop_duplicates(subset='time', inplace=True)
        geotrackpoints = geotrackpoints.set_index(pd.to_datetime(geotrackpoints['time']))
        geotrackpoints = geotrackpoints.sort_index()
        geotrackpoints.dropna(subset=['geometry'], inplace=True)

        return geotrackpoints

    def read_pickle(self, geofile, datetime_col='datetime', crs='EPSG:4326'):
        """
        Read a pickle file
        Parameters
        ----------
        geofile : string or path
            Where the gps information is
        datetime_col : string
            Name of the column where the datetime is
        crs : string
            Projection of the gps information

        Returns
        -------
        A Geopandas DataFrame
        """
        df = pd.read_pickle(geofile)
        geotrackpoints = geopandas.GeoDataFrame(df, crs=crs)
        geotrackpoints[datetime_col] = pd.to_datetime(df[datetime_col])
        geotrackpoints.drop_duplicates(subset=datetime_col, inplace=True)
        geotrackpoints = geotrackpoints.set_index(datetime_col)
        geotrackpoints = geotrackpoints.sort_index()
        geotrackpoints.dropna(subset=['geometry'], inplace=True)

        return geotrackpoints

    def read_csv(self, geofile, datetime_col='datetime', lat_col='Lat', lon_col='Lon'):
        """
        Read a csv file
        Parameters
        ----------
        geofile : string or path
            Where the gps information is
        datetime_col : string
            Name of the column where the datetime is
        lat_col : string
            Column with the Latitude
        lon_col : string
            Column with the Longitude data

        Returns
        -------
        A GeoPandasDataFrame
        """
        df = pd.read_csv(geofile)
        geotrackpoints = self.convert_df(df, datetime_col, lat_col, lon_col)
        return geotrackpoints

    def read_sqlite3(self, geofile, table_name='gpsData', datetime_col='UTC', lat_col='Latitude',
                     lon_col='Longitude'):
        """
        Read a sqlite3 file with geolocation data
        Parameters
        ----------
        geofile : str or Path
            sqlite3 file
        Returns
        -------
        Geopandas DataFrame
        """
        conn = sqlite3.connect(geofile).cursor()
        query = conn.execute("SELECT * FROM %s" % table_name)
        cols = [column[0] for column in query.description]
        df = pd.DataFrame.from_records(data=query.fetchall(), columns=cols)
        geotrackpoints = self.convert_df(df, datetime_col, lat_col, lon_col)
        return geotrackpoints

    def convert_df(self, df, datetime_col, lat_col, lon_col):
        """
        Convert the DataFrame in a GeoPandas DataFrame
        Parameters
        ----------
        df : DataFrame
        datetime_col: str
            String where the time information is
        lat_col: str
            Column name where the latitude is
        lon_col: str
            Column name where the longitude is
        Returns
        -------
        The df turned in a geopandas dataframe
        """
        geotrackpoints = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df[lon_col], df[lat_col]),
                                                crs='EPSG:4326')
        geotrackpoints[datetime_col] = pd.to_datetime(df[datetime_col])
        geotrackpoints.drop_duplicates(subset=datetime_col, inplace=True)
        geotrackpoints = geotrackpoints.set_index(datetime_col)
        geotrackpoints = geotrackpoints.sort_index()
        geotrackpoints.dropna(subset=[lat_col], inplace=True)
        geotrackpoints.dropna(subset=[lon_col], inplace=True)

        return geotrackpoints

    def add_survey_location(self, df, time_tolerance='10min'):
        """
        Add the closest location of the GeoSurvey to each timestamp of the DataFrame.
        Returns a new GeoDataFrame with all the information of df but with added geometry

        Parameters
        ----------
        df : DataFrame
            DataFrame from an ASA output
        """
        self.geotrackpoints.index = self.geotrackpoints.index.tz_localize('UTC')
        if 'datetime' in df.columns:
            datetime_df = pd.DataFrame({'datetime': df.datetime})
        else:
            datetime_df = pd.DataFrame({'datetime': df.index})
        geo_df = pd.merge_asof(datetime_df, self.geotrackpoints, left_on="datetime", right_index=True,
                               tolerance=pd.Timedelta(time_tolerance))
        geo_df = geo_df.set_index('datetime')
        df['geom'] = geo_df.geometry
        df = geopandas.GeoDataFrame(df, geometry='geom', crs=self.geotrackpoints.crs.to_string())
        # Patch to solve the multiindex incompatibiolity with crs in geopandas
        df['geometry'] = df.geometry

        return df

    def plot_survey_color(self, column, df, units=None, map_file=None, save_path=None):
        """
        Add the closest location to each timestamp and color the points in a map

        Parameters
        ----------
        column : string
            Column of the df to plot
        units : string
            Units of the legend
        df : DataFrame or GeoDataFrame
            DataFrame from an ASA output or GeoDataFrame
        map_file : string or Path
            Map that will be used as a basemap
        save_path : str or PathLike or file-like object
        """
        if 'geometry' not in df.columns:
            df = self.add_survey_location(df)
        _, ax = plt.subplots(1, 1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        if df[column].dtype != float:
            df.plot(column=column, ax=ax, legend=True, alpha=0.5, categorical=True, cax=cax)
        else:
            df.plot(column=column, ax=ax, legend=True, alpha=0.5, cmap='YlOrRd', categorical=False,
                    cax=cax)
        if map_file is None:
            ctx.add_basemap(ax, crs=df.crs.to_string(), source=ctx.providers.Stamen.TonerLite,
                            reset_extent=False)
        else:
            ctx.add_basemap(ax, crs=df.crs.to_string(), source=map_file, reset_extent=False,
                            cmap='BrBG')
        ax.set_axis_off()
        ax.set_title('%s Distribution' % (column))
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()

    def add_distance_to(self, df, lat, lon, column='distance'):
        """
        Add the distances to a certain point.
        Returns GeoDataFrame with an added column with the distance to the point lat, lon

        Parameters
        ----------
        df : DataFrame or GeoDataFrame
            ASA output or GeoDataFrame
        lat : float
            Latitude of the point
        lon : float
            Longitude of the point

        """
        if "geometry" not in df.columns:
            df = self.add_survey_location(df)
        df[column] = df['geometry'].apply(distance_m, lat=lat, lon=lon)

        return df

    def add_distance_to_coast(self, df, coastfile, column='coast_dist'):
        """
        Add the minimum distance to the coast.
        Returns GeoDataFrame with an added column with the distance to the coast

        Parameters
        ----------
        df : DataFrame or GeoDataFrame
            ASA output or GeoDataFrame
        coastfile : str or Path
            File with the points of the coast
        """
        if "geometry" not in df.columns:
            df = self.add_survey_location(df)
        coastline = geopandas.read_file(coastfile).loc[0].geometry.coords
        coast_df = geopandas.GeoDataFrame(geometry=[shapely.geometry.Point(xy)
                                                    for xy in coastline])
        df[column] = df['geometry'].apply(min_distance_m, geodf=coast_df)

        return df


def distance_m(coords, lat, lon):
    """
    Return the distance in meters between the coordinates and the point (lat, lon)
    """
    d = distance.distance((lat, lon), (coords.y, coords.x)).m

    return d


def min_distance_m(coords, geodf):
    """
    Return the minimum distance in meters between the coords and the points of the geodf
    """
    distances = geodf['geometry'].apply(distance_m, args=(coords.y, coords.x))

    return distances.min()
