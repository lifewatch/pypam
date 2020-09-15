"""
Module: geolocation.py
Authors: Clea Parcerisas
Institution: VLIZ (Vlaams Institute voor de Zee)
"""

import math
from geopy import distance
import pathlib
import shapely
import datetime
import geopandas
import pandas as pd
import contextily as ctx
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from shapely.geometry import geo



class SurveyLocation:
    def __init__(self, geofile=None, **kwargs):
        """
        Location of all the survey points
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
            else:
                raise Exception('Extension %s is not implemented!' % (extension))
            
            self.geotrackpoints = geotrackpoints
            self.geofile = geofile
        


    def read_gpx(self, geofile):
        """
        Read a GPX from GARMIN
        """
        geotrackpoints = geopandas.read_file(geofile, layer='track_points')
        geotrackpoints.drop_duplicates(subset='time', inplace=True)
        geotrackpoints = geotrackpoints.set_index(pd.to_datetime(geotrackpoints['time']))
        geotrackpoints = geotrackpoints.sort_index()
        geotrackpoints.dropna(subset=['geometry'], inplace=True)

        return geotrackpoints

    
    def read_pickle(self, geofile, datetime_col='datetime', crs='EPSG:4326'):
        df = pd.read_pickle(geofile)
        geotrackpoints = geopandas.GeoDataFrame(df, crs=crs)
        geotrackpoints[datetime_col] = pd.to_datetime(df[datetime_col])
        geotrackpoints = geotrackpoints.set_index(datetime_col)
        geotrackpoints = geotrackpoints.sort_index()
        geotrackpoints.dropna(subset=['geometry'], inplace=True)

        return geotrackpoints


    def read_csv(self, geofile, datetime_col='datetime', lat_col='Lat', lon_col='Lon'):
        df = pd.read_csv(geofile)
        geotrackpoints = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df[lon_col], df[lat_col]))
        geotrackpoints[datetime_col] = pd.to_datetime(df[datetime_col])
        geotrackpoints = geotrackpoints.set_index(datetime_col)
        geotrackpoints = geotrackpoints.sort_index()
        geotrackpoints.dropna(subset=[lat_col], inplace=True)
        geotrackpoints.dropna(subset=[lon_col], inplace=True)

        return geotrackpoints
       

    def add_survey_location(self, df):
        """
        Add the closest location of the GeoSurvey to each timestamp of the DataFrame.
        Returns a new GeoDataFrame with all the information of df but with added geometry

        Parameters
        ----------
        df : DataFrame
            DataFrame from an ASA output
        """
        for i in df.index:
            if 'datetime' in df.columns:
                t = df.iloc[i].datetime
            else:
                t = i
            idx = self.geotrackpoints.index.get_loc(t, method='nearest')
            df.loc[i, 'geo_time'] = self.geotrackpoints.index[idx]
        
        good_points_mask = abs(df.geo_time - df.datetime) < datetime.timedelta(seconds=600)
        if good_points_mask.sum() < len(df):
            print('This file %s is not corresponding with the timestamps!' % (self.geofile))
        geo_df = self.geotrackpoints.reindex(df.geo_time)
        # geo_df['geo_time'] = df.geo_time.values
        geo_df = geo_df.merge(df, on='geo_time')

        return geo_df
    

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
        """
        if 'geometry' not in df.columns: 
            df = self.add_survey_location(df)
        _, ax = plt.subplots(1, 1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        if df[column].dtype != float:
            im = df.plot(column=column, ax=ax, legend=True, alpha=0.5, categorical=True, cax=cax) 
        else: 
            im = df.plot(column=column, ax=ax, legend=True, alpha=0.5, cmap='YlOrRd', categorical=False, cax=cax) 
        if map_file is None:
            ctx.add_basemap(ax, crs=df.crs.to_string(), source=ctx.providers.Stamen.TonerLite, reset_extent=False)
        else:
            ctx.add_basemap(ax, crs=df.crs.to_string(), source=map_file, reset_extent=False, cmap='BrBG')
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
        coastline = geopandas.read_file(coastfile).loc[0].geometry.coords
        coast_df = geopandas.GeoDataFrame(geometry=[shapely.geometry.Point(xy) for xy in coastline])
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