"""
Module: geolocation.py
Authors: Clea Parcerisas
Institution: VLIZ (Vlaams Institute voor de Zee)
"""

import math
import shapely
import datetime
import geopandas
import pandas as pd
import contextily as ctx
import matplotlib.pyplot as plt



class SurveyLocation:
    def __init__(self, geofile):
        """
        Location of all the survey points

        Parameters
        ----------
        geofile : string or Path
            Can be a gpx file or a pickle file with a GeoDataFrame
        """
        extension = geofile.split('.')[-1]
        if extension == 'gpx':
            geotrackpoints = geopandas.read_file(geofile, layer='track_points')
            geotrackpoints.drop_duplicates(subset='time', inplace=True)
            self.geotrackpoints = geotrackpoints.set_index(pd.to_datetime(geotrackpoints['time']))
        elif extension == 'pkl':
            self.geotrackpoints = pd.read_pickle(geofile)
        else:
            raise Exception('The extension %s is not implemented' % (extension))
    

    def add_survey_location(self, df):
        """
        Add the closest location of the GeoSurvey to each timestamp of the DataFrame.
        Returns a new GeoDataFrame with all the information of df but with added geometry

        Parameters
        ----------
        df : DataFrame
            DataFrame from an ASA output
        """
        # df['geo_time'] = 0
        for i in df.index:
            if 'datetime' in df.columns:
                t = df.iloc[i].datetime
            else:
                t = i
            idx = self.geotrackpoints.index.get_loc(t, method='nearest')
            df.loc[i, 'geo_time'] = self.geotrackpoints.index[idx]
        
        geo_df = self.geotrackpoints.loc[df['geo_time']]
        geo_df['geo_time'] = df['geo_time'].values
        geo_df = geo_df.merge(df, on='geo_time')

        return geo_df
    

    def plot_survey_color(self, column, units, df, map_file=None):
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
        fig, ax = plt.subplots(1, 1)
        if df[column].dtype != float:
            ax = df.plot(column=column, ax=ax, legend=True, alpha=0.5, categorical=True) 
        else: 
            ax = df.plot(column=column, ax=ax, legend=True, alpha=0.5, cmap='YlOrRd', categorical=False) 
        if map_file is None:
            ctx.add_basemap(ax, crs=df.crs.to_string(), source=ctx.providers.Stamen.TonerLite, reset_extent=False)
        else:
            ctx.add_basemap(ax, crs=df.crs.to_string(), source=map_file, reset_extent=False, cmap='BrBG')
        ax.set_axis_off()
        ax.set_title('%s Distribution' % (column))
        plt.show()
    
    
    def add_distance_to(self, df, lat, lon):
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
        point = shapely.geometry.Point(lat, lon)
        df['distance'] = df['geometry'].distance(point)

        return df