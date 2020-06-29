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
        `geofile` can be a gpx file or a pickle file with a geopandas df
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
        Add the closest location for each timestamp
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
        Add the distances to a certain point
        """
        if "geometry" not in df.columns: 
            df = self.add_survey_location(df)
        point = shapely.geometry.Point(lat, lon)
        df['distance'] = df['geometry'].distance(point)

        return df