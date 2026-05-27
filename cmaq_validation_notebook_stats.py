#!/usr/bin/env python
# coding: utf-8

# ## PM2.5

# In[2]:


import pandas as pd
import numpy as np

# Load the CSV file into a DataFrame
file_path = 'aqs_data_2023/PM25_TOT_WRFCMAQ-BASE_2023_10_EPA_CMAQ_Combine_2020_RWC.csv'
df = pd.read_csv(file_path)

# # Remove any rows with missing data in 'Sample Measurement' or 'CMAQ'
# before = len(df.index)
# df = df.dropna(subset=['Sample Measurement', 'CMAQ'])
# print(round((before - len(df.index))/ before * 100, 2), "% lost to na")

percent_under_detection_limit = ((df.loc[df['Sample Measurement'] <= 2]).shape[0])/ df.shape[0] * 100 
print(f"{round(percent_under_detection_limit,2)}% of AQS measurements under the detection limit of 2ug/m^3")


df_detection_limit = df.loc[df['Sample Measurement'] >= 2]


# In[ ]:


## Time Series Plots

import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def site_stats_table(df):
    stats_list = []

    for site in df['Site Num'].unique():
        site_df = df[df['Site Num'] == site]
        obs = site_df['Sample Measurement'].values
        mod = site_df['CMAQ'].values
        
        if len(obs) == 0:
            continue
        
        # Basic stats
        μd = np.mean(obs)
        μp = np.mean(mod)
        nmb = np.sum(mod - obs) / np.sum(obs) * 100
        nme = np.sum(np.abs(mod - obs)) / np.sum(obs) * 100
        r_value = pearsonr(obs, mod)[0] if len(obs) > 1 else np.nan
        
        lat = site_df['Latitude'].iloc[0]
        lon = site_df['Longitude'].iloc[0]
        n_points = len(obs)
        
        stats_list.append({
            'Site Num': site,
            'Latitude': lat,
            'Longitude': lon,
            'n_points': n_points,
            'μd': μd,
            'μp': μp,
            'NMB': nmb,
            'NME': nme,
            'r': r_value
        })
    
    stats_df = pd.DataFrame(stats_list)
    return stats_df

# Usage
stats_table = site_stats_table(df)
stats_table


# In[34]:



# %%
def pearsonr(x, y):
    """
    Compute the Pearson correlation coefficient between two arrays.

    Parameters:
    x (array-like): First dataset.
    y (array-like): Second dataset.

    Returns:
    r (float): Pearson correlation coefficient.
    p_value (float): Placeholder for compatibility (always None in this implementation).
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if x.shape[0] != y.shape[0]:
        raise ValueError("Input arrays must have the same length.")

    # Compute means
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # Compute Pearson correlation
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sqrt(np.sum((x - x_mean) ** 2) * np.sum((y - y_mean) ** 2))
    r = numerator / denominator if denominator != 0 else 0  # Avoid division by zero

    return r, None  # Returning None as a placeholder for p-value

# %%
def do_overall_stats(df, daily):
    df = df.copy(deep = True)
    if daily:
        grouped = df.groupby(['Latitude', 'Longitude'])
        groups = []
        for (lat, lon), group in grouped:
            group = group.set_index('dt')
            group.index = pd.to_datetime(group.index)

            numeric_cols = group.select_dtypes(include='number').columns
            non_numeric_cols = group.select_dtypes(exclude='number').columns
            numeric_resampled = group[['Sample Measurement', 'CMAQ']].resample('D').mean().dropna()
            non_numeric_resampled = group[non_numeric_cols].resample('D').last()

            group = pd.concat([numeric_resampled, non_numeric_resampled], axis=1).reset_index().dropna(subset = ['Sample Measurement', 'CMAQ'])
            group['Latitude'] = lat
            group['Longitude'] = lon
            groups.append(group)

        df = pd.concat(groups)
        df = pd.concat(groups).reset_index()
    
    df_all = df[['CMAQ', 'Sample Measurement']].dropna()
    observed = df_all['Sample Measurement']
    predicted = df_all['CMAQ']

    # Calculate mean observed (μd) and mean predicted (μp)
    μd = observed.mean()
    μp = predicted.mean()

    # Calculate NMB
    nmb = ((predicted - observed).sum() / observed.sum()) * 100

    # Calculate NME
    nme = (abs(predicted - observed).sum() / observed.sum()) * 100

    # Calculate Pearson correlation coefficient (r)
    if len(observed) > 1:  # Ensure there is enough data for correlation
        r_value, _ = pearsonr(observed, predicted)
    else:
        r_value = np.nan  # Not enough data to compute correlation

    # Append the calculated metrics to the DataFrame
    return pd.DataFrame({
        'μd': [μd],
        'μp': [μp],
        'NMB': [nmb],
        'NME': [nme],
        'r': [r_value]
    })

do_overall_stats(df, daily = True)


# In[65]:


import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import geopandas as gpd
import pandas as pd
#import contextily as ctx

def plot_chem_spatial(metrics_df_2020, marker_size = 200, chosen_stat="NMB", cmap="RdBu", difference=True, pollutant="PM25", stat_desc="NMB (%)", alpha=0.075):
    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(metrics_df_2020, geometry=gpd.points_from_xy(metrics_df_2020.Longitude, metrics_df_2020.Latitude), crs="EPSG:4326")
    
    # Convert to Web Mercator for contextily basemap
    gdf_web = gdf.to_crs(epsg=3857)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(20, 12))
    
    # Normalize color scale
    if difference:
        vMAX = max(abs(gdf_web[chosen_stat].quantile(alpha)), abs(gdf_web[chosen_stat].quantile(1-alpha)))
        norm = Normalize(vmin=-vMAX, vmax=vMAX)
    else:
        vmin = gdf_web[chosen_stat].quantile(alpha)
        vmax = gdf_web[chosen_stat].quantile(1-alpha)
        norm = Normalize(vmin=vmin, vmax=vmax)
    
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    # Scatter plot
    gdf_web.plot(ax=ax, column=chosen_stat, cmap=cmap, norm=norm, edgecolor='k', linewidth=0.5, alpha=1, markersize=marker_size)
    
    # Add contextily basemap
    #ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=7)
    
    # Add colorbar
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', shrink=0.8)
    cbar.set_label(stat_desc, fontsize=18)
    cbar.ax.tick_params(labelsize=16)
    
    # Add title|
    plt.title(f"Spatial Distribution of {stat_desc.split('(')[0]} for {pollutant}", fontsize=22)
    
    # Remove axes for cleaner map
    ax.set_axis_off()
    
    plt.show()

# Example usage
plot_chem_spatial(stats_table, chosen_stat="NMB", cmap="RdBu", difference=True, stat_desc="NMB (%)", pollutant="PM2.5", alpha=0.075)


# In[55]:


plot_chem_spatial(stats_table, chosen_stat="r", cmap="viridis", difference=False, stat_desc="r", pollutant="PM2.5", alpha=0.075)


# ## NO2

# In[57]:


import pandas as pd
import numpy as np

# Load the CSV file into a DataFrame
file_path = 'aqs_data_2023/NO2_WRFCMAQ-BASE_2023_10_EPA_CMAQ_Combine_2020_RWC.csv'
df = pd.read_csv(file_path)

# # Remove any rows with missing data in 'Sample Measurement' or 'CMAQ'
# before = len(df.index)
# df = df.dropna(subset=['Sample Measurement', 'CMAQ'])
# print(round((before - len(df.index))/ before * 100, 2), "% lost to na")

# percent_under_detection_limit = ((df.loc[df['Sample Measurement'] <= 2]).shape[0])/ df.shape[0] * 100 
# print(f"{round(percent_under_detection_limit,2)}% of AQS measurements under the detection limit of 2ug/m^3")


# df_detection_limit = df.loc[df['Sample Measurement'] >= 2]


# In[58]:


## Time Series Plots

import matplotlib.pyplot as plt

# Ensure dt is datetime type
df['dt'] = pd.to_datetime(df['dt'])

# Convert Site Num to int if needed
df['Site Num'] = df['Site Num'].dropna().astype(int)

# Loop through unique sites
#for site in df['Site Num'].unique():
    #site_df = df[df['Site Num'] == site].sort_values('dt')
    
    #plt.figure(figsize=(12, 5))
    #plt.plot(site_df['dt'], site_df['Sample Measurement'], label='Observed')
    #plt.plot(site_df['dt'], site_df['CMAQ'], label='CMAQ', alpha=0.7)
    
    #plt.title(f'Site {site} Time Series')
    #plt.xlabel('Date')
    #plt.ylabel('PM2.5 / Measurement')
    #plt.legend()
    #plt.tight_layout()
    #plt.show()


# In[59]:


import pandas as pd
import numpy as np
from scipy.stats import pearsonr

def site_stats_table(df):
    stats_list = []

    for site in df['Site Num'].unique():
        site_df = df[df['Site Num'] == site]
        obs = site_df['Sample Measurement'].values
        mod = site_df['CMAQ'].values
        
        if len(obs) == 0:
            continue
        
        # Basic stats
        μd = np.mean(obs)
        μp = np.mean(mod)
        nmb = np.sum(mod - obs) / np.sum(obs) * 100
        nme = np.sum(np.abs(mod - obs)) / np.sum(obs) * 100
        r_value = pearsonr(obs, mod)[0] if len(obs) > 1 else np.nan
        
        lat = site_df['Latitude'].iloc[0]
        lon = site_df['Longitude'].iloc[0]
        n_points = len(obs)
        
        stats_list.append({
            'Site Num': site,
            'Latitude': lat,
            'Longitude': lon,
            'n_points': n_points,
            'μd': μd,
            'μp': μp,
            'NMB': nmb,
            'NME': nme,
            'r': r_value
        })
    
    stats_df = pd.DataFrame(stats_list)
    return stats_df

# Usage
stats_table_NO2 = site_stats_table(df)
stats_table_NO2


# In[60]:



# %%
def pearsonr(x, y):
    """
    Compute the Pearson correlation coefficient between two arrays.

    Parameters:
    x (array-like): First dataset.
    y (array-like): Second dataset.

    Returns:
    r (float): Pearson correlation coefficient.
    p_value (float): Placeholder for compatibility (always None in this implementation).
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if x.shape[0] != y.shape[0]:
        raise ValueError("Input arrays must have the same length.")

    # Compute means
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # Compute Pearson correlation
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sqrt(np.sum((x - x_mean) ** 2) * np.sum((y - y_mean) ** 2))
    r = numerator / denominator if denominator != 0 else 0  # Avoid division by zero

    return r, None  # Returning None as a placeholder for p-value

# %%
def do_overall_stats(df, daily):
    df = df.copy(deep = True)
    if daily:
        grouped = df.groupby(['Latitude', 'Longitude'])
        groups = []
        for (lat, lon), group in grouped:
            group = group.set_index('dt')
            group.index = pd.to_datetime(group.index)

            numeric_cols = group.select_dtypes(include='number').columns
            non_numeric_cols = group.select_dtypes(exclude='number').columns
            numeric_resampled = group[['Sample Measurement', 'CMAQ']].resample('D').mean().dropna()
            non_numeric_resampled = group[non_numeric_cols].resample('D').last()

            group = pd.concat([numeric_resampled, non_numeric_resampled], axis=1).reset_index().dropna(subset = ['Sample Measurement', 'CMAQ'])
            group['Latitude'] = lat
            group['Longitude'] = lon
            groups.append(group)

        df = pd.concat(groups)
        df = pd.concat(groups).reset_index()
    
    df_all = df[['CMAQ', 'Sample Measurement']].dropna()
    observed = df_all['Sample Measurement']
    predicted = df_all['CMAQ']

    # Calculate mean observed (μd) and mean predicted (μp)
    μd = observed.mean()
    μp = predicted.mean()

    # Calculate NMB
    nmb = ((predicted - observed).sum() / observed.sum()) * 100

    # Calculate NME
    nme = (abs(predicted - observed).sum() / observed.sum()) * 100

    # Calculate Pearson correlation coefficient (r)
    if len(observed) > 1:  # Ensure there is enough data for correlation
        r_value, _ = pearsonr(observed, predicted)
    else:
        r_value = np.nan  # Not enough data to compute correlation

    # Append the calculated metrics to the DataFrame
    return pd.DataFrame({
        'μd': [μd],
        'μp': [μp],
        'NMB': [nmb],
        'NME': [nme],
        'r': [r_value]
    })

do_overall_stats(df, daily = True)


# In[66]:


plot_chem_spatial(stats_table_NO2, marker_size = 300, chosen_stat="NMB", cmap="RdBu", difference=True, stat_desc="NMB (%)", pollutant="NO2", alpha=0.075)


# In[68]:


plot_chem_spatial(stats_table_NO2, marker_size = 300, chosen_stat="r", cmap="viridis", difference=False, stat_desc="r", pollutant="NO2", alpha=0.075)


# In[ ]:





# ## O3

# In[72]:


import pandas as pd
import numpy as np

# Load the CSV file into a DataFrame
file_path = 'aqs_data_2023/O3_WRFCMAQ-BASE_2023_10_EPA_CMAQ_Combine_2020_RWC.csv'
df = pd.read_csv(file_path)
df['Sample Measurement'] *= 1000

# # Remove any rows with missing data in 'Sample Measurement' or 'CMAQ'
# before = len(df.index)
# df = df.dropna(subset=['Sample Measurement', 'CMAQ'])
# print(round((before - len(df.index))/ before * 100, 2), "% lost to na")

# percent_under_detection_limit = ((df.loc[df['Sample Measurement'] <= 2]).shape[0])/ df.shape[0] * 100 
# print(f"{round(percent_under_detection_limit,2)}% of AQS measurements under the detection limit of 2ug/m^3")


# df_detection_limit = df.loc[df['Sample Measurement'] >= 2]


# In[73]:


## Time Series Plots

import matplotlib.pyplot as plt

# Ensure dt is datetime type
df['dt'] = pd.to_datetime(df['dt'])

# Convert Site Num to int if needed
df['Site Num'] = df['Site Num'].dropna().astype(int)

# Loop through unique sites
#for site in df['Site Num'].unique():
    #site_df = df[df['Site Num'] == site].sort_values('dt')
    
    #plt.figure(figsize=(12, 5))
    #plt.plot(site_df['dt'], site_df['Sample Measurement'], label='Observed')
    #plt.plot(site_df['dt'], site_df['CMAQ'], label='CMAQ', alpha=0.7)
    
    #plt.title(f'Site {site} Time Series')
    #plt.xlabel('Date')
    #plt.ylabel('PM2.5 / Measurement')
    #plt.legend()
    #plt.tight_layout()
    #plt.show()


# In[74]:


import pandas as pd
import numpy as np
from scipy.stats import pearsonr

def site_stats_table(df):
    stats_list = []

    for site in df['Site Num'].unique():
        site_df = df[df['Site Num'] == site]
        obs = site_df['Sample Measurement'].values
        mod = site_df['CMAQ'].values
        
        if len(obs) == 0:
            continue
        
        # Basic stats
        μd = np.mean(obs)
        μp = np.mean(mod)
        nmb = np.sum(mod - obs) / np.sum(obs) * 100
        nme = np.sum(np.abs(mod - obs)) / np.sum(obs) * 100
        r_value = pearsonr(obs, mod)[0] if len(obs) > 1 else np.nan
        
        lat = site_df['Latitude'].iloc[0]
        lon = site_df['Longitude'].iloc[0]
        n_points = len(obs)
        
        stats_list.append({
            'Site Num': site,
            'Latitude': lat,
            'Longitude': lon,
            'n_points': n_points,
            'μd': μd,
            'μp': μp,
            'NMB': nmb,
            'NME': nme,
            'r': r_value
        })
    
    stats_df = pd.DataFrame(stats_list)
    return stats_df

# Usage
stats_table_O3 = site_stats_table(df)
stats_table_O3


# In[75]:



# %%
def pearsonr(x, y):
    """
    Compute the Pearson correlation coefficient between two arrays.

    Parameters:
    x (array-like): First dataset.
    y (array-like): Second dataset.

    Returns:
    r (float): Pearson correlation coefficient.
    p_value (float): Placeholder for compatibility (always None in this implementation).
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if x.shape[0] != y.shape[0]:
        raise ValueError("Input arrays must have the same length.")

    # Compute means
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # Compute Pearson correlation
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sqrt(np.sum((x - x_mean) ** 2) * np.sum((y - y_mean) ** 2))
    r = numerator / denominator if denominator != 0 else 0  # Avoid division by zero

    return r, None  # Returning None as a placeholder for p-value

# %%
def do_overall_stats(df, daily):
    df = df.copy(deep = True)
    if daily:
        grouped = df.groupby(['Latitude', 'Longitude'])
        groups = []
        for (lat, lon), group in grouped:
            group = group.set_index('dt')
            group.index = pd.to_datetime(group.index)

            numeric_cols = group.select_dtypes(include='number').columns
            non_numeric_cols = group.select_dtypes(exclude='number').columns
            numeric_resampled = group[['Sample Measurement', 'CMAQ']].resample('D').mean().dropna()
            non_numeric_resampled = group[non_numeric_cols].resample('D').last()

            group = pd.concat([numeric_resampled, non_numeric_resampled], axis=1).reset_index().dropna(subset = ['Sample Measurement', 'CMAQ'])
            group['Latitude'] = lat
            group['Longitude'] = lon
            groups.append(group)

        df = pd.concat(groups)
        df = pd.concat(groups).reset_index()
    
    df_all = df[['CMAQ', 'Sample Measurement']].dropna()
    observed = df_all['Sample Measurement']
    predicted = df_all['CMAQ']

    # Calculate mean observed (μd) and mean predicted (μp)
    μd = observed.mean()
    μp = predicted.mean()

    # Calculate NMB
    nmb = ((predicted - observed).sum() / observed.sum()) * 100

    # Calculate NME
    nme = (abs(predicted - observed).sum() / observed.sum()) * 100

    # Calculate Pearson correlation coefficient (r)
    if len(observed) > 1:  # Ensure there is enough data for correlation
        r_value, _ = pearsonr(observed, predicted)
    else:
        r_value = np.nan  # Not enough data to compute correlation

    # Append the calculated metrics to the DataFrame
    return pd.DataFrame({
        'μd': [μd],
        'μp': [μp],
        'NMB': [nmb],
        'NME': [nme],
        'r': [r_value]
    })

do_overall_stats(df, daily = True)


# In[76]:


plot_chem_spatial(stats_table_O3, marker_size = 300, chosen_stat="NMB", cmap="RdBu", difference=True, stat_desc="NMB (%)", pollutant="O3", alpha=0.075)


# In[77]:


plot_chem_spatial(stats_table_O3, marker_size = 300, chosen_stat="r", cmap="viridis", difference=False, stat_desc="r", pollutant="O3", alpha=0.075)


# In[ ]:




