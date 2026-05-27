# Check wrf-cmaq output
# Need EPA annual AQS data
# Will append the corresponding CMAQ value to the file

import pandas as pd
import numpy as np
from netCDF4 import Dataset
from datetime import timedelta, date,datetime;

domain='WRFCMAQ-BASE'
time='hourly'
year='2023'
month='04'
#epa_code=['42401','42602','44201','42101','88101']; var=['SO2','NO2','O3','CO','PM25_TOT']
stdate,enddt = 1,30
yr = 2023
mo = 4
start_dt = datetime(yr, mo, stdate)
end_dt = datetime(yr,mo,enddt,23)
#fstart='COMBINE_ACONC_BASE_all_layer_20230708_D02_'
fstart='COMBINE_ACONC_BASE_202304_D02_'

#input and output dirs
#dir_cmaq ='/projects/b1045/rcs_support/wrf-cmaq/WRF_DIR/work_dir/WRFv451_CMAQv55/CCTM/scripts/WRFCMAQ-BASE_all_layer_20230708_D02-sw_feedback/CMAQ_output_202307_combined_D02/'
#dir_cmaq ='/projects/b1045/rcs_support/wrf-cmaq/WRF_DIR/work_dir/WRFv451_CMAQv55/CCTM/scripts/WRFCMAQ-BASE_202310_D02-sw_feedback/CMAQ_output_202310_combined_D02/'
dir_cmaq = '/projects/b1045/rcs_support/wrf-cmaq/WRF_DIR/work_dir/WRFv451_CMAQv55/CCTM/scripts/WRFCMAQ-BASE_202304_D02-sw_feedback/CMAQ_output_202304_combined_D02/'
dir_epa='aqs_data_2023/'

#gridfile

grid = 'lat_lon_chicago_d02.nc'
#grid="/projects/b1045/wrf-cmaq/output/CONUS4K/output_CONUS4K_d02_4km_sf_rrtmg_10_10_1_v3852/lat_lon_CONUS4K_d02.nc"

# EPA codes for the diff species to make the file names of AQS fies
epa_code=['42401','42602','44201','42101','88101','81102']; var=['SO2','NO2','O3','CO','PM25_TOT','PM10'] #numerical identi vars
#epa_code=['88101']; var=['PM25_TOT'] #numerical identi vars

epa_files = [dir_epa+'%s_%s_%s.csv'%(time,epa_code[i],year,) for i in range(len(epa_code))]

#la,lo='lat','lon'
la,lo='LAT','LON'
cmaq_lon,cmaq_lat=np.array(Dataset(grid)[lo][0][0]),np.array(Dataset(grid)[la][0][0])
#cmaq_lon,cmaq_lat=np.array(Dataset(grid).variables[lo]),np.array(Dataset(grid).variables[la])
llat,ulat,llon,ulon=cmaq_lat.min(), cmaq_lat.max(), cmaq_lon.min(), cmaq_lon.max()

def find_index(stn_lon, stn_lat, wrf_lon, wrf_lat):
    """
    Find the nearest grid points in a gridded dataset (e.g., CMAQ or WRF) for a given set of station coordinates.

    This function identifies the indices of the closest longitude and latitude points in the `wrf_lon` and `wrf_lat`
    arrays, which represent the grid points of a model dataset (like CMAQ or WRF), based on the station's 
    latitude and longitude coordinates provided in `stn_lon` and `stn_lat`.

    Parameters:
    ----------
    stn_lon : list or array
        Longitude(s) of the station(s) for which we need to find the closest grid points.
        
    stn_lat : list or array
        Latitude(s) of the station(s) for which we need to find the closest grid points.
        
    wrf_lon : array-like
        Longitudes of the model's grid (e.g., WRF/CMAQ grid).
        
    wrf_lat : array-like
        Latitudes of the model's grid (e.g., WRF/CMAQ grid).

    Returns:
    -------
    xx : list
        List of x-coordinates (longitude indices) in the model grid that are closest to the station's longitude.
    
    yy : list
        List of y-coordinates (latitude indices) in the model grid that are closest to the station's latitude.
    
    Notes:
    -----
    - The function calculates the absolute difference between the station's coordinates and the model grid coordinates 
      to find the minimum (nearest) distance.
    - It uses the `np.maximum` function to account for the larger deviation between latitude or longitude.
    - It outputs the index of the nearest grid point for each station.

    Example:
    --------
    # Given station coordinates (lon, lat) and model grid (wrf_lon, wrf_lat)
    stn_lon = [-75.0]
    stn_lat = [40.0]
    wrf_lon = model_lon_array  # CMAQ/WRF longitudes
    wrf_lat = model_lat_array  # CMAQ/WRF latitudes
    
    # Find the nearest grid point for the given station
    x_index, y_index = find_index(stn_lon, stn_lat, wrf_lon, wrf_lat)
    """
    
    # Initialize lists to store the grid indices for each station
    xx = []
    yy = []

    # Loop over each station's latitude and longitude
    for i in range(len(stn_lat)):
        #print(i)
        # Calculate the absolute differences between the station's coordinates and all grid points
        abslat = np.abs(wrf_lat - stn_lat[i])  # Absolute latitude difference
        abslon = np.abs(wrf_lon - stn_lon[i])  # Absolute longitude difference
        
        # Take the maximum of the longitude and latitude differences to handle irregular grids
        c = np.maximum(abslon, abslat)
        #print(c)  # Debugging: Print the difference array

        # Find the indices of the minimum difference (i.e., the nearest grid point)
        #print(np.where(c == np.min(c)))  # Debugging: Print the index of the nearest point
        args = np.where(c == np.min(c))  # Get the x (lon) and y (lat) indices
        #print(args)
        
        # Append the indices of the nearest grid point to the lists
        xx.append(args[-2])
        yy.append(args[-1])

    # Since np.where returns arrays, extract the first element from each index
    xx = [xx[i][0] for i in range(len(xx))]
    yy = [yy[i][0] for i in range(len(yy))]

    # Return the lists of x and y indices for the nearest grid points
    return xx, yy

#find_index([epa_lon[i]], [epa_lat[i]], cmaq_lon, cmaq_lat)

def make_cmaq_fnames(fstart,year,month,stdate,enddt):
        fnames = [fstart+year+month+str(i).zfill(2)+'.nc' for i in range(stdate,enddt)]
        return fnames

def crop_epa(file, start_dt, end_dt,llat,ulat,llon,ulon):
    #print("crop_epa")
    #for t in range(1):
    df = pd.read_csv(file)
    # If the header contains "Date_GMT", it uses underscores.
    if any(col == "Date_GMT" for col in df.columns):
        df.columns = [c.replace("_", " ") for c in df.columns]


    df = df[(df['Latitude'] >= llat) & (df['Latitude'] <= ulat) & (df.Longitude >= llon) & (df.Longitude <= ulon)].reset_index()
    try:
        df['Datetime GMT']=pd.to_datetime(df['Date GMT']+ ' ' + df['Time GMT'])
    except Exception as e:
        print(e, "trying with underscore")
        df['Datetime GMT']=pd.to_datetime(df['Date_GMT']+ ' ' + df['Time_GMT'])
    df= df[(df['Datetime GMT'] >= pd.to_datetime(start_dt) ) & (df['Datetime GMT'] <= pd.to_datetime(end_dt))]
    lon,lat=df['Longitude'].unique(), df['Latitude'].unique()
    #return_files =[dir_epa+'%s_%s_%s.csv'%(time,epa_code[i],year,) for i in range(len(epa_code))]
    return lon,lat,df

fnames = make_cmaq_fnames(fstart,year,month,stdate,enddt+1)
cmaq=[Dataset(dir_cmaq+fnames[i]) for i in range(len(fnames))]

for v in range(len(var)):
	# Print information about the current EPA file and geographic bounding box
	print(epa_files[v], start_dt, end_dt, llat, ulat, llon, ulon)

	# Crop the EPA data based on the specified time range and bounding box
	epa_lon, epa_lat, df = crop_epa(epa_files[v], start_dt, end_dt, llat, ulat, llon, ulon)

	# Extract CMAQ variable data for the current variable
	cmaqv = [np.array(cmaq[d][var[v]]) for d in range(len(cmaq))]

	dataframes = []

	# Loop through each EPA station's longitude and latitude
	for i in range(len(epa_lon)):
		# Filter the EPA dataframe for the current station's coordinates
		tmp = df[(df['Longitude'] == epa_lon[i]) & (df['Latitude'] == epa_lat[i])]
		
		# Get unique metadata: state, county, and units of measure
		tsn = tmp['State Name'].unique()[0]
		tcn = tmp['County Name'].unique()[0]
		um = tmp['Units of Measure'].unique()[0]
		
		# Find the closest grid index in the CMAQ dataset for the current EPA station
		x, y = find_index([epa_lon[i]], [epa_lat[i]], cmaq_lon, cmaq_lat)
		x, y = x[0], y[0]
		
		# Skip edge cases where the index is out of bounds (e.g., at the boundary of the dataset)
		if (x != 0) | (y != 0) | (x != cmaq_lon.shape[0]) | (y != cmaq_lat.shape[1]):  # remove edge cases
			# Resample the EPA data to 1-hour intervals and fill missing values with NaN
			tmp.index = tmp['Datetime GMT']
			t_index = pd.date_range(start=start_dt, end=end_dt, freq='1h')
			#tmp = tmp.resample('1H').mean().reindex(t_index).fillna(np.nan)
			
			# Separate numeric and non-numeric columns
			num_cols = tmp.select_dtypes(include='number').columns
			non_num_cols = tmp.select_dtypes(exclude='number').columns

			# Resample numeric columns with mean
			tmp_num = tmp[num_cols].resample('1h').mean().reindex(t_index).fillna(np.nan)

			# Resample non-numeric columns by taking the first non-null value
			tmp_non = tmp[non_num_cols].resample('1h').first()

			# Combine back together
			tmp = pd.concat([tmp_num, tmp_non], axis=1)
			
			# Add grid indices, state name, county name, and units of measure to the EPA data
			tmp['x'], tmp['y'] = x, y
			tmp['State Name'] = tsn
			tmp['County Name'] = tcn
			tmp['Units of Measure'] = um

			# Retrieve CMAQ data at the matching grid index for the current EPA station
			z = [cmaqv[d][:, 0, x, y] for d in range(len(cmaq))]
			zi = []

			# Flatten the CMAQ data into a single list
			for q in range(len(z)):
				zi = zi + list(z[q])

			# Convert the CMAQ data to a flat array
			z = np.array(zi).ravel()
			print(len(z))
			
			# If the CMAQ data is shorter than the EPA data, pad it with NaN
			if len(z) < len(tmp): 
				z = list(z) + [np.nan] * (len(tmp) - len(z))
			
			# Add the CMAQ data to the EPA dataframe
			tmp['CMAQ'] = z

			# Remove 'dt' and 'level_0' columns, then reset the dataframe index
			tmp['dt'] = tmp.index
			tmp['level_0'] = tmp.index
			tmp = tmp.reset_index(drop=True)
			
			dataframes.append(tmp)
				
		else:
			print(tmp['State Name'],tmp['County Name'],  epa_lon[i], epa_lat[i], "failed")

	# Remove rows where 'Sample Measurement' is zero or negative (invalid values)
	#final_df.loc[final_df['Sample Measurement'] <= 0] = np.nan  # Remove zeros
	#final_df.loc[final_df['Sample Measurement'] < 0] = np.nan   # Remove negatives

	# Save the combined EPA and CMAQ data to a CSV file
	final_df = pd.concat(dataframes, ignore_index=True)
	final_df.to_csv(dir_epa + var[v] + '_' + domain + '_' + str(year) + '_' + str(mo) + '_EPA_CMAQ_Combine_2020_RWC.csv')
	print('done ' + var[v])
