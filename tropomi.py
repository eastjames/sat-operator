# TROPOMI XCH4 observation operator for GEO-Chem
# Code is from https://github.com/geoschem/integrated_methane_inversion/tree/main/src/inversion_scripts/operators
# Used here for more convenient use offline
# james east


import numpy as np
import xarray as xr
import pandas as pd
import datetime
import cftime
from shapely.geometry import Polygon



def apply_average_tropomi_operator(
    filename,
    blended,
    n_elements,
    gc_startdate,
    gc_enddate,
    xlim,
    ylim,
    gc_cache,
    pedge_cache = None,
    tunits_out = 'hours since 2019-01-01'
):
    """
    Apply the averaging tropomi operator to map GEOS-Chem methane data to TROPOMI observation space.

    Arguments
        filename       [str]        : TROPOMI netcdf data file to read
        blended        [bool]       : if True, use blended TROPOMI+GOSAT data
        n_elements     [int]        : Number of state vector elements
        gc_startdate   [datetime64] : First day of inversion period, for GEOS-Chem and TROPOMI
        gc_enddate     [datetime64] : Last day of inversion period, for GEOS-Chem and TROPOMI
        xlim           [float]      : Longitude bounds for simulation domain
        ylim           [float]      : Latitude bounds for simulation domain
        gc_cache       [str]        : Path to GEOS-Chem output data
        tunits_out     [str]        : cftime-compatible datetime units

    Returns
        output         [dict]       : Dictionary with:
                                        - obs_GC : GEOS-Chem and TROPOMI methane data
                                        - TROPOMI methane
                                        - GEOS-Chem methane
                                        - TROPOMI lat, lon
                                        - TROPOMI lat index, lon index
    """

    # Read TROPOMI data
    assert isinstance(blended, bool), 'blended must be type bool'
    if blended:
        TROPOMI = read_blended(filename)
    else:
        TROPOMI = read_tropomi(filename)
    if TROPOMI == None:
        print(f"Skipping {filename} due to file processing issue.")
        return TROPOMI

    # We're only going to consider data within lat/lon/time bounds, with QA > 0.5, and with safe surface albedo values
    if blended:
        # Only going to consider data within lat/lon/time bounds and without problematic coastal pixels
        sat_ind = filter_blended(TROPOMI, xlim, ylim, gc_startdate, gc_enddate)
    else:
        # Only going to consider data within lat/lon/time bounds, with QA > 0.5, and with safe surface albedo values
        sat_ind = filter_tropomi(TROPOMI, xlim, ylim, gc_startdate, gc_enddate)

    # get the lat/lons of gc gridcells
    gc_lat_lon = get_gc_lat_lon(gc_cache, gc_startdate)

    # map tropomi obs into gridcells and average the observations
    # into each gridcell. Only returns gridcells containing observations
    obs_mapped_to_gc = average_tropomi_observations(TROPOMI, gc_lat_lon, sat_ind)
    n_gridcells = len(obs_mapped_to_gc)
    print(n_gridcells)

    # create list to store the dates/hour of each gridcell
    all_strdate = [gridcell["time"] for gridcell in obs_mapped_to_gc]
    all_strdate = list(set(all_strdate))

    # Read GEOS_Chem data for the dates of interest
    all_date_gc = read_all_geoschem(all_strdate, gc_cache, pedge_cache)

    # Initialize array with n_gridcells rows and 6 columns. Columns are TROPOMI CH4, GEOSChem CH4, longitude, latitude, observation counts
    obs_GC = np.zeros([n_gridcells, 9], dtype=np.float32)
    obs_GC.fill(np.nan)

    # For each gridcell dict with tropomi obs:
    for i, gridcell_dict in enumerate(obs_mapped_to_gc):

        # Get GEOS-Chem data for the date of the observation:
        p_sat = gridcell_dict["p_sat"]
        dry_air_subcolumns = gridcell_dict["dry_air_subcolumns"]  # mol m-2
        apriori = gridcell_dict["apriori"]  # mol m-2
        avkern = gridcell_dict["avkern"]
        strdate = gridcell_dict["time"]
        GEOSCHEM = all_date_gc[strdate]

        # Get GEOS-Chem pressure edges for the cell
        p_gc = GEOSCHEM["PEDGE"][gridcell_dict["iGC"], gridcell_dict["jGC"], :]
        # Get GEOS-Chem methane for the cell
        gc_CH4 = GEOSCHEM["CH4"][gridcell_dict["iGC"], gridcell_dict["jGC"], :]
        # Get merged GEOS-Chem/TROPOMI pressure grid for the cell
        merged = merge_pressure_grids(p_sat, p_gc)
        # Remap GEOS-Chem methane to TROPOMI pressure levels
        sat_CH4 = remap(
            gc_CH4,
            merged["data_type"],
            merged["p_merge"],
            merged["edge_index"],
            merged["first_gc_edge"],
        )  # ppb
        # Convert ppb to mol m-2
        sat_CH4_molm2 = sat_CH4 * 1e-9 * dry_air_subcolumns  # mol m-2
        # Derive the column-averaged XCH4 that TROPOMI would see over this ground cell
        # using eq. 46 from TROPOMI Methane ATBD, Hasekamp et al. 2019
        virtual_tropomi = (
            sum(apriori + avkern * (sat_CH4_molm2 - apriori))
            / sum(dry_air_subcolumns)
            * 1e9
        )  # ppb
        
        # time index
        dt = pd.to_datetime(gridcell_dict['time'],format='%Y%m%d_%H')
        tidx = cftime.date2num(dt, tunits_out)
        
        # Save actual and virtual TROPOMI data
        obs_GC[i, 0] = gridcell_dict[
            "methane"
        ]  # Actual TROPOMI methane column observation
        obs_GC[i, 1] = virtual_tropomi  # Virtual TROPOMI methane column observation
        obs_GC[i, 2] = gridcell_dict["lon_sat"]  # TROPOMI longitude
        obs_GC[i, 3] = gridcell_dict["lat_sat"]  # TROPOMI latitude
        obs_GC[i, 4] = gridcell_dict["observation_count"]  # observation counts
        obs_GC[i, 5] = gridcell_dict["iGC"] # GC i index
        obs_GC[i, 6] = gridcell_dict["jGC"] # GC j index
        obs_GC[i, 7] = tidx

    # Output
    output = {}

    # Always return the coincident TROPOMI and GEOS-Chem data
    output["obs_GC"] = obs_GC

    return output


def accumulate_to_dataset(
    obs,
    #gc_startdate,
    #gc_enddate,
    gc_lat_lon,
    #gc_cache,
    tunits_in = 'hours since 2019-01-01',
    #nominal_date = '2019-01-01'
    
):
    '''
    take output from averaging operator and 
    put the values back on the GC grid
    '''
    # don't process empty packet
    if obs.shape[0] == 0:
        return None
    
    # get the lat/lons of gc gridcells
    #tmpdate = pd.to_datetime(nominal_date)
    #gc_lat_lon = get_gc_lat_lon(gc_cache, tmpdate)
    
    # time coords
    hours = np.unique(obs[:,7])
    cfdates = cftime.num2date(hours,tunits_in)
    tcoords = xr.cftime_range(
        cfdates[0],
        cfdates[-1],
        freq='H'
    ).to_datetimeindex()
    ishape = gc_lat_lon['lon'].shape[0]
    jshape = gc_lat_lon['lat'].shape[0]
    tshape = tcoords.shape[0]
    
    # dataarray shape
    dimlist = ['time','lat','lon']
    emptysurface = np.full((tshape,jshape,ishape),np.nan)
    
    # dataset
    outds = xr.Dataset(
        data_vars = dict(
            tropomi_methane=(dimlist, emptysurface.copy()),
            geoschem_methane=(dimlist, emptysurface.copy()),
            observation_count=(dimlist, emptysurface.copy())
        ),
        coords = {
            **gc_lat_lon,
            'time':tcoords
        }
    )
    
    # time indices for all data
    thour = obs[:,7].astype(int)
    tidx = thour - thour.min()
    
    outds['tropomi_methane'].values[
        tidx, # time
        obs[:,6].astype(int), # lat
        obs[:,5].astype(int) # lon
    ] = obs[:,0]
    
    outds['geoschem_methane'].values[
        tidx, # time
        obs[:,6].astype(int), #lat
        obs[:,5].astype(int) #lon
    ] = obs[:,1]
    
    outds['observation_count'].values[
        tidx, # time
        obs[:,6].astype(int), # lat
        obs[:,5].astype(int) # lon
    ] = obs[:,4]
    
    for v in ['tropomi_methane','geoschem_methane']:
        outds[v].attrs['units'] = 'ppb'
        outds[v].attrs['long_name'] = 'dry column average CH4'
    
    outds['observation_count'].attrs['units'] = '1'
    outds['observation_count'].attrs['long_name'] = 'number of tropomi observations in grid cell'
    
    return outds


def read_tropomi(filename):
    """
    Read TROPOMI data and save important variables to dictionary.

    Arguments
        filename [str]  : TROPOMI netcdf data file to read

    Returns
        dat      [dict] : Dictionary of important variables from TROPOMI:
                            - CH4
                            - Latitude
                            - Longitude
                            - QA value
                            - Time (utc time reshaped for orbit)
                            - Averaging kernel
                            - SWIR albedo
                            - NIR albedo
                            - Blended albedo
                            - CH4 prior profile
                            - Dry air subcolumns
                            - Latitude bounds
                            - Longitude bounds
                            - Vertical pressure profile
    """

    # Initialize dictionary for TROPOMI data
    dat = {}

    # Store methane, QA, lat, lon
    try:
        with xr.open_dataset(filename, group="PRODUCT") as _:
            pass
    except Exception as e:
        print(f"Error opening {filename}: {e}")
        return None

    with xr.open_dataset(filename, group="PRODUCT") as tropomi_data:
        dat["methane"] = tropomi_data["methane_mixing_ratio_bias_corrected"].values[0, :, :]
        dat["qa_value"] = tropomi_data["qa_value"].values[0, :, :]
        dat["longitude"] = tropomi_data["longitude"].values[0, :, :]
        dat["latitude"] = tropomi_data["latitude"].values[0, :, :]

        # Store UTC time
        utc_str = tropomi_data["time_utc"].values[0,:]
        utc_str = np.array([d.replace("Z","") for d in utc_str]).astype("datetime64[ns]")
        dat["time"] = np.repeat(utc_str[:, np.newaxis], dat["methane"].shape[1], axis=1)

    # Store column averaging kernel, SWIR and NIR surface albedo
    with xr.open_dataset(filename, group="PRODUCT/SUPPORT_DATA/DETAILED_RESULTS") as tropomi_data:
        dat["column_AK"] = tropomi_data["column_averaging_kernel"].values[0, :, :, ::-1]
        dat["swir_albedo"] = tropomi_data["surface_albedo_SWIR"].values[0, :, :]
        dat["nir_albedo"] = tropomi_data["surface_albedo_NIR"].values[0, :, :]
        dat["blended_albedo"] = 2.4 * dat["nir_albedo"] - 1.13 * dat["swir_albedo"]

    # Store methane prior profile, dry air subcolumns
    with xr.open_dataset(filename, group="PRODUCT/SUPPORT_DATA/INPUT_DATA") as tropomi_data: 
        dat["methane_profile_apriori"] = tropomi_data["methane_profile_apriori"].values[
            0, :, :, ::-1
        ]  # mol m-2
        dat["dry_air_subcolumns"] = tropomi_data["dry_air_subcolumns"].values[
            0, :, :, ::-1
        ]  # mol m-2

        # Also get pressure interval and surface pressure for use below
        pressure_interval = (
            tropomi_data["pressure_interval"].values[0, :, :] / 100
        )  # Pa -> hPa
        surface_pressure = (
            tropomi_data["surface_pressure"].values[0, :, :] / 100
        )  # Pa -> hPa

    # Store latitude and longitude bounds for pixels
    with xr.open_dataset(filename, group="PRODUCT/SUPPORT_DATA/GEOLOCATIONS") as tropomi_data:
        dat["longitude_bounds"] = tropomi_data["longitude_bounds"].values[0, :, :, :]
        dat["latitude_bounds"] = tropomi_data["latitude_bounds"].values[0, :, :, :]

    # Store vertical pressure profile
    n1 = dat["methane"].shape[0]  # length of along-track dimension (scanline) of retrieval field
    n2 = dat["methane"].shape[1]  # length of across-track dimension (ground_pixel) of retrieval field
    pressures = np.full([n1, n2, 12+1], np.nan, dtype=np.float32)
    for i in range(12 + 1):
        pressures[:, :, i] = surface_pressure - i * pressure_interval
    dat["pressures"] = pressures

    return dat


def read_blended(filename):
    """
    Read Blended TROPOMI+GOSAT data and save important variables to dictionary.
    Arguments
        filename [str]  : Blended TROPOMI+GOSAT netcdf data file to read
    Returns
        dat      [dict] : Dictionary of important variables from Blended TROPOMI+GOSAT:
                            - CH4
                            - Latitude
                            - Longitude
                            - Time (utc time reshaped for orbit)
                            - Averaging kernel
                            - SWIR albedo
                            - NIR albedo
                            - Blended albedo
                            - CH4 prior profile
                            - Dry air subcolumns
                            - Latitude bounds
                            - Longitude bounds
                            - Surface classification
                            - Chi-Square for SWIR
                            - Vertical pressure profile
    """

    # Test to make sure the file can be opened
    try:
        with xr.open_dataset(filename) as _:
            pass
    except Exception as e:
        print(f'Error opening {filename}: {e}')
        return None

    assert "BLND" in filename, "BLND not in filename, but a blended function is being used"

    # Initialize dictionary for Blended TROPOMI+GOSAT data
    dat = {}

    # Extract data from netCDF file to our dictionary
    with xr.open_dataset(filename) as blended_data:

        dat["methane"] = blended_data["methane_mixing_ratio_blended"].values[:]
        dat["longitude"] = blended_data["longitude"].values[:]
        dat["latitude"] = blended_data["latitude"].values[:]
        dat["column_AK"] = blended_data["column_averaging_kernel"].values[:, ::-1]
        dat["swir_albedo"] = blended_data["surface_albedo_SWIR"][:]
        dat["nir_albedo"] = blended_data["surface_albedo_NIR"].values[:]
        dat["blended_albedo"] = 2.4 * dat["nir_albedo"] - 1.13 * dat["swir_albedo"]
        dat["methane_profile_apriori"] = blended_data["methane_profile_apriori"].values[:, ::-1]
        dat["dry_air_subcolumns"] = blended_data["dry_air_subcolumns"].values[:, ::-1]
        dat["longitude_bounds"] = blended_data["longitude_bounds"].values[:]
        dat["latitude_bounds"] = blended_data["latitude_bounds"].values[:]
        dat["surface_classification"] = (blended_data["surface_classification"].values[:].astype("uint8") & 0x03).astype(int)
        dat["chi_square_SWIR"] = blended_data["chi_square_SWIR"].values[:]

        # Remove "Z" from time so that numpy doesn't throw a warning
        utc_str = blended_data["time_utc"].values[:]
        dat["time"] = np.array([d.replace("Z","") for d in utc_str]).astype("datetime64[ns]")

        # Need to calculate the pressure for the 13 TROPOMI levels (12 layer edges)
        pressure_interval = (blended_data["pressure_interval"].values[:] / 100)  # Pa -> hPa
        surface_pressure = (blended_data["surface_pressure"].values[:] / 100)    # Pa -> hPa
        n = len(dat["methane"])
        pressures = np.full([n, 12 + 1], np.nan, dtype=np.float32)
        for i in range(12 + 1):
            pressures[:, i] = surface_pressure - i * pressure_interval
        dat["pressures"] = pressures

    # Add an axis here to mimic the (scanline, groundpixel) format of operational TROPOMI data
    # This is so the blended data will be compatible with the TROPOMI operators
    for key in dat.keys():
        dat[key] = np.expand_dims(dat[key], axis=0)

    return dat


def filter_tropomi(tropomi_data, xlim, ylim, startdate, enddate):
    """
    Description:
        Filter out any data that does not meet the following
        criteria: We only consider data within lat/lon/time bounds,
        with QA > 0.5, and with safe surface albedo values
    Returns:
        numpy array with satellite indices for filtered tropomi data.
    """
    return np.where(
        (tropomi_data["longitude"] > xlim[0])
        & (tropomi_data["longitude"] < xlim[1])
        & (tropomi_data["latitude"] > ylim[0])
        & (tropomi_data["latitude"] < ylim[1])
        & (tropomi_data["time"] >= startdate)
        & (tropomi_data["time"] <= enddate)
        & (tropomi_data["qa_value"] >= 1.0)
        #& (tropomi_data["swir_albedo"] > 0.05)
        #& (tropomi_data["blended_albedo"] < 0.85)
        & (tropomi_data["longitude_bounds"].ptp(axis=2) < 5)
    )


def filter_blended(blended_data, xlim, ylim, startdate, enddate):
    """
    Description:
        Filter out any data that does not meet the following
        criteria: We only consider data within lat/lon/time bounds,
        that don't cross the antimeridian, and we filter out all
        coastal pixels (surface classification 3) and inland water
        pixels with a poor fit (surface classifcation 2, 
        SWIR chi-2 > 20000) (recommendation from Balasus et al. 2023)
    Returns:
        numpy array with satellite indices for filtered tropomi data.
    """
    return np.where(
        (blended_data["longitude"] > xlim[0])
        & (blended_data["longitude"] < xlim[1])
        & (blended_data["latitude"] > ylim[0])
        & (blended_data["latitude"] < ylim[1])
        & (blended_data["time"] >= startdate)
        & (blended_data["time"] <= enddate)
        & (blended_data["longitude_bounds"].ptp(axis=2) < 5)
        & ~((blended_data["surface_classification"] == 3) | ((blended_data["surface_classification"] == 2) & (blended_data["chi_square_SWIR"][:] > 20000)))
    )


def get_gc_lat_lon(gc_cache, start_date):
    """
    get dictionary of lat/lon values for gc gridcells

    Arguments
        gc_cache    [str]   : path to gc data
        start_date  [str]   : start date of the inversion

    Returns
        output      [dict]  : dictionary with the following fields:
                                - lat : list of GC latitudes
                                - lon : list of GC longitudes
    """
    gc_ll = {}
    date = pd.to_datetime(start_date).strftime("%Y%m%d")
    file_species = f"GEOSChem.SpeciesConc.{date}_0000z.nc4"
    filename = f"{gc_cache}/{file_species}"
    gc_data = xr.open_dataset(filename)
    gc_ll["lon"] = gc_data["lon"].values
    gc_ll["lat"] = gc_data["lat"].values

    gc_data.close()
    return gc_ll


def average_tropomi_observations(TROPOMI, gc_lat_lon, sat_ind):
    """
    Map TROPOMI observations into appropriate gc gridcells. Then average all
    observations within a gridcell for processing. Use area weighting if
    observation overlaps multiple gridcells.

    Arguments
        TROPOMI        [dict]   : Dict of tropomi data
        gc_lat_lon     [list]   : list of dictionaries containing  gc gridcell info
        sat_ind        [int]    : index list of Tropomi data that passes filters

    Returns
        output         [dict[]]   : flat list of dictionaries the following fields:
                                    - lat                 : gridcell latitude
                                    - lon                 : gridcell longitude
                                    - iGC                 : longitude index value
                                    - jGC                 : latitude index value
                                    - lat_sat             : averaged tropomi latitude
                                    - lon_sat             : averaged tropomi longitude
                                    - overlap_area        : averaged overlap area with gridcell
                                    - p_sat               : averaged pressure for sat
                                    - dry_air_subcolumns  : averaged
                                    - apriori             : averaged
                                    - avkern              : averaged average kernel
                                    - time                : averaged time
                                    - methane             : averaged methane
                                    - observation_count   : number of observations averaged in cell
                                    - observation_weights : area weights for the observation

    """
    n_obs = len(sat_ind[0])
    print("Found", n_obs, "TROPOMI observations.")
    gc_lats = gc_lat_lon["lat"]
    gc_lons = gc_lat_lon["lon"]
    gridcell_dicts = get_gridcell_list(gc_lons, gc_lats)
    dlon_median = np.median(np.diff(gc_lons))
    dlat_median = np.median(np.diff(gc_lats))

    for k in range(n_obs):
        iSat = sat_ind[0][k]  # lat index
        jSat = sat_ind[1][k]  # lon index

        # Find GEOS-Chem lats & lons closest to the corners of the TROPOMI pixel
        longitude_bounds = TROPOMI["longitude_bounds"][iSat, jSat, :]
        latitude_bounds = TROPOMI["latitude_bounds"][iSat, jSat, :]
        corners_lon_index = []
        corners_lat_index = []

        for l in range(4):
            iGC = nearest_loc(longitude_bounds[l], gc_lons, tolerance=max(dlon_median,0.5))
            jGC = nearest_loc(latitude_bounds[l], gc_lats, tolerance=max(dlon_median,0.5))
            corners_lon_index.append(iGC)
            corners_lat_index.append(jGC)

        # If the tolerance in nearest_loc() is not satisfied, skip the observation
        if np.nan in corners_lon_index + corners_lat_index:
            continue

        # Get lat/lon indexes and coordinates of GEOS-Chem grid cells closest to the TROPOMI corners
        ij_GC = [(x, y) for x in set(corners_lon_index) for y in set(corners_lat_index)]
        gc_coords = [(gc_lons[i], gc_lats[j]) for i, j in ij_GC]

        # Compute the overlapping area between the TROPOMI pixel and GEOS-Chem grid cells it touches
        overlap_area = np.zeros(len(gc_coords))
        dlon = gc_lons[1] - gc_lons[0]
        dlat = gc_lats[1] - gc_lats[0]

        # Polygon representing TROPOMI pixel
        polygon_tropomi = Polygon(np.column_stack((longitude_bounds, latitude_bounds)))
        for gridcellIndex in range(len(gc_coords)):
            # Define polygon representing the GEOS-Chem grid cell
            coords = gc_coords[gridcellIndex]
            geoschem_corners_lon = [
                coords[0] - dlon / 2,
                coords[0] + dlon / 2,
                coords[0] + dlon / 2,
                coords[0] - dlon / 2,
            ]
            geoschem_corners_lat = [
                coords[1] - dlat / 2,
                coords[1] - dlat / 2,
                coords[1] + dlat / 2,
                coords[1] + dlat / 2,
            ]
            polygon_geoschem = Polygon(
                np.column_stack((geoschem_corners_lon, geoschem_corners_lat))
            )
            # Calculate overlapping area as the intersection of the two polygons
            if polygon_geoschem.intersects(polygon_tropomi):
                overlap_area[gridcellIndex] = polygon_tropomi.intersection(
                    polygon_geoschem
                ).area
        # If there is no overlap between GEOS-Chem and TROPOMI, skip to next observation:
        total_overlap_area = sum(overlap_area)

        # iterate through any gridcells with observation overlap
        # weight each observation if observation extent overlaps with multiple
        # gridcells
        for index, overlap in enumerate(overlap_area):
            if not overlap == 0:
                # get the matching dictionary for the gridcell with the overlap
                gridcell_dict = gridcell_dicts[ij_GC[index][0]][ij_GC[index][1]]
                gridcell_dict["lat_sat"].append(TROPOMI["latitude"][iSat, jSat])
                gridcell_dict["lon_sat"].append(TROPOMI["longitude"][iSat, jSat])
                gridcell_dict["overlap_area"].append(overlap)
                gridcell_dict["p_sat"].append(TROPOMI["pressures"][iSat, jSat, :])
                gridcell_dict["dry_air_subcolumns"].append(
                    TROPOMI["dry_air_subcolumns"][iSat, jSat, :]
                )
                gridcell_dict["apriori"].append(
                    TROPOMI["methane_profile_apriori"][iSat, jSat, :]
                )
                gridcell_dict["avkern"].append(TROPOMI["column_AK"][iSat, jSat, :])
                gridcell_dict[
                    "time"
                ].append(  # convert times to epoch time to make taking the mean easier
                    int(pd.to_datetime(str(TROPOMI["time"][iSat, jSat])).strftime("%s"))
                )
                gridcell_dict["methane"].append(
                    TROPOMI["methane"][iSat, jSat]
                )  # Actual TROPOMI methane column observation
                # record weights for averaging later
                gridcell_dict["observation_weights"].append(
                    overlap / total_overlap_area
                )
                # increment the observation count based on overlap area
                gridcell_dict["observation_count"] += overlap / total_overlap_area

    # filter out gridcells without any observations
    gridcell_dicts = [
        item for item in gridcell_dicts.flatten() if item["observation_count"] > 0
    ]
    # weighted average observation values for each gridcell
    for gridcell_dict in gridcell_dicts:
        gridcell_dict["lat_sat"] = np.average(
            gridcell_dict["lat_sat"],
            weights=gridcell_dict["observation_weights"],
        )
        gridcell_dict["lon_sat"] = np.average(
            gridcell_dict["lon_sat"],
            weights=gridcell_dict["observation_weights"],
        )
        gridcell_dict["overlap_area"] = np.average(
            gridcell_dict["overlap_area"],
            weights=gridcell_dict["observation_weights"],
        )
        gridcell_dict["methane"] = np.average(
            gridcell_dict["methane"],
            weights=gridcell_dict["observation_weights"],
        )
        # take mean of epoch times and then convert gc filename time string
        gridcell_dict["time"] = (
            pd.to_datetime(
                datetime.datetime.fromtimestamp(int(np.mean(gridcell_dict["time"])))
            )
            .round("60min")
            .strftime("%Y%m%d_%H")
        )
        # for multi-dimensional arrays, we only take the average across the 0 axis
        gridcell_dict["p_sat"] = np.average(
            gridcell_dict["p_sat"],
            axis=0,
            weights=gridcell_dict["observation_weights"],
        )
        gridcell_dict["dry_air_subcolumns"] = np.average(
            gridcell_dict["dry_air_subcolumns"],
            axis=0,
            weights=gridcell_dict["observation_weights"],
        )
        gridcell_dict["apriori"] = np.average(
            gridcell_dict["apriori"],
            axis=0,
            weights=gridcell_dict["observation_weights"],
        )
        gridcell_dict["avkern"] = np.average(
            gridcell_dict["avkern"],
            axis=0,
            weights=gridcell_dict["observation_weights"],
        )
    return gridcell_dicts



def nearest_loc(query_location, reference_grid, tolerance=0.5):
    """Find the index of the nearest grid location to a query location, with some tolerance."""

    distances = np.abs(reference_grid - query_location)
    ind = distances.argmin()
    if distances[ind] >= tolerance:
        return np.nan
    else:
        return ind
    
    

def get_gridcell_list(lons, lats):
    """
    Create a 2d array of dictionaries, with each dictionary representing a GC gridcell.
    Dictionaries also initialize the fields necessary to store for tropomi data
    (eg. methane, time, p_sat, etc.)

    Arguments
        lons     [float[]]      : list of gc longitudes for region of interest
        lats     [float[]]      : list of gc latitudes for region of interest

    Returns
        gridcells [dict[][]]     : 2D array of dicts representing a gridcell
    """
    # create array of dictionaries to represent gridcells
    gridcells = []
    for i in range(len(lons)):
        for j in range(len(lats)):
            gridcells.append(
                {
                    "lat": lats[j],
                    "lon": lons[i],
                    "iGC": i,
                    "jGC": j,
                    "methane": [],
                    "p_sat": [],
                    "dry_air_subcolumns": [],
                    "apriori": [],
                    "avkern": [],
                    "time": [],
                    "overlap_area": [],
                    "lat_sat": [],
                    "lon_sat": [],
                    "observation_count": 0,
                    "observation_weights": [],
                }
            )
    gridcells = np.array(gridcells).reshape(len(lons), len(lats))
    return gridcells


def read_geoschem(date, gc_cache, pedge_cache=None):
    """
    Read GEOS-Chem data and save important variables to dictionary.

    Arguments
        date           [str]   : Date of interest
        gc_cache       [str]   : Path to GEOS-Chem output data
        pedge_cache       [str]   : Path to GEOS-Chem output data, defaults to `gc_cache`

    Returns
        dat            [dict]  : Dictionary of important variables from GEOS-Chem:
                                    - CH4
                                    - Latitude
                                    - Longitude
                                    - PEDGE
    """
    
    #JDE EDIT
    if pedge_cache is None:
        pedge_cache = gc_cache
    
    # separate date strings to daily and hourly to only open 
    # a faily file, and then extract the hour of interest
    mydate = pd.to_datetime(date,format='%Y%m%d_%H')
                            
    # Assemble file paths to GEOS-Chem output collections for input data
    file_species = f"GEOSChem.SpeciesConc.{mydate.strftime('%Y%m%d')}_0000z.nc4"
    #file_pedge = f"GEOSChem.LevelEdgeDiags.{mydate.strftime('%Y%m%d')}_0000z.nc4"
    file_pedge = f"GEOSChem.StateMet.{mydate.strftime('%Y%m%d')}_0000z.nc4"
    
    #END JDE EDIT

    # Read lat, lon, CH4 from the SpeciecConc collection
    filename = f"{gc_cache}/{file_species}"
    gc_data = xr.open_dataset(filename)
    LON = gc_data["lon"].values
    LAT = gc_data["lat"].values
    #CH4 = gc_data["SpeciesConc_CH4"].values[0, :, :, :] # orig
    CH4 = gc_data["SpeciesConcVV_CH4"].values[mydate.hour, :, :, :] #jde
    CH4 = CH4 * 1e9  # Convert to ppb
    CH4 = np.einsum("lij->jil", CH4)
    Ap = gc_data['hyai'].values
    Bp = gc_data['hybi'].values
    gc_data.close()

    # Read PEDGE from the LevelEdgeDiags collection
    filename = f"{gc_cache}/{file_pedge}"
    gc_data = xr.open_dataset(filename)

    # create pressure edges
    psfc_var = 'Met_PSC2WET'
    psfc = gc_data[psfc_var].values[mydate.hour,:,:] # no lev dim
    psfc = np.einsum('ij->ji', psfc) 
    PEDGE = Ap[None,None,:] + (Bp[None,None,:] * psfc[:,:,None]) # pedge = a + (b * ps)
    gc_data.close()

    # Store GEOS-Chem data in dictionary
    dat = {}
    dat["lon"] = LON
    dat["lat"] = LAT
    dat["PEDGE"] = PEDGE
    dat["CH4"] = CH4

    return dat


def read_all_geoschem(all_strdate, gc_cache, pedge_cache=None):
    """
    Call readgeoschem() for multiple dates in a loop.

    Arguments
        all_strdate    [list, str] : Multiple date strings
        gc_cache       [str]       : Path to GEOS-Chem output data

    Returns
        dat            [dict]      : Dictionary of dictionaries. Each sub-dictionary is returned by read_geoschem()
    """

    dat = {}
    for strdate in all_strdate:
        dat[strdate] = read_geoschem(strdate, gc_cache, pedge_cache)

    return dat



def merge_pressure_grids(p_sat, p_gc):
    """
    Merge TROPOMI & GEOS-Chem vertical pressure grids

    Arguments
        p_sat   [float]    : Pressure edges from TROPOMI (13 edges)     <--- 13-1 = 12 pressure layers
        p_gc    [float]    : Pressure edges from GEOS-Chem (48 edges)   <--- 48-1 = 47 pressure layers

    Returns
        merged  [dict]     : Merged grid dictionary
                                - p_merge       : merged pressure-edge grid
                                - data_type     : for each pressure edge in the merged grid, is it from GEOS-Chem or TROPOMI?
                                - edge_index    : indexes of pressure edges
                                - first_gc_edge : index of first GEOS-Chem pressure edge in the merged grid
    """

    # Combine p_sat and p_gc into merged vertical pressure grid
    p_merge = np.zeros(len(p_sat) + len(p_gc))
    p_merge.fill(np.nan)
    data_type = np.zeros(len(p_sat) + len(p_gc), dtype=int)
    data_type.fill(-99)
    edge_index = []
    i = 0
    j = 0
    k = 0
    while (i < len(p_sat)) or (j < len(p_gc)):
        if i == len(p_sat):
            p_merge[k] = p_gc[j]
            data_type[k] = 2  # geos-chem edge
            j = j + 1
            k = k + 1
            continue
        if j == len(p_gc):
            p_merge[k] = p_sat[i]
            data_type[k] = 1  # tropomi edge
            edge_index.append(k)
            i = i + 1
            k = k + 1
            continue
        if p_sat[i] >= p_gc[j]:
            p_merge[k] = p_sat[i]
            data_type[k] = 1  # tropomi edge
            edge_index.append(k)
            i = i + 1
            k = k + 1
        else:
            p_merge[k] = p_gc[j]
            data_type[k] = 2  # geos-chem edge
            j = j + 1
            k = k + 1

    # Find the first GEOS-Chem pressure edge
    first_gc_edge = -99
    for i in range(len(p_sat) + len(p_gc) - 1):
        if data_type[i] == 2:
            first_gc_edge = i
            break

    # Save data to dictionary
    merged = {}
    merged["p_merge"] = p_merge
    merged["data_type"] = data_type
    merged["edge_index"] = edge_index
    merged["first_gc_edge"] = first_gc_edge

    return merged



def remap(gc_CH4, data_type, p_merge, edge_index, first_gc_edge):
    """
    Remap GEOS-Chem methane to the TROPOMI vertical grid.

    Arguments
        gc_CH4        [float]   : Methane from GEOS-Chem
        p_merge       [float]   : Merged TROPOMI + GEOS-Chem pressure levels, from merge_pressure_grids()
        data_type     [int]     : Labels for pressure edges of merged grid. 1=TROPOMI, 2=GEOS-Chem, from merge_pressure_grids()
        edge_index    [int]     : Indexes of pressure edges, from merge_pressure_grids()
        first_gc_edge [int]     : Index of first GEOS-Chem pressure edge in merged grid, from merge_pressure_grids()

    Returns
        sat_CH4       [float]   : GEOS-Chem methane in TROPOMI pressure coordinates
    """

    # Define CH4 concentrations in the layers of the merged pressure grid
    CH4 = np.zeros(
        len(p_merge) - 1,
    )
    CH4.fill(np.nan)
    k = 0
    for i in range(first_gc_edge, len(p_merge) - 1):
        CH4[i] = gc_CH4[k]
        if data_type[i + 1] == 2:
            k = k + 1
    if first_gc_edge > 0:
        CH4[:first_gc_edge] = CH4[first_gc_edge]

    # Calculate the pressure-weighted mean methane for each TROPOMI layer
    delta_p = p_merge[:-1] - p_merge[1:]
    sat_CH4 = np.zeros(12)
    sat_CH4.fill(np.nan)
    for i in range(len(edge_index) - 1):
        start = edge_index[i]
        end = edge_index[i + 1]
        sum_conc_times_pressure_weights = sum(CH4[start:end] * delta_p[start:end])
        sum_pressure_weights = sum(delta_p[start:end])
        sat_CH4[i] = (
            sum_conc_times_pressure_weights / sum_pressure_weights
        )  # pressure-weighted average

    return sat_CH4
