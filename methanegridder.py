#!/n/home06/jeast/miniconda3/envs/jpy01/bin/python
import sys
sys.path.append('./')
import tropomi
import pandas as pd
import numpy as np
import xarray as xr
from glob import glob
import matplotlib.pyplot as plt
import yaml

def main():
    # open config file
    with open('config.yml', 'r') as file:
        cfg = yaml.safe_load(file)
    
    startdate = str(cfg['startdate'])
    enddate = str(cfg['enddate'])
    gc_cache = str(cfg['gc_cache'])
    tropomidir = str(cfg['tropomidir'])
    #tropomipattern = str(cfg['tropomipattern'])
    trversion = str(cfg['trversion'])
    trkind = cfg['trkind'] #list
    datadir = str(cfg['datadir'])
    outfile = str(cfg['outfile'])
    pedge_cache = None
    blended = bool(cfg['blended'])
    usecached = bool(cfg['usecached'])
    
    gc_startdate = pd.to_datetime(startdate,format='%Y%j')
    gc_enddate = pd.to_datetime(enddate,format='%Y%j')
    
    # save daily file
    for tdate in pd.date_range(gc_startdate,gc_enddate)[:-1]:
        fname = tdate.strftime(f'{datadir}/{outfile}')
        if usecached and os.path.isfile(fname):
            print(f'\n\n----> SKIPPING {tdate.strftime("%Y-%m-%d")}, FILE EXISTS <----',flush=True)
            continue

        print(f'\n\n----> Processing {tdate.strftime("%Y-%m-%d")} <----',flush=True)
        
        # get the lat/lons of gc gridcells
        gc_lat_lon = tropomi.get_gc_lat_lon(gc_cache, tdate)
        
        # all swaths for today
        tpat = f'{tropomidir}/*{tdate.strftime("%Y%m%d")}*.nc'
        tpaths = sorted(glob(tpat))
        # remove if not day of interest 
        for f in tpaths:
            datestr = tdate.strftime('%Y%m%d')
            dstart = f.split('_')[-5]
            dend = f.split('_')[-6]
            myversion = f.split('_')[-2]
            mykind = f.split('_')[-13]
            if (
                ((datestr in dstart) | (datestr in dend )) &
                (mykind in trkind) &
                (myversion in trversion)
            ):
                continue
            else:
                print(f'skipping {f}',flush=True)
                tpaths = [myf for myf in tpaths if myf != f]
    
        # grid each swath
        tfs = []
        for f in tpaths:
            print(f,flush=True)
            tf = tropomi.apply_average_tropomi_operator(
                filename = f,
                blended = blended,
                n_elements = None,
                gc_startdate = tdate,
                gc_enddate = tdate + pd.Timedelta('1D'),
                xlim = np.array([-180,180]),
                ylim = np.array([-90,90]),
                gc_cache = gc_cache,
                pedge_cache = pedge_cache
            )
            tfs.append(tf)
            print('\n',flush=True)
            
        # convert to xarray dataset
        dslist = [tropomi.accumulate_to_dataset(obs['obs_GC'],gc_lat_lon) for obs in tfs]
        # strip Nones
        dslist = [ds for ds in dslist if ds is not None]
        
        # merge
        dsout = xr.merge(dslist,compat='no_conflicts')
        
        # save
        dsout.to_netcdf(
            fname,
            encoding = {v:{'zlib':True,'complevel':1} for v in dsout.data_vars}
        )
        
    return

if __name__ == '__main__':
    main()

