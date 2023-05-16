TROPOMI observation operator for GEOS-Chem
------------------------------------------

Adapted from IMI code, specifically the files:
https://github.com/geoschem/integrated_methane_inversion/blob/main/src/inversion_scripts/operators/TROPOMI_operator.py
https://github.com/geoschem/integrated_methane_inversion/blob/main/src/inversion_scripts/operators/operator_utilities.py


## Contents
* config.yml
    * user configuration file for GEOS-Chem and TROPOMI file paths and other options
* methanegridder.py
    * main wrapper code to grid L2 retrievals into gridded L3 daily files
* tropomi-match.ipynb
    * notebook James made to look at results, messy
* tropomi.py
    * main code for observation operator
    * applies area-weighting and averages all observations in a grid cell
    * applies GEOS-Chem vertical profiles to retrieval


## Usage
1. Edit config.yml according to needs
2. Optionally create your own batch submission to run on a compute node
3. Run `./methanegridder.py` at the command line or in a batch job in an environment with required python libraries

## required python libraries
* numpy
* xarray
* pandas
* datetime
* cftime
* yaml
* shapely

