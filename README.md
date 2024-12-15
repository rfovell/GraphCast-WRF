# GraphCast-WRF
Sources and methods for downscaling GraphCast forecasts in GRIB format with WRF-ARW (or MPAS).
Author: Robert Fovell, University at Albany, rfovell@albany.edu

In Brewer, Fovell, and Capps (2025, <i>Weather and Forecasting</i>, hereafter BFC25), we presented WRF-ARW simulations initialized with the GraphCast (GC; https://www.science.org/doi/10.1126/science.adi2336) machine learning weather prediction model. This document provides a step-by-step procedure for how those runs were made.  The simulations used a (now) older version of ECMWF's [ai-models](https://github.com/ecmwf-lab/ai-models) front-end, ECMWF's [ai-models-graphcast](https://github.com/ecmwf-lab/ai-models-graphcast) code, and Google's [graphcast](https://github.com/google-deepmind/graphcast) package and were made between December 2023 to February 2024.

These instructions work on computers I have access to and may need substantial modification for your use.  Software evolution is already making it difficult to re-create the exact environment we used.  Differences in software versions and compilers may (will) result in some variations in the results.

These instructions also presume you have created an account on [Copernicus](https://www.copernicus.eu/en/access-data) to access ERA5 reanalysis data in GRIB format for Step #2. 

An example is provided, referencing a GC run initialized from ERA5 reanalysis, and WRF initialized with a combination of GC's and GFS' forecasts.  GC does not provide all fields needed to initialize WRF, so a second source model is required, and BFC25 used GFS for this.  Specifically, GC does not provide soil fields, skin temperature, SST, surface pressure, and 2 m relative humidity, and nor does it provide invariant fields like soilhgt and land-sea mask.

The specific example involves GC forecasts started at 12/28/2021 at 00 UTC and information from the GFS operational forecast from the same cycle.  The example uses ai-models/ai-models-graphcast to fetch required ERA5 information from Copernicus and run GC, and the GFS 0.25 degree data came from AWS.  These are "last mile" simulations in that they start with GC and GFS forecasts instead of 0-h analyses.  The forecast period starts 12/30/2021 at 00 UTC.  See BFC25 for more information.  For the paper, we did not use surface pressure and 2 m relative humidity information for GFS but the Vtable below includes those fields.  Inclusion makes a small difference.

Before and at this writing, a significant deficiency of the ai-models approach we used is that all of GC's forecasts (apart from hour 0) are produced as a batch at the end of the model run.

Software used/needed:
Anaconda Python;
[WRF-ARW V4.5.1](https://github.com/wrf-model/WRF/releases/tag/v4.5.1);
[WPS V4.5](https://github.com/wrf-model/WPS/releases/tag/v4.5) [with modification to ungrib/src/rd_grib2.F, and METGRID.TBL, below];
ECMWF’s front-end: ai-models and ai-models-graphcast; GraphCast (version 0.1);
Vtable.GC; Vtable.GFSSOIL; ungrib/src/rd_grib2.F for WPS; metgrid/METGRID.TBL for your run directory


The conda environment used for the paper's experiments included the following:

ai-models                 0.5.3                    
ai-models-graphcast       0.0.7                    
flax                      0.8.2                    
graphcast                 0.1                      
jax                       0.4.23                   
jaxlib                    0.4.23                   
numpy                     1.26.4                   


--------------------------------------------------------------------------------------------------------
(1) Install software into new conda environment and acquire GC assets [done once]
--------------------------------------------------------------------------------------------------------
conda create -n GCRUN <br>
conda activate GCRUN

• these commands install more recent versions of ai-models and ai-models-graphcast than we used, but older versions of flax, jax, and jaxlib needed to operate the version of GC we employed

conda install python=3.10 <br>
pip install ai-models <br>
pip install ai-models-graphcast <br>
pip install flax==0.8.2 <br>
pip install jax==0.4.23 <br>
pip install jaxlib==0.4.23 <br>

• get and install ECMWF's ai-models-graphcast. This will be the latest version and not the one used in our experiments.  This will create a new directory called "ai-models-graphcast"

git clone https://github.com/ecmwf-lab/ai-models-graphcast.git <br>
cd ai-models-graphcast

• revert to an older graphcast version to avoid this error: "AttributeError: module 'jax' has no attribute 'tree'"<br>
• to revert, replace the contents of requirements.txt with this line, which reverts to last verified version that works with this software setup:<br>

git+https://github.com/deepmind/graphcast.git@e80f4d4 <br>

• install non-cpu version (what we used; runs more slowly) <br>
pip install -r requirements.txt <br>

• we are installing jax and jaxlib again to ensure we have a version that works with this code <br>
• this now generates some complaints about chex and optax that apparently can be ignored <br>

pip install jax==0.4.23 <br>
pip install jaxlib==0.4.23 <br>

• and now we need an older numpy or the code breaks <br>

pip install numpy==1.26.4 <br>

• download the assets that will be stored in subdirectories "params" and "stats" <br>
• this command may result in some error messages and a request for a URL but only after the needed assets are downloaded, so you should be OK to break out of the request <br>

ai-models --download-assets graphcast <br>

• files obtained: <br>
- in params: 'GraphCast_operational - ERA5-HRES 1979-2021 - resolution 0.25 - pressure levels 13 - mesh 2to6 - precipitation output only.npz' <br>
- in stats: diffs_stddev_by_level.nc, mean_by_level.nc, stddev_by_level.nc <br>

--------------------------------------------------------------------------------------------------------
(2) Run GC initialized with ERA5 reanalysis
--------------------------------------------------------------------------------------------------------

• This example starts with ERA5 reanalysis data at 12/28/2021 at 12Z (and 6Z) and runs 72 hours <br>
• ERA5 files are obtained from Copernicus.  You have to have a Copernicus account for this to work. Fetching these data can be very slow <br>
• Files named out-*.grib are created holding GC forecasts <br>

ai-models --input cds --date 20211228 --time 12 --path 'out-{step}.grib' --lead-time 72 graphcast <br>

--------------------------------------------------------------------------------------------------------
(3) Get GFS operational forecasts, and process them with ungrib.exe using Vtable.GFSSOIL
--------------------------------------------------------------------------------------------------------

• in this version, GFS is providing soil information, skin temperature, and snow to WRF <br>
• it is also providing soilhgt and land-sea mask, both of which could also be provided from ERA5 invariant fields that GC was trained on <br>
• this example presumes the intermediate files are called 'GFSSOIL' <br>
• this example used interval_seconds=21600 owing to the 6-h forecast interval of GC <br>
• for this specific BFC25 example, we used GFS information from the 2021122800 GFS run obtained from Amazon Web Services for forecast hours  <br>


--------------------------------------------------------------------------------------------------------
(4) Process GC GRIB files with ungrib.exe using Vtable.GC
--------------------------------------------------------------------------------------------------------

• BCS25 started simulations on 12/30/2021 at 00 UTC, which means GC forecast hours  <br>
• this example presumes the intermediate files are called 'GC' <br>

--------------------------------------------------------------------------------------------------------
(5) Use metgrid.exe to combine the GFS and GC intermediate files
--------------------------------------------------------------------------------------------------------

• the modified METGRID.TBL for the metgrid/ directory contains an entry for geopotential <br>
• the unpacked GC information is listed second so its fields get first priority <br>

&metgrid <br>
 fg_name = 'GFSSOIL','GC', <br>

--------------------------------------------------------------------------------------------------------
(6) Run real.exe and wrf.exe as usual
--------------------------------------------------------------------------------------------------------

 num_metgrid_levels                  = 14,	[owing to GC] <br>
 num_metgrid_soil_levels             = 4,	[soil information from GFS] <br>


--------------------------------------------------------------------------------------------------------
NOTES:
--------------------------------------------------------------------------------------------------------

(a) Installing ai-models-graphcast in the manner outlined now provokes messages like these, which do not appear to influence results: <br>

chex 0.1.88 requires jax>=0.4.27, but you have jax 0.4.23 which is incompatible.<br>
chex 0.1.88 requires jaxlib>=0.4.27, but you have jaxlib 0.4.23 which is incompatible.<br>
optax 0.2.4 requires jax>=0.4.27, but you have jax 0.4.23 which is incompatible.<br>
optax 0.2.4 requires jaxlib>=0.4.27, but you have jaxlib 0.4.23 which is incompatible.<br>

(b) An error message like this -- AttributeError: module 'jax' has no attribute 'tree' -- may mean you are using a too-recent version of jax and jaxlib<br>

(c) The following error message means you are using numpy version 2 or later.  These runs were made using numpy version 1.26.4.<br>

ValueError: Unable to avoid copy while creating an array as requested.<br>
If using `np.array(obj, copy=False)` replace it with `np.asarray(obj)` to allow a copy when needed (no behavior c
hange in NumPy 1.x).<br>

[END]
