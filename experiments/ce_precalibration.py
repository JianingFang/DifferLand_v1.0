# script for pre-calibrating canopy efficiency parameter
# for days where the evaporative fraction (EF) is above
# the 80th percentile (well-watered conditions)
# Assumption: the driver files are in ../drivers
# Ouput: save pre-calibrated values to ./ce_opt.csv

import jax
import jax.numpy as jnp
import os
import numpy as np
import pandas as pd
import xarray as xr
from datetime import date
from datetime import timedelta
from tqdm import tqdm
import sys
sys.path.insert(1, '..')
from DifferLand.model.DALEC_990_parinfo import dalec990_parmin, dalec990_parmax
from DifferLand.model.auxi.ACM import ACM
 # vectorize the ACM model for more efficient computation
ACM_V = jax.jit(jax.vmap(jax.jit(ACM), in_axes=[None, 0, 0, 0, 0, 0, 0, None]))

DRIVERS_DIR = "../drivers/"
OUTPUT_FILENAME = "./ce_opt.csv"
EF_PERCENTILE = 80 # EF above the 80th percentile as used in our manuscript


def ce_grid_search(ACM_V, driver_ds, lat, train_sel, ef_train, ef_bound, doy_wet_train, t_max_wet_train, t_min_wet_train, lai_wet_train, rad_wet_train, ca_wet_train):
    ce_grid = jnp.linspace(dalec990_parmin.canopy_efficiency.item(),
                            dalec990_parmax.canopy_efficiency.item(), 500)
    loss_grid = []
    for ce in ce_grid:
        gpp_pred_wet = ACM_V(lat, doy_wet_train, t_max_wet_train, t_min_wet_train,
                             lai_wet_train, rad_wet_train, ca_wet_train, ce)
        gpp_target_wet = driver_ds.GPP.values[train_sel][ef_train >= ef_bound]
        # minimize the MSE loss between predicted and target GPP on well-watered days
        gpp_loss = np.nanmean((gpp_pred_wet - gpp_target_wet) ** 2)
        loss_grid.append(gpp_loss)
    return ce_grid,loss_grid


def get_training_data(DRIVERS_DIR, EF_PERCENTILE, f):
    driver_ds = xr.open_dataset(os.path.join(DRIVERS_DIR, f))
    # extract the variables needed for the ACM model in the driver file
    lat = driver_ds.LAT.values.item()
    doy = driver_ds.DOY.values
    t_max = driver_ds.T2M_MAX.values
    t_min = driver_ds.T2M_MIN.values
    lai = driver_ds.LAI.values
    rad = driver_ds.SSRD.values
    ca = driver_ds.CO2.values
    ef = driver_ds.EF.values
    time_vals = driver_ds.time.values
    if time_vals[1] - time_vals[0] != 1:
        print("WARNING: the script assumes driver in daily temporal resolution," 
              + "change the script if your data is in another temporal resolution.")
    time = np.array([date(2001, 1, 1) + timedelta(days=int(i)) for i in time_vals])
    
    # assuming that all drivers are in daily temporal resolution
    n_years = np.int64(np.round(len(time) / 365.25))
    train_years = np.round(np.ceil(n_years / 2))
    train_end_date = date(int(time[0].year + train_years), 1, 1)
    train_sel = time<train_end_date
    
    # use only the training period for ce pre-calibration 
    doy_train = doy[train_sel]
    t_max_train = t_max[train_sel]
    t_min_train = t_min[train_sel]
    lai_train = lai[train_sel]
    rad_train = rad[train_sel]
    ca_train = ca[train_sel]
    ef_train = ef[train_sel]

    # select days where EF in the upper 80 percentile to represent well-watered conditions.
    ef_bound = np.nanpercentile(ef_train, EF_PERCENTILE)
    doy_wet_train = doy_train[ef_train >= ef_bound]
    t_max_wet_train = t_max_train[ef_train >= ef_bound]
    t_min_wet_train = t_min_train[ef_train >= ef_bound]
    lai_wet_train = lai_train[ef_train >= ef_bound]
    rad_wet_train = rad_train[ef_train >= ef_bound]
    ca_wet_train = ca_train[ef_train >= ef_bound]
    return driver_ds,lat,train_sel,ef_train,ef_bound,doy_wet_train,t_max_wet_train,t_min_wet_train,lai_wet_train,rad_wet_train,ca_wet_train

def main():
    print("Precalibrating ce to to timesteps when evaporative fraction (EF) is above the 80th percentile")

    # the script process all "*.nc" in the driver folder (used in Fang & Gentine 2024, 
    # change the line below to use your own data)
    driver_file_list = sorted([f for f in os.listdir(DRIVERS_DIR) if ".nc" in f])
    if driver_file_list == 0:
        print("No driver file found in the DRIVERS_DIR: {}".format(DRIVERS_DIR))
    print("{} drivers files found in the DRIVERS_DIR: {}".format(len(driver_file_list), DRIVERS_DIR))

    daily_data_dict={"sitename":[], "ce_opt":[]}

    for f in tqdm(driver_file_list):
        sitename = f.split("_")[0]
        driver_ds, lat, train_sel, ef_train, ef_bound, doy_wet_train, t_max_wet_train, t_min_wet_train, lai_wet_train, rad_wet_train, ca_wet_train = get_training_data(DRIVERS_DIR, EF_PERCENTILE, f)
        
        # using grid search should be sufficient, as the optimization problem 
        # here is only along a single dimension
        # search between the minimum and maximum ce range, with a discretization of 500 steps
        ce_grid, loss_grid = ce_grid_search(ACM_V, driver_ds, lat, train_sel, ef_train, ef_bound, doy_wet_train, t_max_wet_train, t_min_wet_train, lai_wet_train, rad_wet_train, ca_wet_train)

        loss_grid = jnp.array(loss_grid)
        ce_opt = ce_grid[np.argmin(loss_grid)].item()
        daily_data_dict["sitename"].append(sitename)
        daily_data_dict["ce_opt"].append(ce_opt)

    # format as a pandas dataframe
    daily_ce_df = pd.DataFrame(daily_data_dict)

    # save to file
    daily_ce_df.to_csv(OUTPUT_FILENAME, index=False)
    print("Precalibrated ce values saved to : {}".format(OUTPUT_FILENAME))


if __name__ == "__main__":
    main()