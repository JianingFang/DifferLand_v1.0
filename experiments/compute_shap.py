import jax
import jax.numpy as jnp
from functools import partial
import os
import numpy as np
import pandas as pd
import pickle
import xarray as xr
import shap
import sys
import argparse

# Import the hybrid carbon model components needed to compute the SHAP values
sys.path.insert(1, '..')
from DifferLand.model.DALEC990 import DALEC990
from DifferLand.util.preprocessing import get_train_test_sel, generate_met_matrix
from DifferLand.optimization.forward import parameter_prediction_forward
from DifferLand.model.auxi.ACM import ACM

DRIVERS_DIR = "../drivers/"
OUTPUT_DIR = "./output/"
SHAP_DIR = "./shap/"

def create_folder_if_not_exists(folder_path, verbose=False):
    """Create a folder if the folder doesn't exist"""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        if verbose:
            print("Created folder: {}".format(folder_path))
    else:
        if verbose:
                print("Folder path exists: {}".format(folder_path))
    return

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(prog = 'HAP Calculation',
                                 description = 'Run the script to derive the SHAP values for ET and GPP.')

    parser.add_argument("-n", "--sitename", help="Enter the SITE_ID of FLUXNET site, e.g. US-Var",  metavar="SITE_ID", required=True)
    parser.add_argument("-d", "--driver_name", help="Enter the name of the driver file", metavar="FILENAME.nc", required=True)
    parser.add_argument("-c", "--model_configuration", metavar="CONFIG_IDX", help="Enter the index for model configuration, range from 1-6." 
                        + "(1) baseline; (2) beta-JS; (3) beta-nn; (4) GPP&ET(NN)_MET; (5) GPP&ET(NN)_MET+LAI; (6) GPP(ACM)_ET(NN)",
                         type=int, required=True)
    parser.add_argument("-i", "--run_index", metavar="RUN_IDX", help="Enter an index for the run, used to identify independent calibrations", type=int, required=True)
    parser.add_argument("-s", "--silent", help="Silence message printing in the terminal.", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    return args

def get_stress_type(model_configuration):
    """Get stress type str for the model_configuration index"""
    if model_configuration == 1:
        return "baseline"
    elif model_configuration == 2:
        return "default"
    elif model_configuration == 3:
        return "nn_paw"
    elif model_configuration == 4:
        return "nn_whole_no_lai"
    elif model_configuration == 5:
        return "nn_whole"
    elif model_configuration == 6:
        return "gpp_acm_et_nn"
    else:
        print("ERROR: Model type must be between 1-6:")
        print("1) baseline")
        print("2) JS-beta")
        print("3) nn-beta")
        print("4) GPP&ET(NN)_MET")
        print("5) GPP&ET(NN)_MET+LAI")
        print("6) GPP(ACM)_ET(NN)")
        sys.exit(1)

def get_X_matrix_for_shap(met_matrix, output_matrix):
    """Return a pandas dataframe containing the variables needed to compute SHAP values """
    lat = met_matrix[:, 9]
    doy = met_matrix[:, 5]
    t_max = met_matrix[:, 2]
    t_min = met_matrix[:, 1]
    lai = output_matrix[:, 0]
    rad = met_matrix[:, 3]
    ca = met_matrix[:, 4]
    vpd = met_matrix[:, 7]
    paw_pool = output_matrix[:, 28]
    
    return pd.DataFrame({"lat":lat, "doy":doy, "t_max":t_max, "t_min":t_min, "lai":lai, "rad":rad, "ca":ca, "vpd":vpd,
                        "paw_pool":paw_pool})

def predict_gpp_and_et(X, dalec_parameters, gpp_params, water_stress_type, var_type, train_sel, mean_and_stds):
    """Predicted GPP and ET using various model configurations"""
    ce = dalec_parameters[10]  # Canopy Efficiency (time invariant)
    uWUE = dalec_parameters[17]  # IWUE: GPP*VPD/ET: gC/kgH2o *hPa
    wilting_point_frac = dalec_parameters[20]
    boese_r = dalec_parameters[29]
    field_capacity = dalec_parameters[19]
    
    lat = X[:, 0]
    doy = X[:, 1]
    t_max = X[:, 2]
    t_min = X[:, 3]
    lai = X[:, 4]
    rad = X[:, 5]
    ca = X[:, 6]
    vpd = X[:, 7]
    paw_pool = X[:, 8]
    
    mean_temp, std_temp, mean_solar, std_solar, mean_vpd, std_vpd, mean_ca, std_ca = mean_and_stds
    
    norm_temp = ((t_max + t_min)/2 - mean_temp) / std_temp
    norm_solar = (rad - mean_solar) / std_solar
    norm_vpd = (vpd - mean_vpd) / std_vpd
    norm_ca = (ca - mean_ca) / std_ca
    
    if water_stress_type == "baseline":
        beta = 1
        gpp = ACM(lat=lat, doy=doy, t_max=t_max, t_min=t_min, lai=lai, rad=rad, ca=ca, ce=ce) * beta
        ET = gpp * jnp.sqrt(vpd) / uWUE + rad * boese_r
    elif water_stress_type == "default":        
        wilting_point =field_capacity * wilting_point_frac
        beta = (paw_pool - wilting_point) / (field_capacity - wilting_point)
        beta = jnp.where(beta <=1, beta, 1)
        beta = jnp.where(beta >=0, beta, 0)
        gpp = ACM(lat=lat, doy=doy, t_max=t_max, t_min=t_min, lai=lai, rad=rad, ca=ca, ce=ce) * beta
        ET = gpp * jnp.sqrt(vpd) / uWUE + rad * boese_r
    elif water_stress_type == "nn_paw":
        beta = jax.nn.sigmoid(parameter_prediction_forward(gpp_params, jnp.array([paw_pool/1500,]).T)).squeeze()
        gpp = ACM(lat=lat, doy=doy, t_max=t_max, t_min=t_min, lai=lai, rad=rad, ca=ca, ce=ce) * beta
        ET = gpp * jnp.sqrt(vpd) / uWUE + rad * boese_r
    elif water_stress_type == "nn_whole":
        beta = -9999
        result = parameter_prediction_forward(gpp_params, jnp.array([norm_temp, lai/8, norm_solar,  paw_pool/1500, norm_vpd, norm_ca]).T)
        raw_gpp = result[:, 0]
        raw_ET = result[:, 1]
        gpp = jnp.maximum(0.01 * raw_gpp , raw_gpp)
        ET = jnp.maximum(0.01 * raw_ET, raw_ET)
    elif water_stress_type == "nn_whole_no_lai":
        beta = -9999
        result = parameter_prediction_forward(gpp_params, jnp.array([norm_temp, norm_solar,  paw_pool/1500, norm_vpd, norm_ca]).T)
        raw_gpp = result[:, 0]
        raw_ET = result[:, 1]
        gpp = jnp.maximum(0.01 * raw_gpp , raw_gpp)
        ET = jnp.maximum(0.01 * raw_ET, raw_ET)
    elif water_stress_type == "gpp_acm_et_nn":
        beta = -9999
        beta_params, et_params = gpp_params
        beta = jax.nn.sigmoid(parameter_prediction_forward(beta_params, jnp.array([paw_pool/1500,]).T)).squeeze()
        gpp = ACM(lat=lat, doy=doy, t_max=t_max, t_min=t_min, lai=lai, rad=rad, ca=ca, ce=ce) * beta
        raw_ET = parameter_prediction_forward(et_params, jnp.array([norm_temp, lai/8, norm_solar,  paw_pool/1500, norm_vpd, norm_ca, gpp/8]).T)[:, 0]
        ET = jnp.maximum(0.01 * raw_ET, raw_ET)
    if var_type == "gpp":
        return gpp
    elif var_type == "ET":
        return ET

def start_message(verbose, sitename, driver_name, run_index, stress_type):
    if verbose:
        print("*"*70)
        print("This script computes SHAP values for GPP and ET")
        print("*"*70)

        print("Sitename: {}".format(sitename))
        print("Driver name: {}".format(driver_name))
        print("Run index: {}".format(run_index))
        print("PAW limitation type: {}".format(stress_type))
        print()

def get_driver_ds(sitename, driver_name):
    """Load driver data from NetCDF file"""
    try:
        driver_ds = xr.open_dataset(os.path.join(DRIVERS_DIR, driver_name))
    except FileNotFoundError:
        print("Driver file {} not found!".format(os.path.join(DRIVERS_DIR, driver_name)))
        sys.exit(1)

    return driver_ds

def load_output_pickle(sitename, run_index, stress_type):
    output_save_name = os.path.join(OUTPUT_DIR, "daily_{}_{}_{}.pickle".format(sitename, stress_type, run_index))
    try:
        with open(output_save_name, "rb") as fp:
            output_load = pickle.load(fp)
    except FileNotFoundError:
        print("ERROR: {} cannot be found".format(output_save_name))
        sys.exit(1)
    return output_load

def compute_mean_and_stds(train_sel, met_matrix_full):
    mean_temp = jnp.mean((met_matrix_full[train_sel, 1] + met_matrix_full[train_sel, 2])/2)
    std_temp = jnp.std((met_matrix_full[train_sel, 1] + met_matrix_full[train_sel, 2])/2)

    mean_solar = jnp.mean(met_matrix_full[train_sel, 14])
    std_solar = jnp.std(met_matrix_full[train_sel, 14])

    mean_vpd = jnp.mean(met_matrix_full[train_sel, 7])
    std_vpd = jnp.std(met_matrix_full[train_sel, 7])

    mean_ca = jnp.mean(met_matrix_full[train_sel, 4])
    std_ca = jnp.std(met_matrix_full[train_sel, 4])

    mean_and_stds = [mean_temp, std_temp, mean_solar, std_solar, mean_vpd, std_vpd, mean_ca, std_ca]
    return mean_and_stds

def main():
    args = parse_args()
    verbose = not args.silent

    sitename = args.sitename
    driver_name = args.driver_name
    model_configuration = args.model_configuration
    run_index = args.run_index
    stress_type = get_stress_type(model_configuration)
    start_message(verbose, sitename, driver_name, run_index, stress_type)
    create_folder_if_not_exists(SHAP_DIR)

    output_load = load_output_pickle(sitename, run_index, stress_type)

    param_state = output_load["param_state"]
    output_matrix_full = output_load["output_matrix_full"]
    param_initial_cal, pool_initial_cal, gpp_params_cal = param_state

    driver_ds = get_driver_ds(sitename, driver_name)

    train_sel, test_sel = get_train_test_sel(driver_ds)
    met_matrix_full = generate_met_matrix(driver_ds, train_sel, test_sel, train_mode=False)
    model = DALEC990(np.sum(train_sel), water_stress_type=stress_type,  reco=True)
    dalec_params = model.unnormalize(param_initial_cal)

    mean_and_stds = compute_mean_and_stds(train_sel, met_matrix_full)


    predict_gpp = jax.jit(partial(predict_gpp_and_et, dalec_parameters=dalec_params,
                                   gpp_params=gpp_params_cal, water_stress_type=stress_type,
                                     var_type="gpp", train_sel=train_sel, mean_and_stds=mean_and_stds))
    
    predict_et = jax.jit(partial(predict_gpp_and_et, dalec_parameters=dalec_params,
                                  gpp_params=gpp_params_cal, water_stress_type=stress_type,
                                    var_type="ET", train_sel=train_sel, mean_and_stds=mean_and_stds))

    X = get_X_matrix_for_shap(met_matrix_full, output_matrix_full)

    X100 = shap.utils.sample(X, 100)  # 100 instances for use as the background distribution

    if verbose:
        print("Computing SHAP values for GPP")
    explainer_gpp = shap.KernelExplainer(predict_gpp, X100)
    shap_values_gpp = explainer_gpp(X)
    dump_str_gpp = os.path.join(SHAP_DIR, "gpp_{}_{}_{}_shap.pickle".format(sitename, stress_type, run_index))
    with open(dump_str_gpp, "wb") as f:
        pickle.dump(shap_values_gpp, f)
    if verbose:
        print("GPP SHAP values saved to {}".format(dump_str_gpp))

    if verbose:
        print("Computing SHAP values for ET")
    explainer_et = shap.KernelExplainer(predict_et, X100)
    shap_values_et = explainer_et(X)
    dump_str_et = os.path.join(SHAP_DIR, "et_{}_{}_{}_shap.pickle".format(sitename, stress_type, run_index))
    with open(dump_str_et, "wb") as f:
        pickle.dump(shap_values_et, f)
    if verbose:
        print("ET SHAP values saved to {}".format(dump_str_et))


if __name__ == "__main__":
    main()
