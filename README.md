# DifferLand v1.0
**Authors**: Jianing Fang (jf3423@columbia.edu), Pierre Gentine (pg2328@columbia.edu)

This repository contains the implementation of a JAX-based, automatically differentiable hybrid terrestrial biosphere model used in the manuscript "Exploring Optimal Complexity for Soil Water Stress Representation: A Hybrid-Machine Learning Approach." The code enables the reader to reproduce the results in the manuscript. We encourage readers to adapt the code to suit their own modeling needs. We are happy to discuss model details and collaboration opportunities.

## Project Structure:

- `./DifferLand`: This directory contains the source code for DifferLand.
    - `model`: Here lies the JAX-based implementation of the DALEC990 model, which is automatically differentiable. The DALEC990 model offers six different configurations for representing the influence of soil water availability on GPP and ET. It is designed to be easily adaptable and extendable to meet diverse modeling requirements.
    - `optimization`: code related to optimization and computing loss function terms
    - `util`: code to assist parameter initialization, normalization, data preprocessing, and output visualization
- `./notebooks`: This folder includes notebooks for analyzing and interpreting results, as well as generating figures for our manuscript.
- `./wavelet`: MATLAB scripts for generating wavelet coherence analysis figures presented in our manuscript. These scripts require the [MATLAB wavelet package](https://www.mathworks.com/products/wavelet.html) for execution.
- `./experiments`: Contains scripts for canopy efficiency precalibration, model calibration, and SHAP value computation.
- `./drivers`: This directory stores drivers for the 16 sites, including target variables.
- `./output_nc`: For ease of analysis, we provide a NetCDF file containing the output of the best fitting run (evaluated on the training period) for each site and model configuration in this directory. Users can generate additional NetCDF files for other runs by adapting the script provided in the accompanying notebook.

## Running the Code:

0) Prepare the model driver files containing the forcings and target variables. The driver files utilized in our experiment are stored in the `./drivers` folder. Detailed information regarding data preprocessing can be found in our manuscript.
1) Navigate to the project directory. Create a conda environment by executing `conda env create -f environment.yml` to install all dependencies. Then activate the environment by typing `conda activate DifferLand`
2) For ACM-based models, precalibration of canopy efficiency parameters is necessary to represent well-watered conditions, defined as days where the evaporative fraction exceeds the 80th percentile during the training period. To initiate this process, navigate to the `experiments` folder, and run the script with `python ce_precalibration.py`.
3) With precalibration completed, proceed to calibrate the model. Within the `./experiments` folder, locate the script `calibration.py` to begin the calibration process.

    Type `python calibration.py --help` to see available options.
    ```
    usage: DifferLand Calibration [-h] -n SITE_ID -d FILENAME.nc -c CONFIG_IDX -i RUN_IDX [-s | --silent | --no-silent]

    Run the script to calibrate the physical and neural parameters in the DifferLand model.

    optional arguments:
    -h, --help            show this help message and exit
    -n SITE_ID, --sitename SITE_ID
                            Enter the SITE_ID of FLUXNET site, e.g. US-Var
    -d FILENAME.nc, --driver_name FILENAME.nc
                            Enter the name of the driver file, e.g. US_Var_daily.nc
    -c CONFIG_IDX, --model_configuration CONFIG_IDX
                            Enter the index for model configuration, range from 1-6.(1) baseline; (2) beta-JS; (3) beta-nn; (4) GPP&ET(NN)_MET; (5) GPP&ET(NN)_MET+LAI; (6) GPP(ACM)_ET(NN)
    -i RUN_IDX, --run_index RUN_IDX
                            Enter an index for the run, used to identify independent calibrations
    -s, --silent, --no-silent
                            Silence message printing in the terminal. (default: False)
    ```

    Example: `python calibration.py -n US-Var -d US-Var.nc -c 1 -i 1`

    You should get messages in the terminal that looks like this (use `-s` option to prevent message printing):
    ```
    **********************************************************************  
    Welcome to DifferLand!  
    an automatically differentiable hybrid terrestrial biosphere model
    **********************************************************************  
    Sitename: US-Var  
    Driver name: US-Var.nc  
    Run index: 1  
    PAW limitation type: baseline  
    LEARNING RATE: 0.0005  
    TOTAL ITERATIONS: 25000  
  
    Created folder: ./fig/  
    Created folder: ./log/  
    Created folder: ./output/  
    Precalibrated ce_opt: 38.547  
    Loading datasets for model calibration...  
    Training log will be stored in: ./log/daily_US-Var_baseline_1.log  
    Calibration START!  
    Parameters initialized  
    iter 1, loss: 515.977, test nnse: 0.325  
    Numerical issue encountered, trying to reinitialize the model...  
  
    Parameters initialized  
    iter 1, loss: 112.135, test nnse: 0.655  
    Numerical issue encountered, trying to reinitialize the model...  
  
    Parameters initialized  
    iter 1, loss: 31.525, test nnse: 1.759  
    iter 1001, loss: 43.610, test nnse: 1.584  
    iter 2001, loss: 8.111, test nnse: 1.537  
    iter 3001, loss: 0.509, test nnse: 1.520  
    iter 4001, loss: -0.376, test nnse: 1.515  
    iter 5001, loss: -0.706, test nnse: 1.510  
    iter 6001, loss: -0.815, test nnse: 1.508  
    iter 7001, loss: -0.834, test nnse: 1.508  
    iter 8001, loss: -0.844, test nnse: 1.509  
    iter 9001, loss: -0.852, test nnse: 1.510  
    iter 10001, loss: -0.863, test nnse: 1.512  
    iter 11001, loss: -0.886, test nnse: 1.516  
    iter 12001, loss: -1.011, test nnse: 1.525  
    iter 13001, loss: -1.204, test nnse: 1.540  
    iter 14001, loss: -1.307, test nnse: 1.565  
    iter 15001, loss: -1.382, test nnse: 1.601  
    iter 16001, loss: -1.459, test nnse: 1.660  
    iter 17001, loss: -1.543, test nnse: 1.732  
    iter 18001, loss: -1.634, test nnse: 1.826  
    iter 19001, loss: -1.728, test nnse: 1.906  
    iter 20001, loss: -1.805, test nnse: 1.961  
    iter 21001, loss: -1.870, test nnse: 2.019  
    iter 22001, loss: -1.927, test nnse: 2.071  
    iter 23001, loss: -1.974, test nnse: 2.115  
    iter 24001, loss: -2.011, test nnse: 2.145  
    Training completed!  
    Calibrated figure saved to: ./fig/daily_US-Var_baseline_1.pdf  
    Calibrated parameters, performance metrics, and model output dict saved to: ./output/daily_US-Var_baseline_1.pickle  
    ```
    Now your model output is saved as a .pickle file, and you can check the predicted ET, GPP, RECO, and LAI in the figure.

    If you don't want to run the code yourself, the output from out training experiments is hosted in this Zenodo repository (will be available upon manuscript publication).
4) Compute SHAP values with the script `./experiments/compute_shap.py`. Use `-h` to access the help messages.
5) Analyze and interpret the results using the Jupyter notebook `./notebooks/analyze_local_edc_daily.ipynb` This is the notebook we used to generate the figures in our paper. You may add your own code to interpret the results.

## Model Configurations:
Please refer to the manuscript for further details, the Table 1 is reproduced here for a quick reference
| No | Setup             | GPP Form                         | ET Form                                                  | Effective Trainable Parameters |
|----|-------------------|----------------------------------|----------------------------------------------------------|--------------------------------|
| 1  | baseline          | GPP = 1 × GPPACM                | ET = (GPP × √VPD) / uWUE + SSRD × r                      | 24 physical                    |
| 2  | β-JS              | GPP = JS-βt × GPPACM            | ET = (GPP × √VPD) / uWUE + SSRD × r                      | 26 physical                    |
| 3  | β-nn              | GPP = nn(PAW) × GPPACM          | ET = (GPP × √VPD) / uWUE + SSRD × r                      | 24 physical + 31 nn            |
| 4  | GPP&ET(NN)_MET    | GPP, ET = nn(Ta, SSRD, VPD, PAW, CO2) |                                                      | 21 physical + 192 nn           |
| 5  | GPP&ET(NN)_MET+LAI| GPP, ET = nn(Ta, SSRD, LAI, VPD, PAW, CO2) |                                                  | 21 physical + 202 nn           |
| 6  | GPP(ACM)_ET(NN)   | GPP = nn(PAW) × GPPACM          | ET = nn(Ta, SSRD, LAI, VPD, PAW, GPP, CO2)               | 24 physical + 31 nn + 201 nn  |



## Acknowledgements:
We would like to express our special thanks to Dr. Nuno Carvalhais and members of the Model-Data Integration group at the Max Planck Institute for Biogeochemistry (MPI-BGC) for the many insightful discussions related to this work during J.F.’s academic visit at MPI-BGC. We would also like to thank Dr. Xu Lian for providing the Aridity Index climatology data from TerraClimate, and Dr. Anthony Bloom for the advice on incorporating the ecological & dynamical constraints. J.F. and P.G. are supported by the NASA ROSES-22 Future Investigators in NASA Earth and Space Science and Technology (FINESST) Program (Grant Number: 80NSSC24K0023). This work was also funded by the Land Ecosystem Models based On New Theory, obseRvations, and ExperimEnts (LEMONTREE) project (UREAD 1005109-LEMONTREE), the European Research Council grant USMILE (ERC CU18-3746), and National Science Foundation Science and Technology Center LEAP, Learning the Earth with Artificial intelligence and Physics (AGS-2019625). 

## Citation:

Fang, J., Gentine, P. (2024). Exploring Optimal Complexity for Soil Water Stress Representation: A Hybrid-Machine Learning Approach. Manuscript submitted for publication.

*Our manuscript has been submitted to Journal of Advances in Modeling Earth Systems (JAMES). The citation will be updated upon acceptace.*