from collections import namedtuple
import jax.numpy as jnp
import numpy as np

DALEC990ParamBounds = namedtuple("DALEC990ParamBounds", ["decomposition_rate",
                           "f_gpp",
                           "f_fol",
                           "f_root",
                           "leaf_lifespan",
                           "tor_wood",
                           "tor_root",
                           "tor_litter",
                           "tor_som",
                           "Q10",
                           "canopy_efficiency",
                           "Bday",
                           "flab",
                           "clab_release_period",
                           "Fday",
                           "leaf_fall_period",
                           "LCMA",
                           "Clab",
                           "Cfol",
                           "Croot",
                           "Cwood",
                           "Clitter",
                           "Csom",
                           "uWUE",
                           "PAW_Qmax",
                           "field_capacity",
                           "wilting_point_frac",
                           "initial_PAW",
                           "foliar_cf",
                           "ligneous_cf",
                           "dom_cf",
                           "resilience",
                           "lab_lifespan",
                           "moisture_factor",
                           "h2o_xfer",
                           "PUW_Qmax",
                           "initial_PUW",
                           "boese_r"])




dalec990_parmin_arr = jnp.array([1.0e-5, 0.2e0, 0.01e0, 0.01e0, 1.001e0, 2.5e-5,
                                 0.0001e0, 0.0001e0, 1.0e-7, 0.018e0, 5.0e0, 365.25,
                                 0.01e0, 30.4375e0, 365.25, 30.4375e0, 5.0e0, 1.0e0,
                                 1.0e0, 1.0e0, 1.0e0, 1.0e0, 1.0e0, 0.5, 1.0e0, 1.0e0, 0.01e0,
                                 1.0e0, 0.01e0, 0.01e0, 0.01e0, 0.01e0, 1.001e0, 0.01e0,
                                 0.01e0, 1e0, 1e0, 0.01], dtype=jnp.float32)
    

dalec990_parmin = DALEC990ParamBounds(*dalec990_parmin_arr)

dalec990_param_parmin = dalec990_parmin_arr[jnp.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,24,25,26,27,29,30,31,32,33,34,35,36,38], dtype=jnp.int32)-1]




dalec990_parmax_arr = jnp.array([0.01e0, 0.8e0, 0.5e0, 1.0e0, 8.0e0, 0.001e0, 0.01e0, 0.01e0,
                                 0.001e0, 0.08e0, 50.0e0, 365.25*4, 0.5e0, 100.0e0, 365.25*4,
                                 150.0e0, 200.0e0, 2000.0e0, 2000.0e0, 2000.0e0, 100000.0e0,
                                 2000.0e0, 200000.0e0, 30.0e0, 100000.0e0, 10000.0e0, 0.5e0, 10000.0e0,
                                 1.0e0, 1.0e0, 1.0e0, 1.0e0, 8.0e0, 1.0e0, 1.0e0, 100000.0e0, 
                                 10000.0e0, 0.3e0],
                       dtype=jnp.float32)
          
          
dalec990_parmax = DALEC990ParamBounds(*dalec990_parmax_arr)

dalec990_param_parmax = dalec990_parmax_arr[jnp.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,24,25,26,27,29,30,31,32,33,34,35,36,38], dtype=jnp.int32)-1]

dalec990_pool_parmin = dalec990_parmin_arr[jnp.array([18,19,20,21,22,23,28,37], dtype=jnp.int32)-1]
dalec990_pool_parmax = dalec990_parmax_arr[jnp.array([18,19,20,21,22,23,28,37], dtype=jnp.int32)-1]

DALEC990Outputs = namedtuple("DALEC990Outputs", ["lai", "gpp", "ET", "temperate",
                                                  "respiration_auto", "leaf_production",
                                                    "labile_production", "root_production", "wood_production", "lff", "lrf",
                                                      "labile_release", "leaf_litter", "wood_litter", "root_litter",
                                                        "respiration_hetero_litter", "respiration_hetero_som", "litter_to_som", "q_paw",
                                                          "q_puw", "paw2puw", "nee", "next_labile_pool", "next_foliar_pool",
                                                                    "next_root_pool", "next_wood_pool", "next_litter_pool", "next_som_pool",
                                                                      "next_paw_pool", "next_puw_pool", "beta", "violation"])


dalec990_pfn = DALEC990Outputs(*jnp.arange(32))