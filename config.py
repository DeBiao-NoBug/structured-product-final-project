# ============================================================================
# config.py - Project-wide Configuration
# ============================================================================
"""
Centralized configuration for Quanto Derivative Pricing Project.

This file contains:
1. Random seed for reproducibility
2. Shared parameters across modules
3. File paths and constants

Usage in external modules:
    from config import RANDOM_SEED, set_seeds
    set_seeds()
"""

import numpy as np
import random

# ============================================================================
# RANDOM SEED CONFIGURATION
# ============================================================================

# Master random seed - DO NOT CHANGE after initial run
RANDOM_SEED = 42

def set_seeds(seed=RANDOM_SEED):
    """
    Set all random seeds for reproducibility.
    
    Call this function at the beginning of any .py module or function
    that performs stochastic simulation.
    
    Parameters:
    -----------
    seed : int
        Random seed (default: RANDOM_SEED from config)
    """
    np.random.seed(seed)
    random.seed(seed)
    return seed

# ============================================================================
# PROJECT PARAMETERS (Shared across modules)
# ============================================================================

# Simulation parameters
N_PATHS = 50000          # Number of Monte Carlo paths
N_TIME_STEPS = 252 * 3   # Time steps (daily for 3 years)
TRADING_DAYS_PER_YEAR = 252

# Hull-White parameters (will be estimated in Section 3)
HW_MEAN_REVERSION = None  # To be filled from estimation
HW_VOLATILITY = None      # To be filled from estimation

# Contract specifications
CONTRACT_MATURITY = 3.0   # Years
SETTLEMENT_LAG = 0.25     # Years

# Pricing date
PRICING_DATE = '2021-02-15'

# ============================================================================
# FILE PATHS
# ============================================================================

import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

# Create directories if they don't exist
for directory in [DATA_DIR, OUTPUT_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)
    
    
# # ============================================================================
# # 未来 OOP 封装时的用法示例: pricing_models.py - Pricing Model Classes
# # ============================================================================
# """
# Object-oriented implementation of pricing models for sensitivity analysis.
# """

# import numpy as np
# from config import RANDOM_SEED, set_seeds

# class HullWhiteModel:
#     """Hull-White short rate model for interest rate simulation."""
    
#     def __init__(self, a, sigma, theta_func, zero_curve_data, seed=RANDOM_SEED):
#         """
#         Initialize Hull-White model.
        
#         Parameters:
#         -----------
#         a : float
#             Mean reversion speed
#         sigma : float
#             Rate volatility
#         theta_func : callable
#             Time-dependent drift function
#         zero_curve_data : dict
#             Zero curve parameters from Section 2
#         seed : int
#             Random seed for reproducibility
#         """
#         self.a = a
#         self.sigma = sigma
#         self.theta_func = theta_func
#         self.zero_curve_data = zero_curve_data
#         self.seed = seed
        
#         # Set seed for this instance
#         set_seeds(self.seed)
    
#     def simulate_paths(self, N, T, dt, r0):
#         """
#         Simulate N paths of short rate r(t).
        
#         This method will reset the random seed before each call
#         to ensure reproducibility.
#         """
#         # Reset seed before simulation
#         np.random.seed(self.seed)
        
#         # Simulation code here...
#         pass

# # Usage in sensitivity analysis:
# # model = HullWhiteModel(a=0.05, sigma=0.01, ..., seed=42)
# # paths = model.simulate_paths(...)  # Always gives same result