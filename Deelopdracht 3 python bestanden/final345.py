# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 18:18:55 2025

@author: Janal
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq # NEW: Added for high-precision root finding

# --- 1. Setup Paths and Constants ---
# Note: Ensure these paths are correct for your local machine
path = "C:/Users/Janal/Documents/Opdrachti&v/data/"
json_file_2 = path + "TankData_Gr98_V1.0.json"
json_file_1 = path + "MainShipParticulars_Gr98_V1.0.json"

csv_file_2 = path + "Tank1_Diagram_Volume_Gr98_V1.0.csv"
csv_file_3 = path + "Tank2_Diagram_Volume_Gr98_V1.0.csv"
csv_file_4 = path + "Tank3_Diagram_Volume_Gr98_V1.0.csv"
csv_file_1 = path + "HullAreaData_Gr98_V1.0.csv"
csv_file_5 = path + "TankBHD_Data_Gr98_V1.0.csv"

# General Constants
THICKNESSHULL = 0.02
STEELWEIGHT = 7850
WATERWEIGHT = 1025
G = -9.81
THICKNESSBHD = 0.01
TOLERANCE = 1e-5

# Crane and Load Constants (NECESSARY FOR MOMENT CALCULATION)
PHijsGerei = 0.06
PLast = 1 - PHijsGerei # 0.94
MLast = 230000.0 * G
PKraanHuis = 0.34
PKraanBoom = 0.17
LKraanBoom = 32.5
DiameterBuis = 8
BreedteSchip = 20
FACTOR =2.1

# --- 2. Reading JSON Files ---
try:
    with open(json_file_1, 'r') as f:
        main_ship_data = json.load(f)
    print(f"Successfully read {json_file_1}")

    with open(json_file_2, 'r') as f:
        tank_data = json.load(f)
    print(f"Successfully read {json_file_2}")

except FileNotFoundError as e:
    print(f"Error reading JSON file: {e}")
except json.JSONDecodeError as e:
    print(f"Error decoding JSON: {e}")

print("\n" + "="*30 + "\n")

# --- 3. Reading CSV Files ---
try:
    # Files with 1 info line before header (skiprows=1)
    # Reading Hull Area and Tank Bulkhead Data
    hull_area_df = pd.read_csv(csv_file_1, skiprows=1, skipinitialspace=True)
    tank_bhd_df = pd.read_csv(csv_file_5, skiprows=1, skipinitialspace=True)
    
    print(f"Successfully read {csv_file_1}")
    print(f"Successfully read {csv_file_5}")
    tank1_diag_df = pd.read_csv(csv_file_2, skiprows=2, skipinitialspace=True)
    tank2_diag_df = pd.read_csv(csv_file_3, skiprows=2, skipinitialspace=True)
    tank3_diag_df = pd.read_csv(csv_file_4, skiprows=2, skipinitialspace=True)

    # Files with 2 info lines before header (skiprows=2)
    # Reading Tank Diagram Data


except FileNotFoundError as e:
    print(f"Error reading CSV file: {e}")
    raise
except Exception as e:
    print(f"An error occurred with pandas: {e}")
    raise

def clean_array(arr):
    
    arr = np.asarray(arr) # Ensure the input is a NumPy array
    
    # Use boolean indexing to identify and replace values below the tolerance
    # We use np.abs() to handle both small positive and small negative numbers
    arr[np.abs(arr) < TOLERANCE] = 0.0
    
    return arr
print("\n" + "="*30 + "\n")
def find_variables_by_keyword(keyword):
    """
    Prints the name and value of all variables in the current global scope
    whose names contain the specified keyword (case-insensitive).

    Parameters:
    keyword (str): The substring to search for in variable names.
    """
    
    # Access the global symbol table (dictionary of all defined global variables)
    global_vars = globals()
    
    # Convert the search keyword to lowercase once
    search_term = keyword.lower()
    
    found_count = 0
    
    # Iterate through all variable names and values
    for name, value in global_vars.items():
        if search_term in name.lower():
            # Found a match! Print the result clearly
            print("--- Found Match ---")
            print(f"Name: {name}")
            print(f"Type: {type(value).__name__}")
            
            # Print the value, limiting output for long items like DataFrames
            if isinstance(value, (list, tuple, np.ndarray)) and len(value) > 10:
                print(f"Value: Array/List with {len(value)} elements.")
            elif isinstance(value, pd.DataFrame):
                print(f"Value: DataFrame ({value.shape[0]} rows, {value.shape[1]} columns)")
            else:
                print(f"Value: {value}")
            
            found_count += 1

    if found_count == 0:
        print(f"No variables found containing the keyword: '{keyword}'")
# ================================
# TANK 1 processing
# ================================
# x_tank1: The shared x-axis for this tank (Filling Percentage)
x_tank1 = tank1_diag_df['Tankfilling [% of h_tank]'].to_numpy()

# 1. Interpolator: Filling Height [m] (from %)
y_h1 = tank1_diag_df['Tankfilling [m]'].to_numpy()
cubic_interp_h1 = CubicSpline(x_tank1, y_h1)

# 2. Interpolator: Volume [m3]
y_vol1 = tank1_diag_df['Tankvolume [m3]'].to_numpy()
cubic_interp_vol1 = CubicSpline(x_tank1, y_vol1)

# 3. Interpolator: VCG [m]
y_vcg1 = tank1_diag_df['vcg [m]'].to_numpy()
cubic_interp_vcg1 = CubicSpline(x_tank1, y_vcg1)

# 4. Interpolator: LCG [m]
y_lcg1 = tank1_diag_df['lcg [m]'].to_numpy()
cubic_interp_lcg1 = CubicSpline(x_tank1, y_lcg1)

# 5. Interpolator: TCG [m]
y_tcg1 = tank1_diag_df['tcg [m]'].to_numpy()
cubic_interp_tcg1 = CubicSpline(x_tank1, y_tcg1)


# ================================
# TANK 2 processing
# ================================
# x_tank2: The shared x-axis for this tank (Filling Percentage)
x_tank2 = tank2_diag_df['Tankfilling [% of h_tank]'].to_numpy()

# 1. Interpolator: Filling Height [m] (from %)
y_h2 = tank2_diag_df['Tankfilling [m]'].to_numpy()
cubic_interp_h2 = CubicSpline(x_tank2, y_h2)

# 2. Interpolator: Volume [m3]
y_vol2 = tank2_diag_df['Tankvolume [m3]'].to_numpy()
cubic_interp_vol2 = CubicSpline(x_tank2, y_vol2)

# 3. Interpolator: VCG [m]
y_vcg2 = tank2_diag_df['vcg [m]'].to_numpy()
cubic_interp_vcg2 = CubicSpline(x_tank2, y_vcg2)

# 4. Interpolator: LCG [m]
y_lcg2 = tank2_diag_df['lcg [m]'].to_numpy()
cubic_interp_lcg2 = CubicSpline(x_tank2, y_lcg2)

# 5. Interpolator: TCG [m]
y_tcg2 = tank2_diag_df['tcg [m]'].to_numpy()
cubic_interp_tcg2 = CubicSpline(x_tank2, y_tcg2)

x_tank3 = tank3_diag_df['Tankfilling [% of h_tank]'].to_numpy()

# 1. Interpolator: Filling Height [m] (from %)
# Note: Assuming 'Tankfilling [m]' column exists in Tank3 file similar to others
if 'Tankfilling [m]' in tank3_diag_df.columns:
    y_h3 = tank3_diag_df['Tankfilling [m]'].to_numpy()
    cubic_interp_h3 = CubicSpline(x_tank3, y_h3)
else:
    print("Warning: 'Tankfilling [m]' column missing in Tank 3 Data")

# 2. Interpolator: Volume [m3]
y_vol3 = tank3_diag_df['Tankvolume [m3]'].to_numpy()
cubic_interp_vol3 = CubicSpline(x_tank3, y_vol3)

# 3. Interpolator: VCG [m]
y_vcg3 = tank3_diag_df['vcg [m]'].to_numpy()
cubic_interp_vcg3 = CubicSpline(x_tank3, y_vcg3)

# 4. Interpolator: LCG [m]
y_lcg3 = tank3_diag_df['lcg [m]'].to_numpy()
cubic_interp_lcg3 = CubicSpline(x_tank3, y_lcg3)

# 5. Interpolator: TCG [m]
y_tcg3 = tank3_diag_df['tcg [m]'].to_numpy()
cubic_interp_tcg3 = CubicSpline(x_tank3, y_tcg3)
# --- 4. Data Processing: Interpolators for Tanks ---
# We organize variables by Tank ID to ensure the correct x-axis is used for each.

# ================================
# TANK 1 processing
# ================================

# weights in Newton Hull Area

TransomArea = hull_area_df.loc[0, 'Area [m2]'] * STEELWEIGHT * THICKNESSBHD * G * FACTOR
ShellArea = hull_area_df.loc[1, 'Area [m2]'] * STEELWEIGHT * THICKNESSHULL * G * FACTOR
DeckArea = hull_area_df.loc[2, 'Area [m2]'] * STEELWEIGHT * THICKNESSHULL * G * FACTOR
TOTAL_HULL = TransomArea + ShellArea + DeckArea

# moments from the hull
moments_hull_x = TransomArea * hull_area_df.loc[0, 'lca [m]'] + ShellArea * hull_area_df.loc[1, 'lca [m]'] + DeckArea * hull_area_df.loc[2, 'lca [m]']
moments_hull_y = TransomArea * hull_area_df.loc[0, 'tca [m]'] + ShellArea * hull_area_df.loc[1, 'tca [m]'] + DeckArea * hull_area_df.loc[2, 'tca [m]']
moment_vector_hull = clean_array(np.array([moments_hull_x, moments_hull_y]))

# weights in Newron Tank area (exclude tank 2)
filtered_bhd_df = tank_bhd_df[tank_bhd_df['tcg [m]'] != 0].reset_index(drop=True)
bhd78_df = tank_bhd_df[tank_bhd_df['tcg [m]'] == 0].reset_index(drop=True)
# --- 2. Calculate Weights in Newton (Explicit Indexing) ---
# .iloc[i, 0] accesses the 'BHD Area [m2]' column (index 0)

BHD1_Weight = filtered_bhd_df.iloc[0, 0] * STEELWEIGHT * THICKNESSBHD * G * FACTOR
BHD2_Weight = filtered_bhd_df.iloc[1, 0] * STEELWEIGHT * THICKNESSBHD * G* FACTOR
BHD3_Weight = filtered_bhd_df.iloc[2, 0] * STEELWEIGHT * THICKNESSBHD * G* FACTOR
BHD4_Weight = filtered_bhd_df.iloc[3, 0] * STEELWEIGHT * THICKNESSBHD * G* FACTOR
BHD5_Weight = filtered_bhd_df.iloc[4, 0] * STEELWEIGHT * THICKNESSBHD * G* FACTOR
BHD6_Weight = filtered_bhd_df.iloc[5, 0] * STEELWEIGHT * THICKNESSBHD * G* FACTOR
BHD7_Weight = bhd78_df.iloc[0, 0] * STEELWEIGHT * THICKNESSBHD * G* FACTOR
BHD8_Weight = bhd78_df.iloc[1, 0] * STEELWEIGHT * THICKNESSBHD * G* FACTOR
# --- 3. Calculate Total Moments (Explicit Summation) ---
# Note: LCG is column index 1, TCG is column index 2.
BHD_TOTAL = BHD1_Weight +BHD2_Weight +BHD3_Weight +BHD4_Weight +BHD5_Weight +BHD6_Weight +BHD7_Weight+BHD8_Weight
# Total Moment about the X-axis (Mx) -> Calculated using LCG ('lcg [m]', index 1)
moments_bhd_x = (
    BHD1_Weight * filtered_bhd_df.iloc[0, 1] +
    BHD2_Weight * filtered_bhd_df.iloc[1, 1] +
    BHD3_Weight * filtered_bhd_df.iloc[2, 1] +
    BHD4_Weight * filtered_bhd_df.iloc[3, 1] +
    BHD5_Weight * filtered_bhd_df.iloc[4, 1] +
    BHD6_Weight * filtered_bhd_df.iloc[5, 1]
)

# Total Moment about the Y-axis (My) -> Calculated using TCG ('tcg [m]', index 2)
moments_bhd_y = (
    BHD1_Weight * filtered_bhd_df.iloc[0, 2] +
    BHD2_Weight * filtered_bhd_df.iloc[1, 2] +
    BHD3_Weight * filtered_bhd_df.iloc[2, 2] +
    BHD4_Weight * filtered_bhd_df.iloc[3, 2] +
    BHD5_Weight * filtered_bhd_df.iloc[4, 2] +
    BHD6_Weight * filtered_bhd_df.iloc[5, 2]
)

# --- 4. Final Array ---
moment_vector_bhd = clean_array(np.array([moments_bhd_x, moments_bhd_y]))


#calculating total force upward

buoyant_volume = main_ship_data["VOLUME RELATED DATA (MOULDED)"]["Buoyant_Volume_m3"]
force_upward = -buoyant_volume* WATERWEIGHT *G
moment_vector_force_upward = clean_array(np.array(main_ship_data["VOLUME RELATED DATA (MOULDED)"]["COB_m"])[:-1]*force_upward)
# calculating tank 3

buoyant_volume = tank_data["WB TANK 3"]["Volume_water_ballast_m3"]
force_water_tank3 = buoyant_volume* WATERWEIGHT *G
moment_vector_tank_3 = clean_array(np.array(tank_data["WB TANK 3"]["COV_WB_m"])[:-1]*force_water_tank3) 
print(moment_vector_tank_3)
'''
buoyant_volume = cubic_interp_vol3(72)
force_water_tank3 = buoyant_volume* WATERWEIGHT *G
moment_vector_tank_3 = clean_array(np.array([cubic_interp_lcg3(72)*force_water_tank3,cubic_interp_tcg3(72)*force_water_tank3]))
'''
# --- START: Crane/Deklading Moment Calculation Block ---

# --- 1. Intermediate Mass Calculations (from DEEL 1) ---
SwlMax = MLast / PLast
MHijsGerei = PHijsGerei * SwlMax 
MKraanHuis = PKraanHuis * SwlMax 
MKraanBoom = PKraanBoom * SwlMax 
TotaleMassaKraanMetLast = (MKraanBoom + MKraanHuis + MHijsGerei + MLast)

# --- 2. Intermediate Geometric Calculations (from DEEL 2) ---

LKraanBoom60 = LKraanBoom * np.cos(np.radians(60))
XPosKraanBoom = LKraanBoom60/2 + 7.5

# Transverse Position (Y-coordinate for all components)
YPosKraanCenterline = 31

# --- 3. Overall LCG Calculation (from DEEL 3) ---
XPosZwaartepuntKraan = (
    MKraanHuis * 7.5 +
    MKraanBoom * XPosKraanBoom +
    (MLast + MHijsGerei) * (LKraanBoom60 + 7.5)
) / (TotaleMassaKraanMetLast)

# --- 4. MOMENT CALCULATION (M = r x F) ---

# --- Calculate TOTAL MOMENT VECTOR for Crane Assembly (Crane + Payload) ---


# Mx_Total (Transverse Moment) = -G * Y_Pos * Sum(Mass)
Mx_TOTAL_KRAAN = YPosKraanCenterline * TotaleMassaKraanMetLast

# My_Total (Longitudinal Moment) = G * Sum(Mass * X)
My_TOTAL_KRAAN = XPosZwaartepuntKraan * TotaleMassaKraanMetLast

MOMENT_VECTOR_KRAAN_TOTAAL = clean_array(np.array([Mx_TOTAL_KRAAN, My_TOTAL_KRAAN]))


# --- Calculate MOMENT VECTOR for Deklading (Max Load only) ---
X_Pos_DEKLADING = -2.5
M_DEKLADING =  MLast * 4 
Y_Pos_DEKLADING = YPosKraanCenterline

# Mx_Deklading (Transverse Moment) = -G * (Mass * Y_Pos)
Mx_DEKLADING = Y_Pos_DEKLADING * M_DEKLADING

# My_Deklading (Longitudinal Moment) = G * (Mass * X_Pos)
My_DEKLADING = X_Pos_DEKLADING * M_DEKLADING

MOMENT_VECTOR_DEKLADING = clean_array(np.array([Mx_DEKLADING, My_DEKLADING]))

# --- END: Crane/Deklading Moment Calculation Block ---
# total mass
total_force_downward_1and2 = TotaleMassaKraanMetLast + M_DEKLADING + force_water_tank3 + BHD_TOTAL + TOTAL_HULL + force_upward

total_moment = MOMENT_VECTOR_DEKLADING + MOMENT_VECTOR_KRAAN_TOTAAL + moment_vector_tank_3 + moment_vector_hull+moment_vector_bhd+moment_vector_force_upward
#print
find_variables_by_keyword("moment")

# --- START: Tank 1 Transverse Moment Array Calculation ---

# 1. Define a common, dense X-axis (Filling percentage)
# This array is used to evaluate the splines smoothly.
X_evaluation_T1 = tank1_diag_df['Tankfilling [% of h_tank]'].to_numpy()
print(x_tank1.min(),x_tank1.max())
# 2. Evaluate both splines over the common X-range
# Y_Vol (Volume at % filling) and Y_TCG (TCG at % filling)
Y_Vol_T1 = cubic_interp_vol1(X_evaluation_T1)
Y_TCG_T1 = cubic_interp_tcg1(X_evaluation_T1)

# 3. Perform Vectorized Multiplication
# Transverse Moment (Mx) = Volume * TCG * Density * G
# This gives the Transverse Moment for Tank 1 as a function of filling level.
Mx_T1_array = Y_Vol_T1 * Y_TCG_T1 * WATERWEIGHT * G

# 4. Clean and store the final array
Mx_T1_array_cleaned = clean_array(Mx_T1_array)


# 1. Define the objective function f(X) = Mx(X) - TargetY
def moment_function_shifted(X_val, target_moment):
    # Calculate moment Mx at X_val using the CubicSplines
    moment_at_X = cubic_interp_vol1(X_val) * cubic_interp_tcg1(X_val) * WATERWEIGHT * G
    return moment_at_X - target_moment

target_y = -total_moment[1] # - 42500
a = x_tank1.min() 
b = x_tank1.max() 

try:
    found_x_value_brentq_T1 = brentq(moment_function_shifted, a, b, args=(target_y,))
    # Update the necessary variable for downstream use
    found_x_values_T1 = np.array([found_x_value_brentq_T1])
except ValueError:
    print("Warning: Cannot find unique solution for Tank 1 moment balance in [0, 100]. Setting filling to 0.")
    found_x_values_T1 = np.array([0.0]) # Handle case where target is out of bounds
    
print(f"Target Y (Mx Balance): {target_y}")
print(f"Found X (Tank 1 Filling %): {found_x_values_T1[0]:.4f}")

# --- Visualization (Optional) ---
# Plot the original data
plt.plot(X_evaluation_T1, Mx_T1_array_cleaned, 'o', label='Data points', alpha=0.3)
# Plot the smooth interpolation
x_fine = np.linspace(0, 99, 1000)
plt.plot(x_fine, moment_function_shifted(x_fine, 0), label='Interpolated Curve (Mx)')
# Plot the intersections
plt.hlines(target_y, 0, 99, colors='r', linestyles='dashed', label='Target Y')
plt.plot(found_x_values_T1, [target_y]*len(found_x_values_T1), 'rx', markersize=10, label='Solutions')
plt.legend()
plt.show()
# --- Plotting ---
# Create a smooth range for the spline lines
# FIX: Using x_tank3 instead of x

Volume_tank2 = total_force_downward_1and2/(WATERWEIGHT*G)+cubic_interp_vol1(found_x_values_T1)
print(cubic_interp_tcg1(found_x_values_T1),cubic_interp_vcg1(found_x_values_T1))
print(cubic_interp_tcg1(found_x_values_T1)*cubic_interp_vol1(found_x_values_T1)*WATERWEIGHT*G,"hello")

# 1. Define the objective function f(X) = Volume(X) - TargetV
def volume_function_shifted(X_val, target_volume):
    # Calculate Volume at X_val using the CubicSpline
    volume_at_X = cubic_interp_vol2(X_val)
    return volume_at_X - target_volume

target_y_T2 = -Volume_tank2
a = x_tank2.min()
b = x_tank2.max()

try:
    found_x_value_brentq_T2 = brentq(volume_function_shifted, a, b, args=(target_y_T2,))
    # Update the necessary variable for downstream use
    found_x_values_T2 = np.array([found_x_value_brentq_T2])
except ValueError:
    print("Warning: Cannot find unique solution for Tank 2 volume balance in [0, 100]. Setting filling to 0.")
    found_x_values_T2 = np.array([0.0]) # Handle case where target is out of bounds
    
print(f"Target Y (Volume Balance): {target_y_T2}")
print(f"Found X (Tank 2 Filling %): {found_x_values_T2[0]:.4f}")


# 1. Define a common, dense X-axis (Filling percentage)
# This array is used to evaluate the splines smoothly.
X_evaluation_T2 = tank2_diag_df['Tankfilling [% of h_tank]'].to_numpy()
print(x_tank2.min(),x_tank2.max())
# 2. Evaluate both splines over the common X-range
# Y_Vol (Volume at % filling) and Y_TCG (TCG at % filling)
Y_Vol_T2 = cubic_interp_vol2(X_evaluation_T2)


# 3. Perform Vectorized Multiplication
# Transverse Moment (Mx) = Volume * TCG * Density * G
# This gives the Transverse Moment for Tank 1 as a function of filling level.
Mx_T2_array = Y_Vol_T2

# 4. Clean and store the final array
Mx_T2_array_cleaned = clean_array(Mx_T2_array)



# 1. Generate Dummy Data (0-99)


target_y = -Volume_tank2

# 3. Create a Spline
# s=0 means "No smoothing" (pass through every point exactly)
spline = CubicSpline(X_evaluation_T2, Mx_T2_array_cleaned) # Use CubicSpline for plotting

# 4. Find the X values
# We interpret "f(x) = target" as "f(x) - target = 0"
# We create a shifted spline and find its roots
# The root finding is now handled by the brentq block above.
total_moment = total_moment + np.array([cubic_interp_vol1(found_x_values_T1)[0]*WATERWEIGHT*G*cubic_interp_lcg1(found_x_values_T1)[0],0])
moment_tank2 =(-Volume_tank2*WATERWEIGHT*G+BHD7_Weight+BHD8_Weight)
print(total_moment[0]/moment_tank2)




plt.plot(X_evaluation_T2, Mx_T2_array_cleaned, 'o', label='Data points', alpha=0.3)
# Plot the smooth interpolation
x_fine = np.linspace(0, 99, 1000)
plt.plot(x_fine, spline(x_fine), label='Interpolated Curve')
# Plot the intersections
plt.hlines(target_y, 0, 99, colors='r', linestyles='dashed', label='Target Y')
plt.plot(found_x_values_T2, [target_y]*len(found_x_values_T2), 'rx', markersize=10, label='Solutions')
plt.legend()
plt.show()


x_new = np.linspace(x_tank1.min(), x_tank1.max(), 100)

# Create a 3x2 grid of subplots
fig, axs = plt.subplots(3, 2, figsize=(12, 12))
fig.suptitle('Tank 3 Characteristics vs Filling Percentage', fontsize=16)

# FIX: Ensure no non-breaking spaces are present in this list (or anywhere else)
plots = [
    (0, 0, y_vol1, cubic_interp_vol1, 'Volume', 'Volume [m3]'),
    (0, 1, y_h1, cubic_interp_h1, 'Height Level', 'Height [m]'),
    (1, 0, y_vcg1, cubic_interp_vcg1, 'Vertical CG (VCG)', 'VCG [m]'),
    (1, 1, y_lcg1, cubic_interp_lcg1, 'Longitudinal CG (LCG)', 'LCG [m]'),
    (2, 0, y_tcg1, cubic_interp_tcg1, 'Transverse CG (TCG)', 'TCG [m]')
    
]

for r, c, y_data, spline, title, ylabel in plots:
    ax = axs[r, c]
    # FIX: Using x_tank3 for data points
    ax.plot(x_tank1, y_data, 'o', color='black', markersize=4, label='Data Points')
    ax.plot(x_new, spline(x_new), '-', color='blue', label='Cubic Spline')
    ax.set_title(title)
    ax.set_xlabel('Filling [%]')
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()

# Remove the empty 6th subplot (bottom right)
fig.delaxes(axs[2, 1])


plt.tight_layout()
plt.show()