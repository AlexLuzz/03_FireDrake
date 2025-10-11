from firedrake import *
import numpy as np


# ============================================
# 3. VAN GENUCHTEN FUNCTIONS
# ============================================

def calculate_Se(Hp, alpha=0.08, n=1.6, m=1 - 1/1.6, epsilon=1e-3):
    """Calculate effective saturation Se from pressure head with smoothing"""
    if Hp >= epsilon:
        return 1.0
    elif Hp <= -epsilon:
        return 1.0 / (1.0 + abs(alpha * Hp)**n)**m
    else:
        # Smooth transition near Hp = 0
        Se_neg = 1.0 / (1.0 + abs(alpha * (-epsilon))**n)**m
        weight = (Hp + epsilon) / (2.0 * epsilon)
        return Se_neg + (1.0 - Se_neg) * weight

def calculate_theta(Hp, theta_r=0.05, theta_s=0.40):
    """Calculate water content θ from pressure head"""
    Se = calculate_Se(Hp)
    return theta_r + Se * (theta_s - theta_r)

def calculate_Cm(Hp, 
                 epsilon=1e-3, 
                 alpha=0.08, 
                 m=1 - 1/1.6,
                 theta_r=0.05, 
                 theta_s=0.40):
    """Calculate specific moisture capacity Cm with smoothing"""
    if Hp >= epsilon:
        return 1e-6  # Small value in saturated zone
    elif Hp <= -epsilon:
        Se = calculate_Se(Hp)
        return (alpha * m) / (1.0 - m) * (theta_s - theta_r) * Se**(1.0/m) * (1.0 - Se**(1.0/m))**m
    else:
        # Smooth transition
        return 1e-6

def calculate_kr(Hp,
                 epsilon=0.01,
                 l_param=0.5,
                 kr_min=1e-7,
                 m=1 - 1/1.6):
    """Calculate relative permeability kr with bounds"""
    if Hp >= epsilon:
        return 1.0
    elif Hp <= -epsilon:
        Se = calculate_Se(Hp)
        kr_val = Se**l_param * (1.0 - (1.0 - Se**(1.0/m))**m)**2
        return max(kr_min, min(kr_val, 1.0))  # Bound between kr_min and 1.0
    else:
        # Smooth transition
        kr_neg = calculate_kr(-epsilon)
        weight = (Hp + epsilon) / (2.0 * epsilon)
        return kr_neg + (1.0 - kr_neg) * weight

def calculate_saturation(V, 
                         pressure_head,
                         theta_r=0.05, 
                         theta_s=0.40
                         ):
    """
    Calculate saturation from pressure head using van Genuchten model
    Returns normalized saturation for plotting (0 to 1 scale)
    """
    # Create saturation field
    S_field = Function(V)
    
    # Get pressure values
    p_vals = pressure_head.dat.data[:]
    S_vals = np.zeros_like(p_vals)
    
    # Calculate saturation
    for i, Hp in enumerate(p_vals):
        theta = calculate_theta(Hp)
        # Normalize to [0, 1] range for plotting
        S_vals[i] = max(theta_r/theta_s, min(theta / theta_s, 1.0))
    
    S_field.dat.data[:] = S_vals
    return S_field

def calculate_coefficients(V,
                           p_field,
                           ):
    """Calculate Cm and kr fields from pressure field with smoothing"""
    Cm_field = Function(V)
    kr_field = Function(V)
    
    p_vals = p_field.dat.data[:]
    Cm_vals = np.zeros_like(p_vals)
    kr_vals = np.zeros_like(p_vals)
    
    for i, Hp in enumerate(p_vals):
        Cm_vals[i] = calculate_Cm(Hp)
        kr_vals[i] = calculate_kr(Hp)
    
    Cm_field.dat.data[:] = Cm_vals
    kr_field.dat.data[:] = kr_vals
    
    return Cm_field, kr_field