"""Physical constants used in multiple locations."""

R_Mars_km = 3.3895e3  # [km]
D_offset = 0.034 # [nm], Δλ between H Ly α and D Ly α
IPH_wv_spread = 0.002026 # [nm]--Allows for ± 5 km/s uncertainty in average
                         # IPH velocity. computed by doppler shift equation;
                         # (5 km/s / c) = (Δλ / λ)
IPH_minw = 0.00546  # Min width of IPH in nm, computed from Doppler equation, 
                     # with Δv = sqrt(2kT/m), where m is for H, and T is the 
                     # minimum temperature (~11000 K) from Quemerais 2006, 
                     # Figure 5. 
IPH_maxw = 0.00727  # Max width of IPH in nm, computed from Doppler equation, 
                     # with Δv = sqrt(2kT/m), where m is for H, and T is the 
                     # maximum temperature (~19500 K) from Quemerais 2006, 
                     # Figure 5. 