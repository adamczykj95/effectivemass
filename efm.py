import numpy as np
import pandas as pd
import scipy.integrate
import scipy.interpolate
from scipy.optimize import fsolve

# Some constants
e = 1.60217653E-19 # Charge on electron (Coulomb)
kb = 1.3806505E-23 # boltzmann constant (J/K)
hbar = 1.054571817E-34 # Planck constant

def fermi_integral(j, eta):
    integralLowerLimit = 0  # Set the lower limit of the fermi integrals
    integralUpperLimit = 100  # Set the upper limit of the fermi integrals
    def integrand(epsilon, j, eta):
        return (epsilon ** j) / (1 + np.exp(epsilon - eta))
    definite_integral = scipy.integrate.quad(integrand, integralLowerLimit, integralUpperLimit, args=(j,eta))[0]
    return definite_integral


def seebeck_fermi_relation(eta, r, seebeck):
    return abs(seebeck) - (1E6 * kb / e * (((r + 5/2) * fermi_integral(r+(3/2), eta)) / ((r + 3/2) * fermi_integral(r+(1/2), eta)) - eta))

def check_list_lengths_equal(lists):
    # Takes a list of lists as its argument
    if isinstance(lists, list):
        length = len(lists[0])
        if all(len(lst) == length for lst in lists):
            return True
        else:
            raise ValueError('Lists must be of the same length.')
    else:
        raise ValueError('Input must be of type list')

def rfl_from_seebeck(seebeck_list, r_list):
    # Check length of lists to make sure they are equal
    if check_list_lengths_equal([seebeck_list, r_list]):
        rfl_list = []
        for s in range(0, len(seebeck_list)):
            rfl = float(fsolve(seebeck_fermi_relation, 1, args=(r_list[s], seebeck_list[s])))
            rfl_list.append(rfl)
        
        return rfl_list
    else:
        pass

def efm(rfl_list, carrier_list, temperature_list, r_list):
    # Check length of lists to make sure they are equal
    if check_list_lengths_equal([rfl_list, carrier_list, temperature_list, r_list]):
        efm_list = []
        for rfl in range(0, len(rfl_list)):
            efm = (1 / (2*kb*temperature_list[rfl])) * ((3 * carrier_list[rfl] * 1E6 * np.pi**2 * hbar**3) * \
            \
            ((2*r_list[rfl]+(3/2)) / ((r_list[rfl]+(3/2))**2)) * \
            \
            (fermi_integral(2*r_list[rfl] + (1/2), rfl_list[rfl]) / (fermi_integral(r_list[rfl] + (1/2), rfl_list[rfl])**2))) ** (2/3)
            
            efm = efm / 9.10938356E-31 # mass of electron in Kg
            
            efm_list.append(efm)
        return efm_list
    else:
        pass

def mu0(rfl_list, hall_mobility_list, r_list):
    # Check length of lists to make sure they are equal
    if check_list_lengths_equal([rfl_list, hall_mobility_list, r_list]):
        mu0_list = []
        for rfl in range(0, len(rfl_list)):
            mu0 = hall_mobility_list[rfl] * ((3/2 + r_list[rfl]) / (3/2 + 2*r_list[rfl])) *\
            (fermi_integral(r_list[rfl]+1/2, rfl_list[rfl]) / fermi_integral(2*r_list[rfl] + (1/2), rfl_list[rfl]))
            mu0_list.append(mu0)
        return mu0_list
    else:
        pass

def lorenz(rfl_list, r_list):
    # Check length of list sto make sure they are equal
    if check_list_lengths_equal([rfl_list, r_list]):
        lorenz_number_list = []
        for rfl in range(0, len(rfl_list)):
            lorenz_number = (kb**2/e**2) *\
            (\
            ((r_list[rfl] + (3/2)) * \
            (r_list[rfl] + (7/2)) * \
            fermi_integral(r_list[rfl] + (1/2), rfl_list[rfl]) * \
            fermi_integral(r_list[rfl] + (5/2), rfl_list[rfl])) - \
            (((r_list[rfl] + (5/2))**2) * \
            fermi_integral(r_list[rfl]+(3/2), rfl_list[rfl])**2)\
            ) \
            /\
            (\
            ((r_list[rfl] + (3/2))**2) * \
            (fermi_integral(r_list[rfl] + (1/2), rfl_list[rfl])**2)\
            )
            
            lorenz_number_list.append(lorenz_number)
        return lorenz_number_list
    else:
        pass

