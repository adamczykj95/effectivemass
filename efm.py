import numpy as np
import pandas as pd
import scipy.integrate
import scipy.interpolate
from scipy.optimize import fsolve

# Some constants
e = 1.60217653E-19 # Charge on electron (Coulomb)
kb = 1.3806505E-23 # boltzmann constant (J/K)
hbar = 1.054571817E-34 # Planck constant
m = 9.10938356E-31 # Mass of electron in kg

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
        seebeck_list = np.abs(seebeck_list)
        rfl_list = []
        for s in range(0, len(seebeck_list)):
            rfl = float(fsolve(seebeck_fermi_relation, 1, args=(r_list[s], seebeck_list[s])))
            rfl_list.append(rfl)
        
        return rfl_list
    else:
        return print('rfl_from_seebeck function failed.') 

def seebeck_from_rfl(rfl_list, r_list):
    if check_list_lengths_equal([rfl_list, r_list]):
        seebeck_from_rfl_list = []
        for rfl in range(len(rfl_list)):
            seebeck = (1E6*kb / e * (((r_list[rfl] + 5/2) * fermi_integral(r_list[rfl]+(3/2), rfl_list[rfl])) / ((r_list[rfl] + 3/2) * fermi_integral(r_list[rfl]+(1/2), rfl_list[rfl])) - rfl_list[rfl]))
            seebeck_from_rfl_list.append(seebeck)
        return seebeck_from_rfl_list
    else:
        print("seebeck_from_rfl function failed.")

def efm(rfl_list, carrier_list, temperature_list, r_list):
    # Check length of lists to make sure they are equal
    if check_list_lengths_equal([rfl_list, carrier_list, temperature_list, r_list]):
        efm_list = []
        carrier_list = np.abs(carrier_list)
        for rfl in range(0, len(rfl_list)):
            efm = (1 / (2*kb*temperature_list[rfl])) * ((3 * carrier_list[rfl] * 1E6 * np.pi**2 * hbar**3) * \
            \
            ((2*r_list[rfl]+(3/2)) / ((r_list[rfl]+(3/2))**2)) * \
            \
            (fermi_integral(2*r_list[rfl] + (1/2), rfl_list[rfl]) / (fermi_integral(r_list[rfl] + (1/2), rfl_list[rfl])**2))) ** (2/3)
            
            efm = efm / m # mass of electron in Kg
            
            efm_list.append(efm)
        return efm_list
    else:
        return print('efm function failed')

def mu0(rfl_list, hall_mobility_list, r_list):
    # Check length of lists to make sure they are equal
    if check_list_lengths_equal([rfl_list, hall_mobility_list, r_list]):
        hall_mobility_list = np.abs(hall_mobility_list)
        mu0_list = []
        for rfl in range(0, len(rfl_list)):
            mu0 = hall_mobility_list[rfl] * ((3/2 + r_list[rfl]) / (3/2 + 2*r_list[rfl])) *\
            (fermi_integral(r_list[rfl]+1/2, rfl_list[rfl]) / fermi_integral(2*r_list[rfl] + (1/2), rfl_list[rfl]))
            mu0_list.append(mu0)
        return mu0_list
    else:
        return print('mu0 function failed.')


def lorenz(rfl_list, r_list):
    # Check length of lists to make sure they are equal
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
        return print('lorenz function failed.')
        
def electronic_thermal(lorenz_list, resistivity_list, temperature_list):
    # Check length of lists to make sure they are equal
    if check_list_lengths_equal([lorenz_list, resistivity_list, temperature_list]):
        electronic_thermal_list = []
        for l in range(0, len(lorenz_list)):
            ke = lorenz_list[l] * ((resistivity_list[l] * 1E-5)**(-1)) * temperature_list[l]
            electronic_thermal_list.append(ke)
        return electronic_thermal_list
    else:
        return print('electronic_thermal function failed.')

def psi(rfl_list, r_list):
    if check_list_lengths_equal([rfl_list, r_list]):
        psi_list = []
        for rfl in range(0, len(rfl_list)):
            psi = (8*np.pi*e/3) * ((2*m*kb / ((hbar*2*np.pi)**2)) ** (3/2)) *\
            (1.5 + r_list[rfl]) * fermi_integral(r_list[rfl] + 0.5, rfl_list[rfl])
            psi_list.append(psi)
        return psi_list
    else:
        return print('psi function failed')

def beta(mu0_list, efm_list, temperature_list, kl_list):
    if check_list_lengths_equal([mu0_list, efm_list, temperature_list, kl_list]):
        beta_list = []
        mu0_list = np.abs(mu0_list)
        for mu in range(0, len(mu0_list)):
            beta = (mu0_list[mu]* 1E-4 * efm_list[mu] ** (3/2) * temperature_list[mu] ** (5/2)) / kl_list[mu]
            beta_list.append(beta)
        return beta_list
    else:
        return print('beta function failed')


def zt(seebeck_list, temperature_list, resistivity_list, total_thermal_list):
    if check_list_lengths_equal([seebeck_list, temperature_list, resistivity_list, total_thermal_list]):
        zt_list = []
        seebeck_list = np.abs(seebeck_list)
        for s in range(0, len(seebeck_list)):
            zt = ((seebeck_list[s] * 1E-6)**2 * temperature_list[s]) / (resistivity_list[s] * 1E-5 * total_thermal_list[s])
            zt_list.append(zt)
        return zt_list
    else:
        return print("Experimental zt function failed")

def theoretical_zt(seebeck_list, lorenz_list, psi_list, beta_list):
    if check_list_lengths_equal([seebeck_list, lorenz_list, psi_list, beta_list]):
        zt_list = []
        seebeck_list = np.abs(seebeck_list)
        for s in range(0, len(seebeck_list)):
            zt = ((seebeck_list[s]*1E-6) ** 2) / (lorenz_list[s] + (psi_list[s] * beta_list[s])**-1)
            zt_list.append(zt)
        return zt_list
    else:
        return print("Theoretical zt function failed")

  
def rfl_from_carrier(carrier_list, efm_list, temperature_list, r_list):
    if check_list_lengths_equal([carrier_list, efm_list, temperature_list, r_list]):
        rfl_list = []
        carrier_list = np.abs(carrier_list)
        for cc in range(0, len(carrier_list)):
            def carrier_fermi(rfl, carrier_concentration):
                return abs(carrier_concentration) -\
                (((2*efm_list[cc]*m*kb*temperature_list[cc])**(3/2) \
                / (3*np.pi**2*hbar**3)) *\
                (((r_list[cc]+3/2)**2 * fermi_integral(r_list[cc] + 0.5, rfl)**2)\
                /((2*r_list[cc] + 3/2) * fermi_integral(2*r_list[cc]+0.5, rfl))))
            
            rfl_list.append(float(fsolve(carrier_fermi, 1, args=(carrier_list[cc]*1E6))))
        return rfl_list
    else:
        return print("rfl_from_carrier function failed")



def theoretical_zt_max(efm_list, kl_list, hall_mobility_list, temperature_list, r_list):
    if check_list_lengths_equal([efm_list, kl_list, hall_mobility_list, temperature_list, r_list]):
        zt_max_list = []
        hall_mobility_list = np.abs(hall_mobility_list)
        carrier_concentration_for_max_zt_list = []
        iterations = 100
        carrier_concentration_list = np.logspace(16,22,iterations)
        nested_theoretical_zt_lists = []
        nested_carrier_concentration_lists = []
        
        for efm in range(len(efm_list)):
            temperature_values = iterations * [temperature_list[efm]]
            kl_values = iterations * [kl_list[efm]]
            hall_mobility_values = iterations * [hall_mobility_list[efm]]
            effective_mass_values = iterations * [efm_list[efm]]
            r_values = iterations*[r_list[efm]]
            
            rfl_from_carrier_list = rfl_from_carrier(carrier_concentration_list, effective_mass_values, temperature_values, r_values)
            seebeck_from_rfl_list = seebeck_from_rfl(rfl_from_carrier_list, r_values)
            lorenz_list = lorenz(rfl_from_carrier_list, r_values)
            intrinsic_mobility_list = mu0(rfl_from_carrier_list, hall_mobility_values, r_values)
            beta_list = beta(intrinsic_mobility_list, effective_mass_values, temperature_values, kl_values)
            psi_list = psi(rfl_from_carrier_list, r_values)
            
            theoretical_zt_list = theoretical_zt(seebeck_from_rfl_list, lorenz_list, psi_list, beta_list)
            nested_theoretical_zt_lists.append(theoretical_zt_list)
            nested_carrier_concentration_lists.append(carrier_concentration_list)
            zt_max_list.append(np.max(theoretical_zt_list))
            max_index = np.argmax(theoretical_zt_list)
            carrier_concentration_for_max_zt_list.append(carrier_concentration_list[max_index])
        
        return zt_max_list, carrier_concentration_for_max_zt_list, nested_carrier_concentration_lists, nested_theoretical_zt_lists


def calculate_spb(excel_file_path):
    full_file_path = excel_file_path
    imported_data = pd.read_excel(full_file_path)
    imported_data = imported_data.fillna('0')
    imported_data = imported_data.values
    
    temperature_data =          list(imported_data[0:,0])
    seebeck_data =              list(imported_data[0:,1])
    resistivity_data =          list(imported_data[0:,2])
    carrier_data =              list(imported_data[0:,3])
    hall_mobility_data =        list(imported_data[0:,4])
    scattering_parameter_data = list(imported_data[0:,5])
    
    rfl_data = rfl_from_seebeck(seebeck_data, scattering_parameter_data)
    efm_data = efm(rfl_data, carrier_data, temperature_data, scattering_parameter_data)
    mu0_data = mu0(rfl_data, hall_mobility_data, scattering_parameter_data)
    lorenz_data = lorenz(rfl_data, scattering_parameter_data)
    electronic_thermal_data = electronic_thermal(lorenz_data, resistivity_data, temperature_data)
    
    export_data = [temperature_data, 
    seebeck_data, 
    resistivity_data, 
    carrier_data, 
    hall_mobility_data, 
    scattering_parameter_data,
    rfl_data,
    efm_data,
    mu0_data,
    lorenz_data,
    electronic_thermal_data]
    
    export_labels = ['temperature (K)',
    'seebeck (uV/K)',
    'resistivity (mOhm-cm)',
    'carrier concentration (cm^-3)',
    'hall mobility (cm^2/V*s)',
    'Scattering Parameter',
    'Reduced Fermi Level',
    'Effective Mass (m*/m0)',
    'Intrinsic Mobility (cm^2/V*s)',
    'Lorenz Number (W Ohm K^-2)',
    'Electronic Thermal Conductivity (W/mK)']
    
    df_export = pd.DataFrame(export_data).transpose()
    df_export.columns = export_labels
    
    # File naming part depedneing on if imported file is xls or xlsx
    if excel_file_path.endswith('.xls'):
        export_path = excel_file_path[:-4] + '_spb.xls'
    if excel_file_path.endswith('.xlsx'):
        export_path = excel_file_path[:-5] + '_spb.xlsx'
    
    df_export.to_excel(export_path, index=False)

    return export_path
    
def zt_excel(excel_file_path):
    full_file_path = excel_file_path
    imported_data = pd.read_excel(full_file_path)
    imported_data = imported_data.fillna('0')
    imported_data = imported_data.values
    
    if len(imported_data[0]) != 4:
        raise IndexError
    
    temperature_data =          list(imported_data[0:,0])
    seebeck_data =              list(imported_data[0:,1])
    resistivity_data =          list(imported_data[0:,2])
    thermal_data =              list(imported_data[0:,3])

    zt_data = zt(seebeck_data, temperature_data, resistivity_data, thermal_data)
    
    export_data = [temperature_data, 
    seebeck_data, 
    resistivity_data, 
    thermal_data,
    zt_data]
    
    export_labels = ['temperature (K)',
    'seebeck (uV/K)',
    'resistivity (mOhm-cm)',
    'total thermal conductivity (W/mK)',
    'zT']
    
    df_export = pd.DataFrame(export_data).transpose()
    df_export.columns = export_labels
    
    # File naming part depedneing on if imported file is xls or xlsx
    if excel_file_path.endswith('.xls'):
        export_path = excel_file_path[:-4] + '_spb.xls'
    if excel_file_path.endswith('.xlsx'):
        export_path = excel_file_path[:-5] + '_spb.xlsx'
    
    df_export.to_excel(export_path, index=False)

    return export_path

def theoretical_zt_max_excel(excel_file_path):
    full_file_path = excel_file_path
    imported_data = pd.read_excel(full_file_path)
    imported_data = imported_data.fillna('0')
    imported_data = imported_data.values
    
    if len(imported_data[0]) != 5:
        raise IndexError
    
    temperature_data = list(imported_data[0:,0])
    effmass_data =     list(imported_data[0:,1])
    kl_data =          list(imported_data[0:,2])
    mu_data =          list(imported_data[0:,3])
    r_data =           list(imported_data[0:,4])

    zt_max_data = theoretical_zt_max(effmass_data, kl_data, mu_data, temperature_data, r_data)
    
    export_data = [temperature_data, 
    effmass_data, 
    kl_data, 
    mu_data,
    r_data,
    zt_max_data[0],
    zt_max_data[1]]
    
    export_labels = ['temperature (K)',
    'effective mass (m*/m0)',
    'lattice thermal conductivity (W/mK)',
    'Hall mobility (cm^2/V*s)',
    'scattering parameter',
    'theoretical max zT',
    'carrier concentration for theoretical max zT']
    
    df_export = pd.DataFrame(export_data).transpose()
    df_export.columns = export_labels
    
    # File naming part depedneing on if imported file is xls or xlsx
    if excel_file_path.endswith('.xls'):
        export_path = excel_file_path[:-4] + '_spb.xls'
    if excel_file_path.endswith('.xlsx'):
        export_path = excel_file_path[:-5] + '_spb.xlsx'
    
    df_export.to_excel(export_path, index=False)

    return export_path