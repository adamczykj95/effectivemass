import efm
import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def calculate_spb(excel_file_path):
    full_file_path = excel_file_path
    imported_data = pd.read_excel(full_file_path)
    imported_data = imported_data.fillna('0')
    imported_data = imported_data.values
    
    temperature_data =          list(imported_data[0:,0])
    seebeck_data =              list(imported_data[0:,1])
    resistivity_data =          list(imported_data[0:,2])
    resistivity_data =          list(imported_data[0:,2])
    carrier_data =              list(imported_data[0:,3])
    hall_mobility_data =        list(imported_data[0:,4])
    scattering_parameter_data = list(imported_data[0:,5])
    
    rfl_data = efm.rfl_from_seebeck(seebeck_data, scattering_parameter_data)
    efm_data = efm.efm(rfl_data, carrier_data, temperature_data, scattering_parameter_data)
    mu0_data = efm.mu0(rfl_data, hall_mobility_data, scattering_parameter_data)
    lorenz_data = efm.lorenz(rfl_data, scattering_parameter_data)
    electronic_thermal_data = efm.electronic_thermal(lorenz_data, resistivity_data, temperature_data)
    
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
        export_path = excel_file_path[:-4] + '.xls'
    if excel_file_path.endswith('.xlsx'):
        export_path = excel_file_path[:-5] + '.xlsx'
    
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

    zt_data = efm.zt(seebeck_data, temperature_data, resistivity_data, thermal_data)
    
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
        export_path = excel_file_path[:-4] + '.xls'
    if excel_file_path.endswith('.xlsx'):
        export_path = excel_file_path[:-5] + '.xlsx'
    
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

    zt_max_data = efm.theoretical_zt_max(effmass_data, kl_data, mu_data, temperature_data, r_data)
    
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
    
    # File naming part depending on if imported file is xls or xlsx
    if excel_file_path.endswith('.xls'):
        export_path = excel_file_path[:-4] + '.xls'
    if excel_file_path.endswith('.xlsx'):
        export_path = excel_file_path[:-5] + '.xlsx'
    
    df_export.to_excel(export_path, index=False)

    return export_path

def write_file():
    __name__ = '__main__'
    f = open("static/uploads/demo_file.txt", "w")
    f.write("some content in the file")
    f.close()
    print('efm_excel.py NAME: ', __name__)

