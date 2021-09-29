import efm
import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import boto3


def calculate_spb(filename_timestamped, filepath_timestamped, spb_args):

    temperature_data = spb_args[0]
    seebeck_data = spb_args[1]
    resistivity_data = spb_args[2]
    carrier_data = spb_args[3]
    hall_mobility_data = spb_args[4]
    total_thermal_data = spb_args[5]
    scattering_parameter_data = spb_args[6]
    
    rfl_data = efm.rfl_from_seebeck(seebeck_data, scattering_parameter_data)
    efm_data = efm.efm(rfl_data, carrier_data, temperature_data, scattering_parameter_data)
    mu0_data = efm.mu0(rfl_data, hall_mobility_data, scattering_parameter_data)
    lorenz_data = efm.lorenz(rfl_data, scattering_parameter_data)
    electronic_thermal_data = efm.electronic_thermal(lorenz_data, resistivity_data, temperature_data)
    lattice_thermal_data = np.subtract(total_thermal_data, electronic_thermal_data)
    
    export_data = [temperature_data, 
    seebeck_data, 
    resistivity_data, 
    carrier_data, 
    hall_mobility_data,
    total_thermal_data, 
    scattering_parameter_data,
    rfl_data,
    efm_data,
    mu0_data,
    lorenz_data,
    lattice_thermal_data,
    electronic_thermal_data]
    
    export_labels = ['temperature (K)',
    'seebeck (uV/K)',
    'resistivity (mOhm-cm)',
    'carrier concentration (cm^-3)',
    'hall mobility (cm^2/V*s)',
    'Total thermal conductivity (W/mK)',
    'Scattering Parameter',
    'Reduced Fermi Level',
    'Effective Mass (m*/m0)',
    'Intrinsic Mobility (cm^2/V*s)',
    'Lorenz Number (W Ohm K^-2)',
    'Lattice thermal conductivity (W/mK)',
    'Electronic Thermal Conductivity (W/mK)']
    
    df_export = pd.DataFrame(export_data).transpose()
    df_export.columns = export_labels
    
    # File naming part depedneing on if imported file is xls or xlsx
    if filepath_timestamped.endswith('.xls'):
        export_path = filepath_timestamped[:-4] + '.xls'
    if filepath_timestamped.endswith('.xlsx'):
        export_path = filepath_timestamped[:-5] + '.xlsx'
    
    df_export.to_excel(export_path, index=False)

    excel_upload_data = open(export_path, 'rb') # Open the file into memory
    
    s3 = boto3.resource('s3')
    bucket = 'bucketeer-88c06953-e032-4084-8845-f22694bbd8b4'
    s3.Bucket(bucket).put_object(Key=filename_timestamped, Body=excel_upload_data, ACL='public-read') # upload the excel
    
    excel_link = 'https://bucketeer-88c06953-e032-4084-8845-f22694bbd8b4.s3.amazonaws.com/' + filename_timestamped
    
    excel_upload_data.close()

    return excel_link
    
def zt_excel(zt_args, filename_timestamped, filepath_timestamped):
	
    temperature_data = zt_args[0]
    seebeck_data = zt_args[1]
    resistivity_data = zt_args[2]
    thermal_data = zt_args[3]
    
    zt_data = efm.zt(seebeck_data, temperature_data, resistivity_data, thermal_data)
    
    export_data = [zt_args[0], 
    zt_args[1], 
    zt_args[2], 
    zt_args[3],
    zt_data]
    
    export_labels = ['temperature (K)',
    'seebeck (uV/K)',
    'resistivity (mOhm-cm)',
    'total thermal conductivity (W/mK)',
    'zT']
    
    df_export = pd.DataFrame(export_data).transpose()
    df_export.columns = export_labels
    
    # File naming part depedneing on if imported file is xls or xlsx
    if filepath_timestamped.endswith('.xls'):
        export_path = filepath_timestamped[:-4] + '.xls'
    if filepath_timestamped.endswith('.xlsx'):
        export_path = filepath_timestamped[:-5] + '.xlsx'
    
    df_export.to_excel(export_path, index=False)
    
    excel_upload_data = open(export_path, 'rb') # Open the file into memory
    
    s3 = boto3.resource('s3')
    bucket = 'bucketeer-88c06953-e032-4084-8845-f22694bbd8b4'
    s3.Bucket(bucket).put_object(Key=filename_timestamped, Body=excel_upload_data, ACL='public-read') # upload the excel
    
    excel_link = 'https://bucketeer-88c06953-e032-4084-8845-f22694bbd8b4.s3.amazonaws.com/' + filename_timestamped
    
    excel_upload_data.close()

    return excel_link

def theoretical_zt_max_job(zt_max_args, file_write_location):
        tzt = efm.theoretical_zt_max(*zt_max_args)

        theoretical_zt_max_value = format(float(tzt[0][0]),".3f")
        carrier_for_zt_max_value = "{:0.3e}".format(float(tzt[1][0]))

        timestamp = str(time.time()).replace('.','_')
        excel_file_name = 'theoretical_zt_plot_' + str(timestamp) + '.xlsx'
        full_file_path_excel = os.path.join(file_write_location, excel_file_name)
        export_data = [tzt[2][0], tzt[3][0]]
        export_labels = ['Carrier Concentration (cm^-3)', 'Theoretical zT']
        df_export = pd.DataFrame(export_data).transpose()
        df_export.columns = export_labels
        df_export.to_excel(full_file_path_excel, index=False)

        fig, ax = plt.subplots(1, figsize=(6,6))
        ax.plot(tzt[2][0], tzt[3][0], color="#0000FF")
        ax.scatter(tzt[1], tzt[0], color="#FF0000")
        ax.set_xlabel('Carrier Concentration (cm$^{-3}$)', fontsize=14)
        ax.set_ylabel('Theoretical zT', fontsize=14)
        ax.set_xscale('log')
        plt.tight_layout()
        plot_file_name = 'theoretical_zt_plot_' + str(timestamp) + '.png'
        full_file_path_plot = os.path.join(file_write_location, plot_file_name)
        plt.savefig(full_file_path_plot, dpi=500)
        
        plot_upload_data = open(full_file_path_plot, 'rb')
        excel_upload_data = open(full_file_path_excel, 'rb')
        
        s3 = boto3.resource('s3')
        bucket = 'bucketeer-88c06953-e032-4084-8845-f22694bbd8b4'
        s3.Bucket(bucket).put_object(Key=plot_file_name, Body=plot_upload_data, ACL='public-read', ContentType='image/jpeg') # upload the plot
        s3.Bucket(bucket).put_object(Key=excel_file_name, Body=excel_upload_data, ACL='public-read') # upload the excel
        
        plot_link = 'https://bucketeer-88c06953-e032-4084-8845-f22694bbd8b4.s3.amazonaws.com/' + plot_file_name
        excel_link = 'https://bucketeer-88c06953-e032-4084-8845-f22694bbd8b4.s3.amazonaws.com/' + excel_file_name
        
        plot_upload_data.close()
        excel_upload_data.close()
        
        return theoretical_zt_max_value, carrier_for_zt_max_value, plot_link, excel_link
        

def theoretical_zt_max_excel(imported_data, filename_timestamped, filepath_timestamped):
    
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
    'carrier concentration for theoretical max zT (cm-3)']
    
    df_export = pd.DataFrame(export_data).transpose()
    df_export.columns = export_labels
    
    #File naming part depending on if imported file is xls or xlsx
    if filepath_timestamped.endswith('.xls'):
        export_path = filepath_timestamped[:-4] + '.xls'
    if filepath_timestamped.endswith('.xlsx'):
        export_path = filepath_timestamped[:-5] + '.xlsx'

    df_export.to_excel(export_path, index=False)
    
    excel_upload_data = open(export_path, 'rb') # Open the file into memory
    
    s3 = boto3.resource('s3')
    bucket = 'bucketeer-88c06953-e032-4084-8845-f22694bbd8b4'
    s3.Bucket(bucket).put_object(Key=filename_timestamped, Body=excel_upload_data, ACL='public-read') # upload the excel
    
    excel_link = 'https://bucketeer-88c06953-e032-4084-8845-f22694bbd8b4.s3.amazonaws.com/' + filename_timestamped
    
    excel_upload_data.close()
    
    return excel_link




