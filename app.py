from flask import Flask, render_template, request, url_for, flash, Markup
import efm
import efm_excel
import os
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import pandas as pd
import numpy as np
import warnings
from flask_wtf import FlaskForm
from wtforms import StringField, SelectField, SubmitField
from wtforms.validators import InputRequired, ValidationError
from flask_wtf.file import FileField, FileAllowed, FileRequired
from periodictable import formula

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.config['SECRET_KEY'] = '3508sdfnl3nljnse20851j0adljnsd 0j123_+!#%(*@4j0182@$)*'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 1000*1000 # 1mb max file size

def allowed_file(filename, extensions):
    # This function returns True if the extensions are allowed. 
    # False if the extensions are not allowed.
    return '.' in filename and filename.rsplit('.',1)[1].lower()\
    in extensions

def floatCheck(form, field):
    try:
        float(field.data)
    except ValueError:
        raise ValidationError('Input must be numeric')


@app.route('/<name>')
def download(name=''):
    return send_from_directory(name)

seebeck_units_choices = [(1, Markup('&mu;V/K')), (1E3, 'mV/K'), (1E6, 'V/K')] # Conversions to get to uV/K
resistivity_units_choices = [(1, Markup('m&Omega;&bull;cm')), (1E3, Markup('&Omega;&bull;cm')), (1E5, Markup('&Omega;&bull;m'))] # Conversions to get to mOhm-cm
temperature_units_choices = [(0, 'K'), (273, 'C')]
carrier_units_choices = [(1, Markup('cm<sup>-3</sup>')), (1E-6, Markup('m<sup>-3</sup>'))]
thermal_units_choices = [(1, Markup('W/m&bull;K')), (0.1, Markup('mW/cm&bull;K')), (100, Markup('W/cm&bull;K'))]
mobility_units_choices = [(1, Markup('cm<sup>2</sup>/V&bull;s')), (1E4, Markup('m<sup>2</sup>/V&bull;s'))]
effectivemass_units_choices = [(1, Markup('m<sup>*</sup>/m<sub>e</sub>')),(9.10938356E-31, 'kg')]
diffusivity_units_choices = [(1E-6, Markup('mm<sup>2</sup>/s')), (1E-4, Markup('cm<sup>2</sup>/s')), (1, Markup('m<sup>2</sup>/s'))]
density_units_choices = [(1000, Markup('g&bull;cm<sup>-3</sup>')), (1, Markup('kg&bull;m<sup>-3</sup>'))]
heat_capacity_units_choices = [(1, Markup('J/kg&bull;K')), (1E-3, Markup('J/g&bull;K'))]

class efmassForm(FlaskForm):
    seebeck_efm = StringField(Markup('Seebeck coefficient:'), validators=[InputRequired('Seebeck input required'), floatCheck])
    seebeck_units_efm = SelectField(u'Seebeck Units', choices=seebeck_units_choices)
    
    carrier_efm = StringField(Markup('Carrier Concentration:'), validators=[InputRequired('Carrier concentration input required'), floatCheck])
    carrier_units_efm = SelectField(u'Carrier Units', choices=carrier_units_choices)
    
    temperature_efm = StringField('Temperature:', validators=[InputRequired('Temperature input required'), floatCheck])
    temperature_units_efm = SelectField(u'Temperature Units', choices=temperature_units_choices)
    
    r_efm = StringField('Scattering parameter:', validators=[InputRequired('Scattering parameter input'), floatCheck], default='-0.5')
    
class mu0Form(FlaskForm):
    seebeck_mu0 = StringField(Markup('Seebeck coefficient:'), validators=[InputRequired('Seebeck input required'), floatCheck])
    seebeck_units_mu0 = SelectField(u'Seebeck Units', choices=seebeck_units_choices)
    
    mu_mu0 = StringField(Markup('Mobility:'), validators=[InputRequired('Seebeck input required'), floatCheck])
    mu_units_mu0 = SelectField(u'Mobility Units', choices=mobility_units_choices)
    
    r_mu0 = StringField('Scattering parameter:', validators=[InputRequired('Scattering parameter input'), floatCheck], default='-0.5')

class lorenzForm(FlaskForm):
    seebeck_lorenz = StringField(Markup('Seebeck coefficient:'), validators=[InputRequired('Seebeck input required'), floatCheck])
    seebeck_units_lorenz = SelectField(u'Seebeck Units', choices=seebeck_units_choices)
    
    r_lorenz = StringField('Scattering parameter:', validators=[InputRequired('Scattering parameter input'), floatCheck], default='-0.5')

class keForm(FlaskForm):
    seebeck_ke = StringField(Markup('Seebeck coefficient:'), validators=[InputRequired('Seebeck input required'), floatCheck])
    seebeck_units_ke = SelectField(u'Seebeck Units', choices=seebeck_units_choices)
    
    resistivity_ke = StringField(Markup('Resistivity:'), validators=[InputRequired('Resistivity input required'), floatCheck])
    resistivity_units_ke = SelectField(u'Resistivity Units', choices=resistivity_units_choices)

    temperature_ke = StringField('Temperature:', validators=[InputRequired('Temperature input required'), floatCheck])
    temperature_units_ke = SelectField(u'Temperature Units', choices=temperature_units_choices)
    
    r_ke = StringField('Scattering parameter:', validators=[InputRequired('Scattering parameter input'), floatCheck], default='-0.5')

class psiForm(FlaskForm):
    seebeck_psi = StringField(Markup('Seebeck coefficient:'), validators=[InputRequired('Seebeck input required'), floatCheck])
    seebeck_units_psi = SelectField(u'Seebeck Units', choices=seebeck_units_choices)
    r_psi = StringField('Scattering parameter:', validators=[InputRequired('Scattering parameter input'), floatCheck], default='-0.5')
    
class betaForm(FlaskForm):
    mu0_beta = StringField(Markup('Intrinsic mobility:'), validators=[InputRequired('Intrinsic mobility input required'), floatCheck])
    mu0_units_beta = SelectField(u'Mobility Units', choices=mobility_units_choices)
    
    efmass_beta = StringField(Markup('Effective mass:'), validators=[InputRequired('Effective mass input required'), floatCheck])
    efmass_units_beta = SelectField(u'Effective mass units', choices=effectivemass_units_choices)
    
    temperature_beta = StringField('Temperature:', validators=[InputRequired('Temperature input required'), floatCheck])
    temperature_units_beta = SelectField(u'Temperature Units', choices=temperature_units_choices)
    
    kl_beta = StringField('Lattice Thermal Conductivity:', validators=[InputRequired('Lattice thermal conductivity input'), floatCheck])
    kl_units_beta = SelectField(u'Lattice thermal units', choices=thermal_units_choices)
    

class rflsForm(FlaskForm):
    seebeck_rfl_s = StringField(Markup('Seebeck coefficient:'), validators=[InputRequired('Seebeck input required'), floatCheck])
    seebeck_units_rfl_s = SelectField(u'Seebeck Units', choices=seebeck_units_choices)
    r_rfl_s = StringField('Scattering parameter', validators=[InputRequired('Scattering parameter input'), floatCheck], default='-0.5')

class srflForm(FlaskForm):
    rfl_s_rfl = StringField(Markup('Reduced Fermi level (k<sub>B</sub>T): '), validators=[InputRequired('Reduced Fermi level input required'), floatCheck])
    r_s_rfl = StringField('Scattering parameter: ', validators=[InputRequired('Scattering parameter input'), floatCheck], default='-0.5')

@app.route('/', methods=['GET', 'POST'])
def index():
    efmass_form = efmassForm()
    mu0_form = mu0Form()
    lorenz_form = lorenzForm()
    ke_form = keForm()
    psi_form = psiForm()
    beta_form = betaForm()
    rfl_s_form = rflsForm()
    s_rfl_form = srflForm()
    
    if 'efmass' in request.form:
        if request.method == "POST" and efmass_form.validate_on_submit():
            seebeck = float(efmass_form.seebeck_efm.data) * float(efmass_form.seebeck_units_efm.data)
            
            carrier = float(efmass_form.carrier_efm.data) * float(efmass_form.carrier_units_efm.data)
            temperature = float(efmass_form.temperature_efm.data) + float(efmass_form.temperature_units_efm.data)
            r = float(efmass_form.r_efm.data)
            
            try:
                rfl = efm.rfl_from_seebeck([seebeck], [r])[0]
                efmass_result = "{:.3f}".format(efm.efm([rfl], [carrier], [temperature], [r])[0])
                flash(efmass_result)
            except ZeroDivisionError:
                efmass_result_error = 'Inputs resulted in an error. Try some more realistic numbers.'
                flash(efmass_result_error)
            except OverflowError:
                efmass_result_error = 'Inputs resulted in an error. Try some more realistic numbers.'
                flash(efmass_result_error)
            
    elif 'mu0' in request.form:
        if request.method == "POST" and mu0_form.validate_on_submit():
            seebeck = float(mu0_form.seebeck_mu0.data) * float(mu0_form.seebeck_units_mu0.data)
            mu = float(mu0_form.mu_mu0.data) * float(mu0_form.mu_units_mu0.data)
            r = float(mu0_form.r_mu0.data)
            
            try:
                rfl = efm.rfl_from_seebeck([seebeck], [r])[0]
                mu0_result = "{:.3f}".format(efm.mu0([rfl], [mu], [r])[0])
                flash(mu0_result)
            except ZeroDivisionError:
                mu0_result_error = 'Inputs resulted in an error. Try some more realistic numbers.'
                flash(mu0_result_error)
            except OverflowError:
                mu0_result_error = 'Inputs resulted in an error. Try some more realistic numbers.'
                flash(mu0_result_error)
            
            
    elif 'lorenz' in request.form:
        if request.method == "POST" and lorenz_form.validate_on_submit():
            seebeck = float(lorenz_form.seebeck_lorenz.data) * float(lorenz_form.seebeck_units_lorenz.data)
            r = float(lorenz_form.r_lorenz.data)
            
            try:
                rfl = efm.rfl_from_seebeck([seebeck], [r])[0]
                lorenz_result = "{:.3e}".format(efm.lorenz([rfl], [r])[0])
                flash(lorenz_result)
            except ZeroDivisionError:
                lorenz_result_error = 'Inputs resulted in an error. Try some more realistic numbers.'
                flash(lorenz_result_error)
            except OverflowError:
                lorenz_result_error = 'Inputs resulted in an error. Try some more realistic numbers.'
                flash(lorenz_result_error)

    elif 'ke' in request.form:
        if request.method == "POST" and ke_form.validate_on_submit():
            seebeck = float(ke_form.seebeck_ke.data) * float(ke_form.seebeck_units_ke.data)
            resistivity = float(ke_form.resistivity_ke.data) * float(ke_form.resistivity_units_ke.data)
            temperature = float(ke_form.temperature_ke.data) + float(ke_form.temperature_units_ke.data)
            
            r = float(ke_form.r_ke.data)
            try:
                rfl = efm.rfl_from_seebeck([seebeck], [r])[0]
                lorenz = efm.lorenz([rfl], [r])[0]
                ke_result = "{:.3f}".format(efm.electronic_thermal([lorenz], [resistivity], [temperature])[0])
                flash(ke_result)
            except ZeroDivisionError:
                ke_result_error = 'Inputs resulted in an error. Try some more realistic numbers.'
                flash(ke_result_error)
            except OverflowError:
                ke_result_error = 'Inputs resulted in an error. Try some more realistic numbers.'
                flash(ke_result_error)
            
            
    elif 'psi' in request.form:
        if request.method == "POST" and psi_form.validate_on_submit():
            seebeck = float(psi_form.seebeck_psi.data) * float(psi_form.seebeck_units_psi.data)
            r = float(psi_form.r_psi.data)
            try:
                rfl = efm.rfl_from_seebeck([seebeck], [r])[0]
                psi_result = "{:.3f}".format(efm.psi([rfl], [r])[0])
                flash(psi_result)
            except ZeroDivisionError:
                psi_result_error = 'Inputs resulted in an error. Try some more realistic numbers.'
                flash(psi_result_error)
            except OverflowError:
                psi_result_error = 'Inputs resulted in an error. Try some more realistic numbers.'
                flash(psi_result_error)
            

    elif 'beta' in request.form:
        if request.method == "POST" and beta_form.validate_on_submit():
            mu0 = float(beta_form.mu0_beta.data) * float(beta_form.mu0_units_beta.data)
            efmass = float(beta_form.efmass_beta.data) / float(beta_form.efmass_units_beta.data)
            temperature = float(beta_form.temperature_beta.data) + float(beta_form.temperature_units_beta.data)
            kl = float(beta_form.kl_beta.data) * float(beta_form.kl_units_beta.data)
            beta_result = "{:.3f}".format(efm.beta([mu0], [efmass], [temperature], [kl])[0])
            flash(beta_result)
            

    elif 'rfl_s' in request.form:
        if request.method == "POST" and rfl_s_form.validate_on_submit():
            seebeck = float(rfl_s_form.seebeck_rfl_s.data) * float(rfl_s_form.seebeck_units_rfl_s.data)
            r = float(rfl_s_form.r_rfl_s.data)
            try:
                rfl_result = "{:.3f}".format(efm.rfl_from_seebeck([seebeck], [r])[0])
                flash(rfl_result)
            except ZeroDivisionError:
                rfl_result_error = 'Inputs resulted in an error. Try some more realistic numbers.'
                flash(rfl_result_error)
            except OverflowError:
                rfl_result_error = 'Inputs resulted in an error. Try some more realistic numbers.'
                flash(rfl_result_error)
            

    elif 's_rfl' in request.form:
        if request.method == "POST" and s_rfl_form.validate_on_submit():
            rfl = float(s_rfl_form.rfl_s_rfl.data)
            r = float(s_rfl_form.r_s_rfl.data)
            seebeck_result = "{:.3f}".format(efm.seebeck_from_rfl([rfl], [r])[0])
            flash(seebeck_result)
            
    return render_template("index.html", **locals())


class ztForm(FlaskForm):
    seebeck_zt_rho = StringField(Markup('Seebeck coefficient:'), validators=[InputRequired('Seebeck input required'), floatCheck])
    seebeck_zt_rho_units = SelectField(u'Seebeck Units', choices=seebeck_units_choices)
    
    resistivity_zt_rho = StringField(Markup('Resistivity:'), validators=[InputRequired('Resistivity input required'), floatCheck])
    resistivity_zt_rho_units = SelectField(u'Resistivity Units', choices=resistivity_units_choices)
    
    temperature_zt_rho = StringField(Markup('Temperature:'), validators=[InputRequired('Temperature input required'), floatCheck])
    temperature_zt_rho_units = SelectField(u'Temperature Units', choices=temperature_units_choices)

    thermal_zt_rho = StringField(Markup('Total thermal conductivity:'), validators=[InputRequired('Thermal conductivity input required'), floatCheck])
    thermal_zt_rho_units = SelectField(u'Thermal Conductivity Units', choices=thermal_units_choices)

class zTExcelForm(FlaskForm):
    file_zT = FileField('File upload', validators=[FileRequired('File must be selected'), FileAllowed(['xls','xlsx'], '.xls or .xlsx files only')])

@app.route('/zt/', methods=['GET', 'POST'])
def zt():
    zt_form = ztForm()
    zt_excel_form = zTExcelForm()
    
    if 'zt_rho' in request.form:
        if request.method == "POST" and zt_form.validate_on_submit():
            # These values below are in standard units, uV/K, mOhm-cm, K, W/mK
            seebeck = float(zt_form.seebeck_zt_rho.data) * float(zt_form.seebeck_zt_rho_units.data)
            resistivity = float(zt_form.resistivity_zt_rho.data) * float(zt_form.resistivity_zt_rho_units.data)
            temperature = float(zt_form.temperature_zt_rho.data) + float(zt_form.temperature_zt_rho_units.data)
            thermal = float(zt_form.thermal_zt_rho.data) * float(zt_form.thermal_zt_rho_units.data)
            
            try:
                zT_result = "{:.3f}".format(((seebeck*1E-6)**2 * (temperature))\
                / (resistivity * 1E-5 * thermal))
                flash(zT_result)
            except ZeroDivisionError:
                zT_result_error = 'Inputs resulted in a divide by zero error. Try some different numbers.'
                flash(zT_result_error)
            except OverflowError:
                zT_result_error = 'Inputs resulted in an error. Try some more realistic numbers.'
                flash(zT_result_error)

    elif 'zt_excel' in request.form:
        if request.method == 'POST' and zt_excel_form.validate_on_submit():
            file = zt_excel_form.file_zT.data
            filename = secure_filename(file.filename)
            full_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            split_filename = full_file_path.rsplit('.',1)
            timestamp = str(time.time()).replace('.','_')
            filename_timestamped = split_filename[0] + '_' + timestamp + '.' + split_filename[1]
            file.save(filename_timestamped)
            
            try:
                zt_excel_path = efm_excel.zt_excel(filename_timestamped)
                flash('Upload and calculation succeeded.')
            except OverflowError:
                overflow_error = 'Excel file not formatted correctly (OverflowError)'
                flash(overflow_error)
            except IndexError:
                index_error = 'Excel file not formatted correctly (IndexError)'
                flash(index_error)
    return render_template("zt.html", **locals())
            
class spb_excel_Form(FlaskForm):
    file_spb = FileField('File upload', validators=[FileRequired('File must be selected'), FileAllowed(['xls','xlsx'], '.xls or .xlsx files only')])
    
@app.route('/spb-excel/', methods=['GET', 'POST'])
def spb_excel():
    spb_form = spb_excel_Form()
    
    if 'spb' in request.form:
        if request.method == "POST" and spb_form.validate_on_submit():
            file = spb_form.file_spb.data
            filename = secure_filename(file.filename)
            full_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            split_filename = full_file_path.rsplit('.',1)
            timestamp = str(time.time())
            filename_timestamped = split_filename[0] + '_' + timestamp + '.' + split_filename[1]
            file.save(filename_timestamped)
            try:
                spb_excel_path = efm_excel.calculate_spb(filename_timestamped)
                flash(spb_excel_path)
                
            except OverflowError:
                error_message = 'overflow error'
                flash(error_message)

            except IndexError:
                error_message = 'index error'
                flash(error_message)
                
    return render_template('spb_excel.html', **locals())
    
class theoretical_zt_Form(FlaskForm):
    efmass_tzt = StringField(Markup('Effective mass:'), validators=[InputRequired('Effective mass input required'), floatCheck])
    efmass_units_tzt = SelectField(u'Effective mass units', choices=effectivemass_units_choices)
    
    kl_tzt = StringField(Markup('Lattice thermal conductivity:'), validators=[InputRequired('Lattice thermal conductivity input required'), floatCheck])
    kl_units_tzt = SelectField(u'Lattice Thermal Conductivity units', choices=thermal_units_choices)
    
    mu_tzt = StringField(Markup('Mobility:'), validators=[InputRequired('Mobility input required'), floatCheck])
    mu_units_tzt = SelectField(u'Mobility units', choices=mobility_units_choices)
    
    temperature_tzt = StringField(Markup('Temperature:'), validators=[InputRequired('Temperature input required'), floatCheck])
    temperature_units_tzt = SelectField(u'Temperature units', choices=temperature_units_choices)
    
    r_tzt = StringField(Markup('Scattering parameter (r):'), validators=[InputRequired('Scattering parameter (r) input required'), floatCheck], default='-0.5')
    
    cc_lo_tzt = StringField(Markup('Carrier concentration lower limit (cm<sup>-3</sup>): 1&times;10^'), validators=[InputRequired('Lower carrier concentration limit required'), floatCheck], default='16')
    cc_hi_tzt = StringField(Markup('Carrier concentration upper limit (cm<sup>-3</sup>): 1&times;10^'), validators=[InputRequired('Upper carrier concentration limit required'), floatCheck], default='21')
    points_tzt = StringField(Markup('Number of points (integer):'), validators=[InputRequired('Number of data points required')], default='100')

class theoretical_zt_excel_Form(FlaskForm):
    file_tzt = FileField('File upload', validators=[FileRequired('File must be selected'), FileAllowed(['xls','xlsx'], '.xls or .xlsx files only')])
    
@app.route('/theoretical-zt/', methods=['GET','POST'])
def theoretical_zt():
    tzt_form = theoretical_zt_Form()
    tzt_excel_form = theoretical_zt_excel_Form()
    
    if 'tzt' in request.form:
        if request.method == "POST" and tzt_form.validate_on_submit():
            
            timestamp = time.time()
            # These values below are in standard units, uV/K, mOhm-cm, K, W/mK
            efmass = float(tzt_form.efmass_tzt.data) * float(tzt_form.efmass_units_tzt.data)
            kl = float(tzt_form.kl_tzt.data) * float(tzt_form.kl_units_tzt.data)
            mu = float(tzt_form.mu_tzt.data) * float(tzt_form.mu_units_tzt.data)
            temperature = float(tzt_form.temperature_tzt.data) + float(tzt_form.temperature_units_tzt.data)
            r = float(tzt_form.r_tzt.data)
            low_limit = abs(float(tzt_form.cc_lo_tzt.data))
            high_limit = abs(float(tzt_form.cc_hi_tzt.data))
            points = abs(int(tzt_form.points_tzt.data))
            
            zt_max = efm.theoretical_zt_max([efmass], [kl], [mu], [temperature], [r], low_limit, high_limit, points)  
            theoretical_zt_max_value = format(float(zt_max[0][0]),".3f")
            carrier_for_zt_max_value = "{:0.3e}".format(float(zt_max[1][0]))
            
            full_file_path_excel = os.path.join(app.config['UPLOAD_FOLDER'], 'theoretical_zt_plot_' + str(timestamp) + '.xlsx')
            export_data = [zt_max[2][0], zt_max[3][0]]
            export_labels = ['Carrier Concentration (cm^-3)', 'Theoretical zT']
            df_export = pd.DataFrame(export_data).transpose()
            df_export.columns = export_labels
            df_export.to_excel(full_file_path_excel, index=False)
        
            fig, ax = plt.subplots(1, figsize=(6,6))
            ax.plot(zt_max[2][0], zt_max[3][0], color="#0000FF")
            ax.scatter(zt_max[1], zt_max[0], color="#FF0000")
            ax.set_xlabel('Carrier Concentration (cm$^{-3}$)', fontsize=14)
            ax.set_ylabel('Theoretical zT', fontsize=14)
            ax.set_xscale('log')
            plt.tight_layout()
            full_file_path_plot = os.path.join(app.config['UPLOAD_FOLDER'], 'theoretical_zt_plot_' + str(timestamp) + '.png')
            plt.savefig(full_file_path_plot, dpi=500)
            
            if float(theoretical_zt_max_value) > 100:
                warning_message = "Data may be questionable (solver did not make good progress)"
                flash(warning_message)
            
            success = True
            flash(success)
            flash(theoretical_zt_max_value)
            flash(carrier_for_zt_max_value)
            flash(full_file_path_plot)
            flash(full_file_path_excel)
            
    if 'tzt_excel' in request.form:
        if request.method == "POST" and tzt_excel_form.validate_on_submit():
            file = tzt_excel_form.file_tzt.data
            filename = secure_filename(file.filename)
            full_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            split_filename = full_file_path.rsplit('.',1)
            timestamp = str(time.time())
            filename_timestamped = split_filename[0] + '_' + timestamp + '.' + split_filename[1]
            file.save(filename_timestamped)
            try:
                tzt_excel_path = efm_excel.theoretical_zt_max_excel(filename_timestamped)
                flash(tzt_excel_path)
                
            except OverflowError:
                error_message = 'Overflow error occured.'
                flash(error_message)
            except IndexError:
                error_message = 'Index error occured.'
                flash(error_message)
        
    return render_template('theoretical_zt.html', **locals())
    
@app.route('/plot/', methods=['GET', 'POST'])
def plot():
    timestamp = time.time()
    form_items = ['xdata', 'ydata']
    if request.method == 'GET':
        return render_template('plot.html')
    
    elif request.method == 'POST':
        for item in form_items:
            if item not in request.form:
                return render_template('plot.html', error=True, error_message='No data partition. id probably wrong.')
            
        xdata = request.form['xdata']
        ydata = request.form['ydata']
        form_data = [xdata, ydata]
        
        for data in form_data:
            if data == '':
                return render_template('plot.html', error=True, error_message='Text missing from at least one entry box. Try again.')
        
        xdata = xdata.split('\r\n')
        ydata = ydata.split('\r\n')
        
        if xdata[-1] == '':
            xdata = xdata[:-1]
        if ydata[-1] == '':
            ydata = ydata[:-1]
        if len(xdata) != len(ydata):
            return render_template('plot.html', error=True, error_message='Inputs are not the same length.')
        
        xdata = list(np.float_(xdata))
        ydata = list(np.float_(ydata))
        
        fig, ax = plt.subplots(1, figsize=(6,6))
        ax.plot(xdata, ydata, color="#0000FF")
        ax.set_xlabel('X axis', fontsize=14)
        ax.set_ylabel('Y-axis', fontsize=14)
        
        full_file_path_plot = os.path.join(app.config['UPLOAD_FOLDER'], 'user_input_plot' + str(timestamp) + '.png')
        plt.savefig(full_file_path_plot, dpi=500)
        
        plt.cla()
        
        ax.scatter(xdata, ydata, color="#0000FF")
        ax.set_xlabel('X axis', fontsize=14)
        ax.set_ylabel('Y-axis', fontsize=14)
        
        full_file_path_scatter = os.path.join(app.config['UPLOAD_FOLDER'], 'user_input_scatter' + str(timestamp) + '.png')
        plt.savefig(full_file_path_scatter, dpi=500)
        
        return render_template('plot.html', success=True, plot_file_location=full_file_path_plot, scatter_file_location=full_file_path_scatter)

class thermalForm(FlaskForm):
    diffusivity_k = StringField(Markup('Thermal diffusivity:'), validators=[InputRequired('Thermal diffusivity input required'), floatCheck])
    diffusivity_units_k = SelectField(u'Thermal diffusivity units', choices=diffusivity_units_choices)
    
    density_k = StringField(Markup('Density:'), validators=[InputRequired('Lattice thermal conductivity input required'), floatCheck])
    density_units_k = SelectField(u'Density units', choices=density_units_choices)
    
    cp_k = StringField(Markup('Heat capacity:'), validators=[InputRequired('Mobility input required'), floatCheck])
    cp_units_k = SelectField(u'Heat capacity units', choices=heat_capacity_units_choices)
    
class dulongPetiteForm(FlaskForm):
    composition_dp = StringField(Markup('Composition: '), validators=[InputRequired('Composition input required')])
    
@app.route('/thermal/', methods=['GET', 'POST'])
def thermal():
    k_form = thermalForm()
    dp_form = dulongPetiteForm()
    
    if 'thermal' in request.form:
        if request.method == "POST" and k_form.validate_on_submit():
            diffusivity = float(k_form.diffusivity_k.data) * float(k_form.diffusivity_units_k.data)
            density = float(k_form.density_k.data) * float(k_form.density_units_k.data)
            heat_capacity = float(k_form.cp_k.data) * float(k_form.cp_units_k.data)
            
            try: 
                thermal = "{:.3f}".format(diffusivity * density * heat_capacity)
                flash(thermal)
            except ZeroDivisionError:
                thermal_error_message = 'Divide by zero error encountered. Try different values.'
                flash(thermal_error_message)
    
    elif 'dulong_petit' in request.form:
        print(dp_form.validate_on_submit())
        if request.method == "POST" and dp_form.validate_on_submit():
            R = 8.314 # J/mol*K
            try:
                composition = formula(dp_form.composition_dp.data)
                molar_mass = float(composition.mass) / 1000
                num_atoms = sum(composition.atoms.values())
                heat_capacity = "{:.5f}".format((3 * num_atoms * R) / molar_mass)
                
                composition = str(composition)
                molar_mass = "{:.5f}".format(molar_mass)
                num_atoms = "{:.5f}".format(num_atoms)
                
                flash(composition)
                flash(molar_mass)
                flash(num_atoms)
                flash(heat_capacity)
                
            except ValueError:
                dp_error_message = 'An element was not recognized. Try adjusting your formula.'
                flash(dp_error_message)
                
        
    return render_template("thermal.html", **locals())

@app.route('/thermal/dulong_petit', methods=['POST'])
def dulong_petit():
    form_items = ['n_atoms', 'm', 'm_units']
    molar_mass_dict = {'gmol': 1E-3, 'kgmol': 1}
    
    if request.method == "POST":
        for item in form_items:
            if item not in request.form:
                return render_template("thermal.html", \
                dulong_petit_error=True, error_message="No file partition")
            else:
                continue

        n_atoms = request.form['n_atoms']
        molar_mass = request.form['m']
        molar_mass_units = request.form['m_units']
        
        form_items_list = [n_atoms, molar_mass, molar_mass_units]
        
        
        for item in form_items_list:
            if item == '':        
                return render_template("thermal.html", \
                dulong_petit_error=True, error_message="An input field was left blank")
            else:
                continue
        try:
            n_atoms = float(n_atoms)
            molar_mass = float(molar_mass)
            
        except ValueError:
            return render_template("thermal.html", \
            dulong_petit_error=True, error_message="Invalid inputs. Inputs must be numeric.")
        
        R = 8.314 # gas constant. J/molK
        heat_capacity = "{:.3f}".format((3 * n_atoms * R) / (molar_mass * molar_mass_dict[molar_mass_units]))
        
        return render_template("thermal.html", output=heat_capacity, dulong_petit_success=True)

if __name__ == "__main__":
    app.debug = True
    app.run()
    