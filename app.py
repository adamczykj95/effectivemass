from flask import Flask, render_template, request, url_for
import efm
import os
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
import time
import pandas as pd
import warnings

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

@app.route('/', methods=['GET'])
def index():
    # Displays the index page accessible at '/'
    return render_template("index.html")
    
@app.route('/index/effective_mass', methods=['POST'])
def index_efmass():
    # Displays the index page accessible at '/'
    form_items = ["seebeck", "cc", "T", "r"]
    
    if request.method == "POST":
        for item in form_items:
            if item not in request.form:
                return render_template("index.html", \
                index_efmass=False, error_message="No file partition")
            else:
                continue
                
        seebeck = request.form["seebeck"]
        carrier = request.form["cc"]
        temperature = request.form["T"]
        r = request.form["r"]
        
        form_items_list = [seebeck, carrier, temperature, r]
        
        for item in form_items_list:
            if item == '':        
                return render_template("index.html", \
                index_efmass=False, error_message="An input field was left blank")
            else:
                continue
        try:
            seebeck = float(seebeck)
            carrier = float(carrier)
            temperature = float(temperature)
            r = float(r)
        except ValueError:
            return render_template("index.html", \
            index_efmass=False, error_message="Invalid inputs. Inputs must be numeric.")
        rfl = efm.rfl_from_seebeck([seebeck], [r])[0]
        effective_mass = "{:.3f}".format(efm.efm([rfl], [carrier], [temperature], [r])[0])
    return render_template("index.html", index_efmass=True, output=effective_mass)

@app.route('/index/intrinsic_mobility', methods=['POST'])
def index_mu0():
    # Displays the index page accessible at '/'
    form_items = ["seebeck", "mu", "r"]
    
    if request.method == "POST":
        for item in form_items:
            if item not in request.form:
                return render_template("index.html", \
                index_mu0=False, error_message="No file partition")
            else:
                continue
                
        seebeck = request.form["seebeck"]
        mu = request.form["mu"]
        r = request.form["r"]
        
        form_items_list = [seebeck, mu, r]
        
        for item in form_items_list:
            if item == '':        
                return render_template("index.html", \
                index_mu0=False, error_message="An input field was left blank")
            else:
                continue
        try:
            seebeck = float(seebeck)
            mu = float(mu)
            r = float(r)
        except ValueError:
            return render_template("index.html", \
            index_mu0=False, error_message="Invalid inputs. Inputs must be numeric.")
        rfl = efm.rfl_from_seebeck([seebeck], [r])[0]
        mu0 = "{:.3f}".format(efm.mu0([rfl], [mu], [r])[0])
    return render_template("index.html", index_mu0=True, output=mu0)

@app.route('/index/lorenz_number', methods=['POST'])
def index_lorenz():
    # Displays the index page accessible at '/'
    form_items = ["seebeck"]
    
    if request.method == "POST":
        for item in form_items:
            if item not in request.form:
                return render_template("index.html", \
                index_lorenz=False, error_message="No file partition")
            else:
                continue
                
        seebeck = request.form["seebeck"]
        r = request.form["r"]
        
        form_items_list = [seebeck, r]
        
        for item in form_items_list:
            if item == '':        
                return render_template("index.html", \
                index_lorenz=False, error_message="An input field was left blank")
            else:
                continue
        try:
            seebeck = float(seebeck)
            r = float(r)
        except ValueError:
            return render_template("index.html", \
            index_lorenz=False, error_message="Invalid inputs. Inputs must be numeric.")
        rfl = efm.rfl_from_seebeck([seebeck], [r])[0]
        lorenz = "{:.3e}".format(efm.lorenz([rfl], [r])[0])
    return render_template("index.html", index_lorenz=True, output=lorenz)

@app.route('/index/electronic_thermal_conductivity', methods=['POST'])
def index_ke():
    # Displays the index page accessible at '/'
    form_items = ["seebeck", "resistivity", "T", "r"]
    
    if request.method == "POST":
        for item in form_items:
            if item not in request.form:
                return render_template("index.html", \
                index_ke=False, error_message="No file partition")
            else:
                continue
                
        seebeck = request.form["seebeck"]
        resistivity = request.form["resistivity"]
        temperature = request.form["T"]
        r = request.form["r"]
        
        form_items_list = [seebeck, resistivity, temperature, r]
        
        for item in form_items_list:
            if item == '':        
                return render_template("index.html", \
                index_ke=False, error_message="An input field was left blank")
            else:
                continue
        try:
            seebeck = float(seebeck)
            resistivity = float(resistivity)
            temperature = float(temperature)
            r = float(r)
            
        except ValueError:
            return render_template("index.html", \
            index_ke=False, error_message="Invalid inputs. Inputs must be numeric.")
        rfl = efm.rfl_from_seebeck([seebeck], [r])[0]
        lorenz = efm.lorenz([rfl], [r])[0]
        ke = "{:.3f}".format(efm.electronic_thermal([lorenz], [resistivity], [temperature])[0])
        
    return render_template("index.html", index_ke=True, output=ke)

@app.route('/index/psi', methods=['POST'])
def index_psi():
    # Displays the index page accessible at '/'
    form_items = ["seebeck", "r"]
    
    if request.method == "POST":
        for item in form_items:
            if item not in request.form:
                return render_template("index.html", \
                index_psi=False, error_message="No file partition")
            else:
                continue
                
        seebeck = request.form["seebeck"]
        r = request.form["r"]
        
        form_items_list = [seebeck, r]
        
        for item in form_items_list:
            if item == '':        
                return render_template("index.html", \
                index_psi=False, error_message="An input field was left blank")
            else:
                continue
        try:
            seebeck = float(seebeck)
            r = float(r)
            
        except ValueError:
            return render_template("index.html", \
            index_psi=False, error_message="Invalid inputs. Inputs must be numeric.")
        rfl = efm.rfl_from_seebeck([seebeck], [r])[0]
        psi = "{:.3f}".format(efm.psi([rfl], [r])[0])
        
    return render_template("index.html", index_psi=True, output=psi)

@app.route('/index/beta', methods=['POST'])
def index_beta():
    # Displays the index page accessible at '/'
    form_items = ["mu0", "efmass", "T", "kl"]
    
    if request.method == "POST":
        for item in form_items:
            if item not in request.form:
                return render_template("index.html", \
                index_beta=False, error_message="No file partition")
            else:
                continue
                
        mu0 = request.form["mu0"]
        efmass = request.form["efmass"]
        temperature = request.form["T"]
        kl = request.form["kl"]
        
        form_items_list = [mu0, efmass, temperature, kl]
        
        for item in form_items_list:
            if item == '':        
                return render_template("index.html", \
                index_beta=False, error_message="An input field was left blank")
            else:
                continue
        try:
            mu0 = float(mu0)
            efmass = float(mu0)
            temperature = float(temperature)
            kl = float(kl)
            
        except ValueError:
            return render_template("index.html", \
            index_beta=False, error_message="Invalid inputs. Inputs must be numeric.")
        beta = "{:.3f}".format(efm.beta([mu0], [efmass], [temperature], [kl])[0])
    return render_template("index.html", index_beta=True, output=beta)

@app.route('/index/rfl_from_seebeck', methods=['POST'])
def index_rfl_s():
    # Displays the index page accessible at '/'
    form_items = ["seebeck", "r"]
    
    if request.method == "POST":
        for item in form_items:
            if item not in request.form:
                return render_template("index.html", \
                index_rfl_s=False, error_message="No file partition")
            else:
                continue
                
        seebeck = request.form["seebeck"]
        r = request.form["r"]
        
        form_items_list = [seebeck, r]
        
        for item in form_items_list:
            if item == '':        
                return render_template("index.html", \
                index_rfl_s=False, error_message="An input field was left blank")
            else:
                continue
        try:
            seebeck = float(seebeck)
            r = float(r)
            
        except ValueError:
            return render_template("index.html", \
            index_rfl_s=False, error_message="Invalid inputs. Inputs must be numeric.")
        rfl = "{:.3f}".format(efm.rfl_from_seebeck([seebeck], [r])[0])
    return render_template("index.html", index_rfl_s=True, output=rfl)

@app.route('/index/seebeck_from_rfl', methods=['POST'])
def index_s_rfl():
    # Displays the index page accessible at '/'
    form_items = ["rfl", "r"]
    
    if request.method == "POST":
        for item in form_items:
            if item not in request.form:
                return render_template("index.html", \
                index_s_rfl=False, error_message="No file partition")
            else:
                continue
                
        rfl = request.form["rfl"]
        r = request.form["r"]
        
        form_items_list = [rfl, r]
        
        for item in form_items_list:
            if item == '':        
                return render_template("index.html", \
                index_s_rfl=False, error_message="An input field was left blank")
            else:
                continue
        try:
            rfl = float(rfl)
            r = float(r)
            
        except ValueError:
            return render_template("index.html", \
            index_s_rfl=False, error_message="Invalid inputs. Inputs must be numeric.")
        seebeck = "{:.3f}".format(efm.seebeck_from_rfl([rfl], [r])[0])
    return render_template("index.html", index_s_rfl=True, output=seebeck)



@app.route('/action/', methods=['POST'])
def action():
    # Displays the index page accessible at '/'
    r = float(request.form["r"])
    seebeck = float(request.form["seebeck"])
    rfl = round(efm.rfl_from_seebeck([seebeck], [r])[0],3)
    
    return render_template("index.html", output=rfl, success=True)
    
@app.route('/zt/', methods=['GET'])
def zt():
    # Displays the index page accessible at '/'
    return render_template("zt.html")

@app.route('/zt/', methods=['POST'])
def zt_calculated_resistivity():
    form_items = ["s_units", "s", "r_units", "r", "T", "k"]
    seebeck_dict = {"uV/K": 1E-6, "mV/K": 1E-3, "V/K": 1}
    resistivity_dict = {"mohm-cm":1E-5, "ohm-cm":1E-2, "ohm-m":1}
    temperature_dict = {"kelvin":0, "centigrade":273.15}
    
    if request.method == "POST":
        for item in form_items:
            if item not in request.form:
                return render_template("zt.html", \
                error=True, error_message="No file partition")
            else:
                continue

        seebeck_units = request.form["s_units"]
        seebeck = request.form["s"]
        resistivity_units = request.form["r_units"]
        resistivity = request.form["r"]
        temperature_units = request.form["T_units"]
        temperature = request.form["T"]
        thermal = request.form["k"]
        
        form_items_list = [seebeck_units, seebeck, resistivity_units,\
        resistivity, temperature_units, temperature, thermal]
        
        for item in form_items_list:
            if item == '':        
                return render_template("zt.html", \
                error=True, error_message="An input field was left blank")
            else:
                continue
        try:
            seebeck = float(seebeck)
            resistivity = float(resistivity)
            temperature = float(temperature)
            thermal = float(thermal)
        except ValueError:
            return render_template("zt.html", \
            error=True, error_message="Invalid inputs. Inputs must be numeric.")
        
        if isinstance(seebeck_units,str) == False:
            return render_template("zt.html", \
            error=True, error_message="Invalid seebeck unit inputs. Must be string.")
            
        elif isinstance(resistivity_units,str) == False:
            return render_template("zt.html", \
            error=True, error_message="Invalid resistivity unit inputs. Must be string.")
        
        elif isinstance(temperature_units,str) == False:
            return render_template("zt.html", \
            error=True, error_message="Invalid temperature unit inputs. Must be string.")
            
        zT = round(((seebeck * seebeck_dict[seebeck_units])**2 * (temperature + temperature_dict[temperature_units])) / \
        (resistivity * resistivity_dict[resistivity_units] * thermal),3)
        
        return render_template("zt.html", output=zT, success=True)

@app.route('/zt_excel/', methods=['POST'])
def zt_from_excel():

    allowed_extensions = {'xlsx', 'xls'}
    
    if request.method == 'POST':
        if "zt_file" not in request.files:
            return render_template('zt.html', upload_error=True, upload_error_message="No file partition")
        
        file = request.files["zt_file"]
        if file.filename == '':
            return render_template('zt.html', upload_error=True, upload_error_message="No file selected")
        
        if file and not allowed_file(file.filename, allowed_extensions):
            return render_template('zt.html', upload_error=True, upload_error_message="File type not allowed")
        
        if file and allowed_file(file.filename, allowed_extensions):
            filename = secure_filename(file.filename)
            full_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            split_filename = full_file_path.rsplit('.',1)
            timestamp = str(time.time())
            filename_timestamped = split_filename[0] + '_' + timestamp + '.' + split_filename[1]
            file.save(filename_timestamped)
            try:
                zt_excel_path = efm.zt_excel(filename_timestamped)
                return render_template('zt.html', upload_success=True, download_path=zt_excel_path)
            except OverflowError:
                return render_template('zt.html', upload_error=True, upload_error_message="Excel file not formatted correctly (OverflowError)")
            except IndexError:
                return render_template('zt.html', upload_error=True, upload_error_message="Excel file not formatted correctly (IndexError)")

@app.route('/<name>')
def zt_download(name=''):
    return send_from_directory(app.config['UPLOAD_FOLDER'], name)

@app.route('/spb/', methods=['GET'])
def spb():
    return render_template('spb_models.html')
    
@app.route('/spb/', methods=['POST'])
def spb_calculate_file():

    allowed_extensions = {'xlsx', 'xls'}
    
    if request.method == 'POST':
        if "spb_file" not in request.files:
            return render_template('spb_models.html', error=True, error_message="No file partition")
        
        file = request.files["spb_file"]
        if file.filename == '':
            return render_template('spb_models.html', error=True, error_message="No file selected")
        
        if file and not allowed_file(file.filename, allowed_extensions):
            return render_template('spb_models.html', error=True, error_message="File type not allowed")
        
        if file and allowed_file(file.filename, allowed_extensions):
            filename = secure_filename(file.filename)
            full_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            split_filename = full_file_path.rsplit('.',1)
            timestamp = str(time.time())
            filename_timestamped = split_filename[0] + '_' + timestamp + '.' + split_filename[1]
            file.save(filename_timestamped)
            try:
                spb_excel_path = efm.calculate_spb(filename_timestamped)
                return render_template('spb_models.html', success=True, download_path=spb_excel_path)
            except OverflowError:
                return render_template('spb_models.html', error=True, error_message="Excel file not formatted correctly (OverflowError)")
            except IndexError:
                return render_template('spb_models.html', error=True, error_message="Excel file not formatted correctly (IndexError)")

@app.route('/<name>')
def spb_download(name=''):
    return send_from_directory(app.config['UPLOAD_FOLDER'], name)
    
@app.route('/theoretical-zt/', methods=['GET'])
def theoretical_zt():
    return render_template('theoretical_zt.html')
    
@app.route('/theoretical-zt/', methods=['POST'])
def theoretical_zt_calculated():
    timestamp = time.time()
    form_items = ["efm", "efm_units", "kl", "mu", "T", "T_units"]
    efmass_dict = {"m*/m0": 1, "kg": 9.10938356E-31}
    temperature_dict = {"kelvin":0, "centigrade":273.15}
    
    for item in form_items:
        if item not in request.form:
            return render_template("theoretical_zt.html", \
            error=True, error_message="No file partition")
        else:
            continue

    efmass_units = request.form["efm_units"]
    efmass = request.form["efm"]
    temperature_units = request.form["T_units"]
    temperature = request.form["T"]
    kl = request.form["kl"]
    mu = request.form["mu"]
    r = request.form["r"]
    
    form_items_list = [efmass_units, efmass, temperature_units, temperature, kl, mu, r]
    
    for item in form_items_list:
        if item == '':        
            return render_template("theoretical_zt.html", \
            error=True, error_message="An input field was left blank")
        else:
            continue
    try:
        efmass = float(efmass)
        temperature = float(temperature)
        kl = float(kl)
        mu = float(mu)
        r = float(r)
    except ValueError:
        return render_template("theoretical_zt.html", \
        error=True, error_message="Invalid inputs. Inputs must be numeric.")
    
    if isinstance(efmass_units,str) == False:
        return render_template("zt.html", \
        error=True, error_message="Invalid seebeck unit inputs. Must be string.")
        
    elif isinstance(temperature_units,str) == False:
        return render_template("zt.html", \
        error=True, error_message="Invalid resistivity unit inputs. Must be string.")
    
    try:
        warnings.filterwarnings("error")
        zt_max = efm.theoretical_zt_max([efmass]*efmass_dict[efmass_units], [kl], [mu],\
            [temperature + temperature_dict[temperature_units]], [r])
        warnings.filterwarnings("ignore")    
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
        ax.set_xlim(1E16,1E22)
        
        full_file_path_plot = os.path.join(app.config['UPLOAD_FOLDER'], 'theoretical_zt_plot_' + str(timestamp) + '.png')
        plt.savefig(full_file_path_plot, dpi=500)

        return render_template("theoretical_zt.html", zt=theoretical_zt_max_value, \
        carrier=carrier_for_zt_max_value, plot=full_file_path_plot,\
        excel=full_file_path_excel, success=True)
        
    except RuntimeWarning:
        return render_template("theoretical_zt.html", warning=True)

@app.route('/<name>')
def theoretical_zt_plot(name=''):
    return send_from_directory(name)

@app.route('/theoretical_zt_excel/', methods=['POST'])
def theoretical_zt_from_excel():

    allowed_extensions = {'xlsx', 'xls'}
    
    if request.method == 'POST':
        if "tzt_file" not in request.files:
            return render_template('theoretical_zt.html', upload_error=True, upload_error_message="No file partition")
        
        file = request.files["tzt_file"]
        if file.filename == '':
            return render_template('theoretical_zt.html', upload_error=True, upload_error_message="No file selected")
        
        if file and not allowed_file(file.filename, allowed_extensions):
            return render_template('theoretical_zt.html', upload_error=True, upload_error_message="File type not allowed")
        
        if file and allowed_file(file.filename, allowed_extensions):
            filename = secure_filename(file.filename)
            full_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            split_filename = full_file_path.rsplit('.',1)
            timestamp = str(time.time())
            filename_timestamped = split_filename[0] + '_' + timestamp + '.' + split_filename[1]
            file.save(filename_timestamped)
            try:
                tzt_excel_path = efm.theoretical_zt_max_excel(filename_timestamped)
                return render_template('theoretical_zt.html', upload_success=True, download_path=tzt_excel_path)
            except OverflowError:
                return render_template('theoretical_zt.html', upload_error=True, upload_error_message="Excel file not formatted correctly (OverflowError)")
            except IndexError:
                return render_template('theoretical_zt.html', upload_error=True, upload_error_message="Excel file not formatted correctly (IndexError)")

@app.route('/<name>')
def download(name=''):
    return send_from_directory(name)

@app.route('/thermal/', methods=['GET'])
def thermal():
    return render_template('thermal.html')

@app.route('/thermal/', methods=['POST'])
def thermal_calculated():
    return render_template('thermal.html')

if __name__ == "__main__":
    app.debug = True
    app.run()
    