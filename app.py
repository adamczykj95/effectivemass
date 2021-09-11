from flask import Flask, render_template, request, url_for
import efm

app = Flask(__name__)
app.config['SECRET_KEY'] = '3508sdfnl3nljnse20851j0adljnsd 0j123_+!#%(*@4j0182@$)*'

@app.route('/', methods=['GET'])
def index():
    # Displays the index page accessible at '/'
    return render_template("index.html")
    

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
        temperature = request.form["T"]
        thermal = request.form["k"]
        
        form_items_list = [seebeck_units, seebeck, resistivity_units,\
        resistivity, temperature, thermal]
        
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
            
        zT = round(((seebeck * seebeck_dict[seebeck_units])**2 * temperature) / \
        (resistivity * resistivity_dict[resistivity_units] * thermal),3)
        
        return render_template("zt.html", output=zT, success=True)

    
if __name__ == "__main__":
    app.debug = True
    app.run()
    