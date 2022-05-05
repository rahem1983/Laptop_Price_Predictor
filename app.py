from ast import If
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request, render_template
import pickle


pipe = pickle.load(open('pipeLR.pkl','rb'))

app = Flask(__name__)

@app.route("/predict", methods=['POST'])
def prediction():
    if request.method == 'POST':
        data = request.form.to_dict()
        # return data
        company = data['company']
        type = data['type']
        ram = data['ram']
        weight = data['weight']
        touchscreen = data['touchscreen']
        ips = data['ips']
        screen_size = data['screen_size']
        resolution = data['resolution']
        cpu = data['cpu']
        hdd = data['hdd']
        ssd = data['ssd']
        gpu = data['gpu']
        os = data['os']

        
        
        ppi = None
        if touchscreen == 'Yes':
            touchscreen = 1
        else:
            touchscreen = 0

        if ips == 'Yes':
            ips = 1
        else:
            ips = 0

        X_res = int(resolution.split('x')[0])
        Y_res = int(resolution.split('x')[1])
        ppi = ((X_res**2) + (Y_res**2))**0.5/float(screen_size)

        
        query = np.array([company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])

        query = query.reshape(1,12)
        output = int(np.exp(pipe.predict(query)[0]))
        response = {
            'value': output
        }
        return response
    else :
        return "not called"

    

@app.route("/")
def landing():
    return render_template('/landing.html')

@app.route("/temp")
def temp():
    return render_template('/temp.html')