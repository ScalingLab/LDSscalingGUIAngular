from flask import Flask, request, send_file, render_template
from flask_cors import CORS, cross_origin
from flask_restful import Resource, Api
from json import dumps
from flask_jsonpify import jsonify
import pandas as pd
from pathlib import Path
from costs2 import costs2
#from scalingPolicy import scalingPolicy
from Predictor2 import Predictor2
import base64

app = Flask(__name__)
api = Api(app)

CORS(app)

base_dir = str(Path().resolve())

names = ['Conf', 'profile', 'Cuser', 'rate', 'RT', 'KO', 'Nrequest', '%KO', 'RTStdDev']
#ConfTable=['1WPmedium_1DBlarge ']
ConfTable=['1WPmedium_1DBmedium ','3WPmedium_1DBmedium ','1WPmedium_1DBlarge ','2WPmedium_1DBlarge ','3WPmedium_1DBlarge ']#,'1WPlarge_1DB ']
#ConfTable=['1WPmedium_1DBmedium ','2WPmedium_1DBmedium ','3WPmedium_1DBmedium ']
#CostTable={'medium':0.0288,'large':0.0576,'xlarge':0.1152,'xxlarge':0.2304,'':0.2304}
CostTable={'medium':0.0576,'large':0.1152,'xlarge':0.2304,'xxlarge':0.2304,'':0.2304}

Profiles=[' author ',' editor ',' shopmanager ', ' userreader ']
MixtureProfiles = [' mixturea ',' mixtureb ']
Mixtures=[[0.25,0.25,0.25,0.25],[0.1,0.2,0.2,0.5]]
#Profiles=[' userreader ']

file = base_dir + '/dataset/stats.csv'
df = pd.read_csv(file, names=names, header=None)
costs = costs2(CostTable)
predictor = Predictor2(df, ConfTable, Profiles, Mixtures, MixtureProfiles, costs)

mix=0
predictor.predict(mix)

@app.route("/")
def measuredKO():
    img = predictor.plotCTRL('KO','measured')
    img = base64.b64encode(img.getvalue())
    return jsonify(img)

@app.route("/measuredRT")
def measuredRT():
    img = predictor.plotCTRL('RT','measured')
    img = base64.b64encode(img.getvalue())
    return jsonify(img)

@app.route("/measuredRate")
def measuredRate():
    img = predictor.plotCTRL('Rate','measured')
    img = base64.b64encode(img.getvalue())
    return jsonify(img)

@app.route("/measuredMAR")
def measuredMAR():
    img = predictor.plot_MAR(MixtureProfiles[predictor.MixID],'measured')
    img = base64.b64encode(img.getvalue())
    return jsonify(img)

@app.route("/predictedKO")
def predictedKO():
    img = predictor.plotCTRL('KO','predicted')
    img = base64.b64encode(img.getvalue())
    return jsonify(img)

@app.route("/predictedRT")
def predictedRT():
    img = predictor.plotCTRL('RT','predicted')
    img = base64.b64encode(img.getvalue())
    return jsonify(img)

@app.route("/predictedRate")
def predictedRate():
    img = predictor.plotCTRL('Rate','predicted')
    img = base64.b64encode(img.getvalue())
    return jsonify(img)

@app.route("/predictedMAR")
def predictedMAR():
    img = predictor.plot_MAR(MixtureProfiles[predictor.MixID],'predicted')
    img = base64.b64encode(img.getvalue())
    return jsonify(img)

@app.route("/errorKO")
def errorKO():
    img = predictor.plotCTRL('KO','error')
    img = base64.b64encode(img.getvalue())
    return jsonify(img)

@app.route("/errorRT")
def errorRT():
    img = predictor.plotCTRL('RT','error')
    img = base64.b64encode(img.getvalue())
    return jsonify(img)

@app.route("/errorRate")
def errorRate():
    img = predictor.plotCTRL('Rate','error')
    img = base64.b64encode(img.getvalue())
    return jsonify(img)


if __name__ == '__main__':
     app.run(port=5002)