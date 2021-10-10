from flask import Flask, request
import pandas as pd
from _collections import OrderedDict
import joblib

app=Flask(__name__)

@app.route('/api/clustering')
def get():
    gender = str(request.args['gender'])
    age = str(request.args['age'])
    sector = str(request.args['sector'])
    industry = str(request.args['industry'])
    education = str(request.args['education'])
    income = float(request.args['income'])
    make = str(request.args['make'])
    type = str(request.args['type'])
    model = str(request.args['model'])

    outFileFolder = './Clustering/'
    filePath = outFileFolder + 'ClusteringModel.joblib'

    #open file
    file = open(filePath, "rb")

    #load the trained model
    trained_model = joblib.load(file)    
    
    new_data=OrderedDict([('gender',gender),('dob',age),
    ('sector',sector),('industry',industry),('education',education),
	('income',income),('vmake',make),('vtype',type),('vmodel',model)])
	
    new_data=pd.Series(new_data).values.reshape(1,-1)
    catColumnsPos = [0, 1, 2, 3, 4, 6, 7, 8]
    prediction = trained_model.predict(new_data, categorical = catColumnsPos)

    return str(prediction).strip('[]')

if __name__ == '__main__':
    app.run()