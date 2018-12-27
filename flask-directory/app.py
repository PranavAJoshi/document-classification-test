import flask
import pickle
from flask import Flask, json, request

app = Flask(__name__)
modelname = 'LRmodel.pkl'
vecname = 'vec.pkl'

@app.route('/')
def index():
    return flask.render_template('index.html')
    
@app.route('/words', methods=['GET', 'POST'])
def form():
    if request.method == 'POST':
        data = request.form['content']
        model = pickle.load(open(modelname, 'rb'))
        vec = pickle.load(open(vecname, 'rb'))
        result = getresult(model, vec, data)[0]
        return flask.render_template('result.html', result=result)
    
def getresult(model, vec, data):
    transformed_data = vec.transform([data])
    return model.predict(transformed_data)
    
    
if __name__ == "__main__":
    app.run()
