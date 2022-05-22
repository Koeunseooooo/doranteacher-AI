from flask import Flask, jsonify, request
from flask_restx import Resource, Api, reqparse
import sklearn
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from flask_cors import CORS
from paint import paint

app = Flask(__name__)
app.debug = True
# 보안관련
CORS(app)
api = Api(app)


@app.route('/dd')
def index():
    return 'Hello'


@api.route('/test')
class testAPI(Resource):
    def get(self):
        return jsonify({"result": "연결 잘 됨 from flask"})

    def post(self):
        iris = load_iris()
        parsed_request = request.json.get('content')
        result = iris.feature_names
        print(parsed_request)
        return result


@api.route('/paint-rec')
class paintAPI(Resource):
    def get(self):
        a = paint()
        return jsonify({"result": a})

    def post(self):
        iris = load_iris()
        parsed_request = request.json.get('content')
        result = iris.feature_names
        print(parsed_request)
        return result


if __name__ == '__main__':
    app.run(debug=True)
