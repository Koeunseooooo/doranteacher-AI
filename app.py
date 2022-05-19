from flask import Flask, jsonify, request
from flask_restx import Resource, Api, reqparse
import sklearn
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)
api = Api(app)
app.config['DEBUG'] = True


@app.route('/dd')
def index():
    return 'Hello World!'


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


if __name__ == '__main__':
    app.run(debug=True)
    # app.run(host='127.0.0.1', port=8080)
