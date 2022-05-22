from flask import Flask, jsonify, request
from flask_restx import Resource, Api, reqparse
import sklearn
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from flask_cors import CORS
from recommendAPI import recommendAPI
from commentAPI import commentAPI

app = Flask(__name__)
app.debug = True
# 보안관련
CORS(app)
api = Api(app)

idx = 0


@app.route('/dd')
def index():
    return 'Hello'


@api.route('/test')
class testAPI(Resource):
    def get(self):
        return jsonify({"result": "테스트 완료 from flask"})


@api.route('/recommend')
class recommendAPI(Resource):
    def get(self):
        return jsonify({"result": "get방식 from flask /recommend"})

    def post(self):
        global idx
        idx += 2
        req = request.json.get('doc')
        # recommendAPI()
        # print(req)
        result = jsonify({"idx": idx, "doc": req+"받음"})
        return result


@api.route('/comment')
class commentAPI(Resource):
    def get(self):
        return jsonify({"result": "get방식 from flask /comment"})

    def post(self):
        global idx
        idx += 2
        req = request.json.get('doc')
        # commentAPI()
        # print(req)
        result = jsonify({"idx": idx, "doc": req+"받음"})
        return result


if __name__ == '__main__':
    app.run(debug=True)
