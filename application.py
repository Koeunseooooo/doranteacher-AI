from urllib import response
from flask import Flask, jsonify, request
from flask_restx import Resource, Api, reqparse
from flask_cors import CORS
from comment import comment
from correct import correct
from recommend import recommend
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

application = Flask(__name__)
# 보안관련
CORS(application)
api = Api(application)


@application .route('/dd')
def index():
    return 'Hello'


# @api.route('/test')
# class testAPI(Resource):
#     def get(self):
#         return jsonify({"result": "테스트 완료 from flask"})


@api.route('/recommend')
class recommendAPI(Resource):
    def get(self):
        # req = "내가 제일 좋아하는 음식은 햄버거이다. 그래서 오늘은 햄버거가게에 가서 햄버거를 먹었다. 감자튀김도 들어있는 햄버거세트로 먹었다. 정말 배부르고 맛있었다. 매일 먹고싶지만 그러면 체중이 늘어나겠지? 그래도 매일 매일 먹고싶다"
        # recommend(req)
        return jsonify({"result": "get방식 from flask /recommend"})

    def post(self):
        req = request.json.get('text')
        result = recommend(req)
        # result = jsonify({"result": res})
        return result


@api.route('/test')
class testAPI(Resource):
    def get(self):
        # req = "내가 제일 좋아하는 음식은 햄버거이다. 그래서 오늘은 햄버거가게에 가서 햄버거를 먹었다. 감자튀김도 들어있는 햄버거세트로 먹었다. 정말 배부르고 맛있었다. 매일 먹고싶지만 그러면 체중이 늘어나겠지? 그래도 매일 매일 먹고싶다"
        # recommend(req)
        return jsonify({"result": "get방식 from flask /test"})

    def post(self):
        req = request.json.get('text')
        res = test()
        result = jsonify({"result": res})
        return result

@api.route('/comment')
class commentAPI(Resource):
    def get(self):
        return jsonify({"result": "get방식 from flask /comment"})

    def post(self):
        req = request.json.get('text')
        res = comment(req)
        return res


@api.route('/correct')
class correctAPI(Resource):
    def get(self):
        return jsonify({"result": "get방식 from flask /correct"})

    def post(self):
        req = request.json.get('text')
        print(req)
        original, corrected, correct_string = correct(req)
        result = jsonify(
            {"original": original, "corrected": corrected, "correct_string": correct_string})

        return result


if __name__ == '__main__':
    application.run(host='0.0.0.0', port='8080')
