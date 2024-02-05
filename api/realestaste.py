from flask import Blueprint, jsonify  # jsonify creates an endpoint response object
from flask_restful import Api, request, Resource # used for REST API building
import requests  # used for testing
import random
from __init__ import app, db
from model.realEstateModels import House
# from ml.RealEstateRecEngine import RecEngine
from ai.OpenAIEngine import HouseAIEngine
import os
from cryptography.fernet import Fernet

realestate_api = Blueprint('house', __name__, url_prefix='/api/house')


api = Api(realestate_api)


class houses:

    class _getHouses(Resource):
        def get(self):
            houses = db.session.query(House).all()
            print("Hi")
            return jsonify([house.few_details() for house in houses])
        
    class _gethousedetails(Resource):
        def get(self):
            house = db.session.query(House).filter(House.id == int(request.args.get("id"))).first()
            return jsonify(house.all_details())
        
    class _getOpenAIResponse(Resource):
        model = HouseAIEngine()
        
        def get(self):
            encrypted_code = 'gAAAAABlOL7JjQ7xkp55n2f2BkZ1KBaDu-bQgmbRJU7w6H25i2ImrdLwDcGPpD2YZejoTVAWSXWEM4vJLAB7r2YaXS9UAKOFjw=='
            crypto_key = os.getenv("CRYPTO_KEY")
            print(crypto_key)
            if crypto_key is None:
                raise ValueError("CRYPTO_KEY environment variable is not set.")
        
            cipher_suite = Fernet(crypto_key)
            decrypted_code = cipher_suite.decrypt(encrypted_code).decode()
            code = request.args.get("code")

            if code == decrypted_code:
                return jsonify(self.model.get_openai_answer(request.args.get("question")))
            else:
                return jsonify("UNAUTHORIZED")
        
    api.add_resource(_getHouses, "/houses")
    api.add_resource(_gethousedetails, "/housedetails")
    api.add_resource(_getOpenAIResponse, "/openai")