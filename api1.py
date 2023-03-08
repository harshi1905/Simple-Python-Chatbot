from flask import Flask
from flask_restful import Resource, Api, reqparse

app = Flask(__name__)
api = Api(app)

CHAT={ 
    'message': 'hi'
}

class chatbot(Resource):
    def get(self):
        
        parser = reqparse.RequestParser()
        parser.add_argument('userQuestion', type=str)
        
        args = parser.parse_args()
        return CHAT if args.userQuestion == "hello" else "wrong message"

api.add_resource(chatbot, '/bar', endpoint='bar')

if __name__ == '__main__':
    app.run(debug=True)
    

    