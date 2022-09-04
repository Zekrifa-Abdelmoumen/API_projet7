from flask import Flask
from flask_restful import Api, Resource

app = Flask(__name__)
api = Api(app)


class Hello_world(Resource):
    def get(self, n):
        a = n*n
        return {"number": a
                }


api.add_resource(Hello_world, "/hello_world/<int:n>")

if __name__ == "__main__":
    app.run(debug=True)
