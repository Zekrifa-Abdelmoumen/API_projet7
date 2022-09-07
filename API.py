from flask import Flask
from flask_restful import Api, Resource
import pandas as pd
import pickle as pc

app = Flask(__name__)
api = Api(app)

filename = 'finalized_model.sav'
loaded_model = pc.load(open(filename, 'rb'))
df = pd.read_csv('data_client.csv')
feats = [f for f in df.columns if f != "SK_ID_CURR"]


class Score(Resource):
    def get(self, id_curr):
        x = df[df['SK_ID_CURR'] == id_curr][feats]
        proba = loaded_model.predict_proba(x)[:, 1]

#        feat_imp = list(impact.to_dict()['not_risky'].keys())
        index = pd.Series([0])
        x = x.set_index(index)
        return {"data": {"SK_ID_CURR": id_curr,
                         "proba": float(proba),
                         "detail": x.T.to_dict()[0]}
                }


api.add_resource(Score, "/score/<int:id_curr>")

if __name__ == "__main__":
    app.run(debug=False)
