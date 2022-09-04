from flask import Flask
from flask_restful import Api, Resource
from pandas import read_csv, DataFrame, Series
from pickle import load
from numpy import reshape
from shap import TreeExplainer

app = Flask(__name__)
api = Api(app)

filename = 'finalized_model.sav'
loaded_model = load(open(filename, 'rb'))
explainer = TreeExplainer(loaded_model)
df = read_csv('data_client.csv')
feats = [f for f in df.columns if f != "SK_ID_CURR"]


class Score(Resource):
    def get(self, id_curr):
        x = df[df['SK_ID_CURR'] == id_curr][feats]
        proba = loaded_model.predict_proba(x)[:, 1]
        exp_value = explainer.shap_values(x)[1]
        exp_value = reshape(exp_value, len(feats))
        exp_value = DataFrame(exp_value, index=feats, columns=['not_risky'])
        impact = exp_value
        impact['impact'] = impact['not_risky'].apply(lambda i: abs(i))
        impact = impact.sort_values(by=['impact'], ascending=False).head(10)[["not_risky"]]
        exp_value = exp_value.to_dict()['not_risky']

        feat_imp = list(impact.to_dict()['not_risky'].keys())
        index = Series([0])
        x = x[feat_imp].set_index(index)
        return {"data": {"SK_ID_CURR": id_curr,
                         "proba": float(proba),
                         "impact": exp_value,
                         "detail": x.T.to_dict()[0]}

                }


api.add_resource(Score, "/score/<int:id_curr>")

if __name__ == "__main__":
    app.run(debug=False)
