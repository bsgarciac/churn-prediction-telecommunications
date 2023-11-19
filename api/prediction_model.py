from joblib import load
import pandas as pd
import custom_objects
import shap
import copy

class PredictionModel:

    def __init__(self, version):
        model_file = "churn-v1.0.pkl" if version == "1" else "churn-v2.0.pkl"
        self.model = load(f'models/{model_file}')
        self.transformer = copy.deepcopy(self.model)
        self.transformer.steps.pop()
        self.features_names = self.transformer["transformer"].get_feature_names_out()


    def make_predictions(self, X):
        data = pd.DataFrame([dict(x) for x in X])
        churns = self.model.predict(data)
        probabilities = self.model.predict_proba(data)
        result = []
        for x, churn, probability in zip(X, churns.tolist(), probabilities.tolist()):
            result.append({"customerID": x.customerID, "churn_predicted": churn, "probability_churn": probability[1]})
        return result
    

    def create_shap_df(self, data):
        df = pd.DataFrame(
            self.transformer.transform(data),
            columns=[f.split("__")[1] for f in self.features_names]
        )
        return df

    def explain_predictions(self, X):
        churn_historic = pd.read_json('https://raw.githubusercontent.com/AlberGonglius/Taller-3---Ciencia-de-datos-Aplicada---MINE/main/data/churn_historic.json')
        data = pd.DataFrame([dict(x) for x in X])
        predictions = self.make_predictions(X)

        def model(X):
            return self.model[-1].predict_proba(X)[:,1]

        explainer_dataset = self.create_shap_df(churn_historic[:100])
        X_dataset = self.create_shap_df(data)

        explainer = shap.Explainer(model, explainer_dataset)
        shap_values = explainer.shap_values(X_dataset)
        data = []
        for shap_value, prediction in zip(shap_values, predictions):
            shap_value_sorted = sorted(shap_value, key=abs, reverse=True)
            reasons = {}
            for important_feature in shap_value_sorted[:3]:
                index = list(shap_value).index(important_feature)
                reasons[self.features_names[index]] = important_feature
            prediction["top3_shap_values"] = reasons
            data.append(prediction)
        return data

        
    
    
