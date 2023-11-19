from typing import List
import pandas as pd

from fastapi import FastAPI

from data_model import DataModel
from prediction_model import PredictionModel


app = FastAPI()


@app.post("/{model_version}/predict")
def make_predictions(model_version, data: List[DataModel]):
    predicion_model = PredictionModel(model_version)
    results = predicion_model.make_predictions(data)
    return results

@app.post("/{model_version}/explain")
def make_predictions(model_version, data: List[DataModel]):
    predicion_model = PredictionModel(model_version)
    results = predicion_model.explain_predictions(data)
    return results