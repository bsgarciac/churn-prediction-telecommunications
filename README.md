# Churn Prediction API using Classification Models

## Overview
This API employs various classification models to predict customer churn for a telecommunications company. It exposes endpoints to make predictions, and retrieve model explanations. You can also check out the [Api Rest Usage Video](https://www.youtube.com/watch?v=ijLoX-LJflY&feature=youtu.be) to get a better view of it.

# Contributors
* Juan David Ayala Nariño
* Brayan Steven Garcia Cardenas
* Alberto Jose Mendoza Peñaloza
* Carlos Fernando Montaña Herrera
  

## Notebooks Usage

* Clone this repository to your local machine.
* Open and run the Jupyter Notebook **analysis.ipynb** to explore the EDA of the churn data.
* Open and run the Jupyter Notebook **train.ipynb** to create the Pipelines and train the baseline model and the best model.
* Open and run the Jupyter Notebook **ab_testing.ipynb** to test the API Rest, comparing both models. 

## API Rest Setup
1. **Installation:** Ensure you have Python 3.x installed. Clone the repository and install dependencies using:
    ```bash
    pip install -r requirements.txt
    ```

2. **Dataset Preparation:** Run server.
    ```bash
    uvicorn main:app --reload
    ```

## Endpoints
### 1. `/{model_version}/predict`
- **Method:** POST
- **Parameters:**
  - `data`: JSON payload containing new data for prediction.
  - `model_version`: Model version used for prediction. (1 or 2)
- **Returns:** Predictions for the input data.

### 2. `/{model_version}/explain`
- **Method:** POST
- **Parameters:** 
  - `data`: JSON payload containing new data for prediction.
  - `model_version`: Model version used for prediction. (1 or 2)
- **Returns:** Predictions for the input data and them explanations (top 3 shap values).


## Notes
- Experiment with the different models versions to achieve better prediction accuracy.