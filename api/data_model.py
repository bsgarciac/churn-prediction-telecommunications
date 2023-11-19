from pydantic import BaseModel
from typing import Union

class DataModel(BaseModel):
    customerID: str
    gender:str
    SeniorCitizen:int
    Partner:str
    Dependents: str
    tenure: int
    PhoneService: str 
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str 
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: Union[str, float]
