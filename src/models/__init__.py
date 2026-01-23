from .lgb_model import LightGBMWrapper
from .prophet_model import ProphetWrapper
from .arima_model import ARIMAWrapper
from .torch_model import LSTMWrapper

REGISTRY = {
    "lightgbm": LightGBMWrapper,
    "prophet" : ProphetWrapper,
    "arima"   : ARIMAWrapper,
    "lstm"    : LSTMWrapper,
}

def make_model(kind: str, params: dict):
    if kind not in REGISTRY:
        raise ValueError(f"Unknown model kind: {kind}")
    return REGISTRY[kind](**params)
