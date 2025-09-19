import numpy as np

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return float((np.abs((y_true[mask]-y_pred[mask]) / y_true[mask]).mean()) * 100)

def rmse(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred)**2)))
