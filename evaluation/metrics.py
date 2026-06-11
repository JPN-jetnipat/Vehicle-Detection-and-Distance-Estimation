
import numpy as np
def mae(y_true,y_pred): return np.mean(np.abs(np.array(y_true)-np.array(y_pred)))
def rmse(y_true,y_pred): return np.sqrt(np.mean((np.array(y_true)-np.array(y_pred))**2))
