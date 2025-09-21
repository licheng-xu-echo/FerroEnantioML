import numpy as np
def ee2ddG(ee,T,ret_abs=True):
    '''
    Transformation from ee to ΔΔG
    Parameters
    ----------
    ee : ndarray
        Enantiomeric excess.
    T : ndarray or float
        Temperature (K).

    Returns
    -------
    ddG : ndarray
        ΔΔG (kcal/mol).
    '''
    if ret_abs:
        ddG = -np.abs(8.314 * T * np.log((1-ee)/(1+ee)))  # J/mol
    else:
        ddG = 8.314 * T * np.log((1-ee)/(1+ee))  # J/mol
    ddG = ddG/1000/4.18            # kcal/mol
    return -ddG

def ddG2ee(ddG,T,ret_abs=True):
    '''
    Transformation from ΔΔG to ee. 
    Parameters
    ----------
    ddG : ndarray
        ΔΔG (kcal/mol).
    T : ndarray or float
        Temperature (K).

    Returns
    -------
    ee : ndarray
        Absolute value of enantiomeric excess.
    '''
    
    ddG = ddG*1000*4.18
    ee = (1-np.exp(ddG/(8.314*T)))/(1+np.exp(ddG/(8.314*T)))
    if ret_abs:
        return np.abs(ee)
    else:
        return -ee
