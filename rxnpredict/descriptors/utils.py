import numpy as np
from rdkit import Chem
def process_desc(array,return_idx=False):
    '''
    remove descriptors with zero range
    Parameters
    ----------
    array : ndarray
        Original numpy array.
    return_idx : bool, optional
        If True, return the index of descriptors with non-zero range. The default is False.
        
    Returns
    -------
    array : ndarray
        numpy array with descriptors with non-zero range.
        
    rig_idx : list
    '''
    array = np.array(array,dtype=np.float32)
    desc_len = array.shape[1]
    rig_idx = []
    for i in range(desc_len):
        try:
            desc_range = array[:,i].max() - array[:,i].min()
            if desc_range != 0 and not np.isnan(desc_range) and not np.isinf(desc_range):
                rig_idx.append(i)
        except:
            continue
    array = array[:,rig_idx]
    if return_idx == False:
        return array
    else:
        return array,rig_idx    


def maxminscale(array,return_scale=False):
    '''
    Max-min scaler
    Parameters
    ----------
    array : ndarray
        Original numpy array.
    Returns
    -------
    array : ndarray
        numpy array with max-min scaled.
    '''
    if not return_scale:
        return (array - array.min(axis=0))/(array.max(axis=0)-array.min(axis=0))
    else:
        return (array - array.min(axis=0))/(array.max(axis=0)-array.min(axis=0)),array.max(axis=0),array.min(axis=0)
    
def canonical_smiles(smiles: str):
    original_smi = smiles
    viewed_smi = {original_smi: 1}
    while original_smi != (
        canonical_smi := Chem.CanonSmiles(original_smi, useChiral=True)
    ) and (canonical_smi not in viewed_smi or viewed_smi[canonical_smi] < 2):
        original_smi = canonical_smi
        if original_smi not in viewed_smi:
            viewed_smi[original_smi] = 1
        else:
            viewed_smi[original_smi] += 1
    else:
        return original_smi
