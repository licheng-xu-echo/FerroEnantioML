import numpy as np
from tqdm import tqdm
import pandas as pd
from rdkit import RDLogger,Chem
from rdkit.Chem import rdMolDescriptors,Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from mordred import get_descriptors_from_module
from mordred import Calculator as mord_Calculator
from mordred import descriptors as mord_descriptors
from .calc import StericDescriptor
from scipy.stats import pearsonr
descs = [desc_name[0] for desc_name in Descriptors._descList]
desc_calc = MoleculeDescriptors.MolecularDescriptorCalculator(descs)
RDLogger.DisableLog('rdApp.*')
mordred_descs = get_descriptors_from_module(mord_descriptors, submodule=True)
mord_calc = mord_Calculator(mordred_descs, ignore_3D=True)

def get_morganfp(mols,radius=2,nBits=2048,useChirality=True,descriptor='MorganFingerprints',name=False):
    '''
    generate Morgan fingerprint
    Parameters
    ----------
    mols : [mol]
        list of RDKit mol object.
    Returns
    -------
    mf_desc : ndarray
        ndarray of molecular fingerprint descriptors.
    '''
    fps = [rdMolDescriptors.GetMorganFingerprintAsBitVect(mol,radius=radius,nBits=nBits,useChirality=useChirality,) for mol in mols]
    
    return np.array([list(map(eval,list(fp.ToBitString()))) for fp in fps])

def get_rdkit_desc(mols,descriptor='RDKit'):
    '''
    generate 2D molecular physicochemical descriptors using RDKit
    Parameters
    ----------
    mols : [mol]
        list of RDKit mol object.
    
    Returns
    ----------
    rdkit_desc : ndarray
        ndarray of molecular descriptors.
    
    '''
    return np.array([list(desc_calc.CalcDescriptors(mol)) for mol in mols])

def get_mordred_desc(mols,descriptor='Mordred'):
    '''
    generate 2D molecular physicochemical descriptors using mordred
    Parameters
    ----------
    mols : [mol]
        list of RDKit mol object.
        
    Returns
    -------
    mordred_desc : ndarray
        ndarray of molecular descriptors.
    '''
    desc = np.array([[value if isinstance(value,float) or isinstance(value,int) else np.nan for value in  list(mord_calc(mol))] for mol in tqdm(mols)])
    #df = calc.pandas(mols)
    return desc

def vect_columns_in_dataset(dataset,keys,desc_params,verbose=True,process_empty=False):
    '''
    vectorize specific columns in the dataset DataFrame
    Parameters
    ----------
    dataset : DataFrame
        dataset
    keys : [str]
        column names
    desc_params : dict
        descriptor parameters
    verbose : bool
        whether return redundant information
    '''
    smiles_arr = dataset[keys].to_numpy().T
    split_desc_arr = []
    desc_map_lst = []
    for _smiles_arr in smiles_arr:
        _desc_arr,_desc_map = map_smi_to_desc(_smiles_arr, desc_params, process_empty=process_empty)
        split_desc_arr.append(_desc_arr)
        desc_map_lst.append(_desc_map)
    merge_desc_arr = np.concatenate(split_desc_arr, axis=1)
    if verbose:
        return merge_desc_arr,split_desc_arr,desc_map_lst
    else:
        return merge_desc_arr

def map_smi_to_desc(smi_lst,desc_param,desc_map={},process_empty=False):
    '''
    map SMILES to descriptors
    Parameters
    ----------
    smi_lst : [SMILES]
        list of SMILES.
    desc_param : dict
        descriptor parameters.
    desc_map : dict
        SMILES descriptor map.
    '''
    if desc_map != {}:
        return np.array([desc_map[smi] for smi in smi_lst]),desc_map
    else:
        desc_map = get_smi_desc_map(smi_lst,desc_param,process_empty=process_empty)
        return np.array([desc_map[smi] for smi in smi_lst]),desc_map
    
def sel_high_corr_desc(desc,target,threshold=0.15,return_idx=False):
    assert desc.shape[0] == target.shape[0], "target and desc shape mismatch"
    desc_pearson_lst = [abs(pearsonr(desc[:,idx],target)[0]) for idx in range(desc.shape[1])]
    sel_desc_idx = []
    for idx in range(len(desc_pearson_lst)):
        if desc_pearson_lst[idx] > threshold:
            sel_desc_idx.append(idx)
    if len(sel_desc_idx) == 0:
        print("no descriptor has correlation greater than threshold")
        if return_idx:
            return desc,list(range(desc.shape[1]))
        else:
            return desc
    
    if return_idx:
        return desc[:,sel_desc_idx],sel_desc_idx
    else:
        return desc[:,sel_desc_idx]

def reduce_desc_with_corr_matrix(desc,threshold=0.9,ret_idx=False):
    corr_matrix = pd.DataFrame(desc).corr()
    
    to_keep = []

    considered = set()


    for i in range(corr_matrix.shape[0]):
        if i not in considered:

            highly_correlated = corr_matrix.index[(corr_matrix.iloc[:, i].abs() > threshold) & (corr_matrix.index != i)]
            considered.update(highly_correlated)
            considered.add(i)

            to_keep.append(i)
    reduce_desc = desc[:, to_keep]
    if not ret_idx:
        return reduce_desc
    else:
        return reduce_desc,to_keep
    
def get_smi_desc_map(smi_lst,desc_param,process_empty=False):
    '''
    generate SMILES descriptor map
    Parameters
    ----------
    smi_lst : [str]
        list of SMILES.
    desc_param : dict
        descriptor parameters.
        
    Returns
    -------
    smi_desc_map : dict
        SMILES descriptor map.
    '''
    smi_set = list(set(smi_lst))
    if process_empty:
        if "" in smi_set:
            smi_set.remove("")
    
    mol_set = [Chem.MolFromSmiles(smi) for smi in smi_set]
    set_size = len(smi_set)
    if desc_param['descriptor'] == 'MorganFingerprints':
        #desc_set = get_morganfp(mol_set,radius=desc_param['radius'],nBits=desc_param['n_bits'],useChirality=desc_param['use_chirality'])
        desc_set = get_morganfp(mol_set,**desc_param)
    elif desc_param['descriptor'] == 'RDKit':
        desc_set = get_rdkit_desc(mol_set)
    
    elif desc_param['descriptor'] == 'Mordred':
        desc_set = get_mordred_desc(mol_set)
        
    elif desc_param['descriptor'] == 'SPOC':
        rdkit_desc_set = get_rdkit_desc(mol_set)
        mf_desc_set = get_morganfp(mol_set,**desc_param)
        desc_set = np.concatenate([rdkit_desc_set,mf_desc_set],axis=1)
        
    elif desc_param['descriptor'] == 'OneHot':
        desc_set = []
        for i in range(set_size):
            tmp = np.zeros(set_size)
            tmp[i] = 1
            desc_set.append(tmp)
        desc_set = np.array(desc_set)
    else:
        raise ValueError(f"Descriptor {desc_param['descriptor']} not supported")
    if process_empty:
        try:
            desc_set = np.concatenate([np.zeros((1,len(desc_set[0]))),desc_set],axis=0)
        except:
            if desc_param['descriptor'] == 'MorganFingerprints':
                desc_set = np.concatenate([np.zeros((1,2048)),desc_set],axis=0) ## debug
            else:
                raise ValueError(f"Descriptor {desc_param['descriptor']} not supported")
        smi_set = [""] + smi_set
    if not "name" in desc_param:
        return {smi: desc for smi,desc in zip(smi_set,desc_set)}
    else:
        if not desc_param["name"]:
            return {smi: desc for smi,desc in zip(smi_set,desc_set)}
        else:
            desc_size = len(desc_set[0])
            if desc_param['descriptor'] == 'MorganFingerprints':
                name = [f"MF_{idx}" for idx in range(desc_size)]
            elif desc_param['descriptor'] == 'RDKit':
                name = descs
            elif desc_param['descriptor'] == 'Mordred':
                name = [desc.__name__ for desc in mordred_descs]
            elif desc_param['descriptor'] == 'SPOC':
                name = descs + [f"MF_{idx}" for idx in range(desc_size-len(descs))]
            elif desc_param['descriptor'] == 'OneHot':
                name = [f"OneHot_{idx}" for idx in range(desc_size)]
            else:
                raise ValueError(f"Descriptor {desc_param['descriptor']} not supported")
            #smi_set.append("name")
            #desc_set.append(name)
            smi_desc_map = {smi: desc for smi,desc in zip(smi_set,desc_set)}
            smi_desc_map["name"] = name
            return smi_desc_map    


def get_xtb_loc_glb_desc(xtb_glb_desc_map,xtb_loc_desc_map,
                         at_idx_lst,bo_idx_pair_lst):

    # Global descriptors
    xtb_glb_desc_values = np.array(list(xtb_glb_desc_map.values()))
    xtb_glb_desc_names = list(xtb_glb_desc_map.keys())

    # Local descriptors

    ## Electronic part
    xtb_loc_desc_values = []
    xtb_loc_desc_names_ = []
    desc_name_map = {'charges':['charges'],
                    'q_related':['convCN','q','C6AA','alpha'],
                    'fukui':['f+','f-','f0'],
                    'wbo':['wbo']}
    for key,values in xtb_loc_desc_map.items():
        #print(key,values)
        if key != 'wbo':
            local_xtb_ = np.array([values[idx] for idx in at_idx_lst])
            xtb_loc_desc_values.append(local_xtb_)
            xtb_loc_desc_names_ += desc_name_map[key]

    xtb_loc_desc_values = np.concatenate(xtb_loc_desc_values,axis=1)
    xtb_loc_desc_values = np.concatenate(xtb_loc_desc_values,axis=0)
    xtb_loc_desc_names = [[f'{name}_{idx+1}' for name in xtb_loc_desc_names_] for idx in at_idx_lst]
    xtb_loc_desc_names = np.concatenate(xtb_loc_desc_names,axis=0)

    xtb_bo_desc_values = np.array([xtb_loc_desc_map['wbo'][tuple(sorted(bond_idx))] if tuple(sorted(bond_idx)) in xtb_loc_desc_map['wbo'] else 0.1 for bond_idx in bo_idx_pair_lst])
    xtb_bo_desc_names = [f'wbo_{idx0+1}_{idx1+1}' for idx0,idx1 in bo_idx_pair_lst]
    
    xtb_desc_values = np.concatenate([xtb_glb_desc_values,xtb_loc_desc_values,xtb_bo_desc_values],axis=0)
    xtb_desc_names = xtb_glb_desc_names + list(xtb_loc_desc_names) + xtb_bo_desc_names
    return xtb_desc_values,xtb_desc_names

def get_steric_loc_desc(xyz_file,idx_for_bv,idx_for_sterimol):
    ## Steric part
    steric_desc = StericDescriptor(xyz_file)
    bv_desc_values = []
    b1_desc_values = []
    b5_desc_values = []
    l_desc_values = []
    steric_loc_desc_names = []
    idx_for_bv_cat = np.concatenate(idx_for_bv)
    for idx in idx_for_bv_cat:
        bv_desc_values += steric_desc.BV(idx)
        steric_loc_desc_names.append(f'bv_{idx+1}')
    for idx_pair in idx_for_sterimol:
        b1,b5,l = steric_desc.Sterimol(dummy_index=idx_pair[0],attached_index=idx_pair[1])
        b1_desc_values.append(b1)
        b5_desc_values.append(b5)
        l_desc_values.append(l)
        steric_loc_desc_names += [f'b1_{idx_pair[0]+1}_{idx_pair[1]+1}',f'b5_{idx_pair[0]+1}_{idx_pair[1]+1}',f'l_{idx_pair[0]+1}_{idx_pair[1]+1}']
    steric_loc_desc_values = np.concatenate([bv_desc_values,b1_desc_values,b5_desc_values,l_desc_values],axis=0)
    return steric_loc_desc_values,steric_loc_desc_names