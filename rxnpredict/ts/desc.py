import numpy as np

def sel_satisfy_ts_data(dataset,ts_keys,smiles_tsdesc_map,verbose=False):
    ret_idx_lst = []
    for idx,pair in enumerate(dataset[ts_keys].to_numpy()):
        if ",".join(pair) in smiles_tsdesc_map:
            ret_idx_lst.append(idx)
        else:
            if verbose:
                print(f'{",".join(pair)} not in smiles_tsdesc_map')
            else:
                pass
    return ret_idx_lst

def vec_ts_pair_in_dataset(dataset,ts_keys,smiles_tsdesc_map):
    ts_desc = []
    for pair in dataset[ts_keys].to_numpy():
        ts_desc.append(smiles_tsdesc_map[",".join(pair)])
    return np.array(ts_desc)

def get_diff_ts_desc(desc_val,desc_name):
    desc_val_r = desc_val[:,:len(desc_name)//2]
    desc_val_s = desc_val[:,len(desc_name)//2:]
    desc_name_r = desc_name[:len(desc_name)//2]
    desc_name_s = desc_name[len(desc_name)//2:]
    desc_val_diff = np.abs(desc_val_r-desc_val_s)
    desc_name_diff = [f'{rname}_{sname}' for rname,sname in zip(desc_name_r,desc_name_s)]
    return desc_val_diff,desc_name_diff

def load_map(map_file):
    inf_map = {}
    with open(map_file,'r') as f:
        lines = f.readlines()
    for line in lines[1:]:
        smiles = line.strip().split("'")[1]
        index = line.strip().split("'")[2].split(',')[1]
        inf_map[smiles] = int(index)
    return inf_map