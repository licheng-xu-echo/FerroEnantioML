from rdkit import Chem
import pandas as pd
from rdkit.Chem.rdmolops import ReplaceCore

def frag2rad(smiles):
    rwmol = Chem.RWMol(Chem.MolFromSmiles(smiles))
    rwmol.RemoveAtom(0)
    rwmol.GetAtomWithIdx(0).SetNumRadicalElectrons(
        rwmol.GetAtomWithIdx(0).GetNumRadicalElectrons() + 1
    )
    return Chem.MolToSmiles(rwmol.GetMol())

def get_dedup_index_map(original_df, dedup_df):
    index_map = {}
    kept_indices = dedup_df.index
    for index in kept_indices:
        matching_indices = original_df[(original_df.loc[index] == original_df).all(axis=1)].index
        index_map[index] = list(matching_indices)
    return index_map

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
    
def mod_title_in_gjf(gjf_file, new_title):
    with open(gjf_file, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if "#" in line:
            title_line = i + 2
            lines[title_line] = f'{new_title}\n'
            break
    with open(gjf_file, 'w') as f:
        f.writelines(lines)

def write_title(freeze_site,loc_site):
    title = "Freeze index"
    for idx_ in freeze_site:
        title += f"  {idx_}"
    title += " // Local index"
    for idx_ in loc_site:
        title += f"  {idx_}"
    return title

def get_amino_type_and_substituent_lst(lig_smi_lst, amino_structure, l_amino_substructure, d_amino_substructure):
    amino_type_lst = []
    for smi in lig_smi_lst:
        mol = Chem.MolFromSmiles(smi)
        if mol.HasSubstructMatch(amino_structure) and len(mol.GetAtomWithIdx(mol.GetSubstructMatch(amino_structure)[2]).GetNeighbors()) < 3:
            if mol.HasSubstructMatch(l_amino_substructure, useChirality=True):
                amino_type_lst.append("L")
            elif mol.HasSubstructMatch(d_amino_substructure, useChirality=True):
                amino_type_lst.append("D")
            else:
                amino_type_lst.append("NA")
        else:
            amino_type_lst.append("NA")
    amino_substituents = [Chem.MolToSmiles(ReplaceCore(Chem.MolFromSmiles(smi),amino_structure)) if type_ != "NA" else None for smi,type_ in zip(lig_smi_lst, amino_type_lst)]
    amino_substituents_rad = [[frag2rad(sub_smi) for sub_smi in smi.split('.') if sub_smi != ""] if smi != None else None for smi in amino_substituents]
    amino_sub_num_lst = [len(item) for item in amino_substituents_rad if item != None]
    max_amino_sub_num = max(amino_sub_num_lst) if amino_sub_num_lst else 0
    amino_substituents_rad_align = []
    for item in amino_substituents_rad:
        if item == None:
            amino_substituents_rad_align.append(None)
        else:
            amino_substituents_rad_align.append(item + ["[H]" for _ in range(max_amino_sub_num - len(item))])
    return amino_type_lst, amino_substituents_rad_align, max_amino_sub_num

def get_ferr_type_and_substituent_lst(rct1_smi_lst, core_fe_1, core_fe_2=None):
    fe_type_lst = []
    fe_substituents = []
    for smi in rct1_smi_lst:
        mol = Chem.MolFromSmiles(smi)

        if smi == "CC(N(C)C)C12C3C4C5C1[Fe]45321678C2C1C6C7C28":
            fe_type_lst.append(3)
            #print(123)
            for match in mol.GetSubstructMatches(core_fe_1):
                match_smi = Chem.MolToSmiles(ReplaceCore(Chem.MolFromSmiles(smi),core_fe_1,match))
                #print(match_smi)
                if "N" in match_smi.split(".")[0]:
                    fe_substituents.append(match_smi)
                    break
                else:
                    match_smi_blks = match_smi.split(".")
                    fe_substituents.append(f"{match_smi_blks[1]}.{match_smi_blks[0]}")
                    continue
            #print(f"type_3",len(fe_type_lst),len(fe_substituents))

        elif mol.HasSubstructMatch(core_fe_1):
            fe_type_lst.append(1)
            for match in mol.GetSubstructMatches(core_fe_1):
                match_smi = Chem.MolToSmiles(ReplaceCore(Chem.MolFromSmiles(smi),core_fe_1,match))
                if "N" in match_smi.split(".")[0]:
                    fe_substituents.append(match_smi)
                    break
                else:
                    continue
            #print(f"type_1",len(fe_type_lst),len(fe_substituents))
        elif core_fe_2 != None and mol.HasSubstructMatch(core_fe_2):
            fe_type_lst.append(2)
            fe_substituents.append(Chem.MolToSmiles(ReplaceCore(Chem.MolFromSmiles(smi),core_fe_2)))
            #print(f"type_2",len(fe_type_lst),len(fe_substituents))
        else:
            fe_type_lst.append("NA")
            fe_substituents.append(None)

    fe_substituents_rad = [[frag2rad(sub_smi) for sub_smi in smi.split('.') if sub_smi != ""] if smi != None else None for smi in fe_substituents]
    max_ferr_sub_num = max([len(item) for item in fe_substituents_rad if item != None])
    fe_substituents_rad_align = []
    for item in fe_substituents_rad:
        if item == None:
            fe_substituents_rad_align.append(None)
        else:
            fe_substituents_rad_align.append(item + ["[H]" for _ in range(max_ferr_sub_num - len(item))])
    return fe_type_lst, fe_substituents_rad_align, max_ferr_sub_num

def dedup_and_ret_sub_table(rct1_smi_lst,lig_smi_lst,amino_type_lst,fe_type_lst,fe_substituents_rad_align,amino_substituents_rad_align,max_ferr_sub_num,max_amino_sub_num):
    sub_info_map = {"amino_type":[],"fe_type":[]}
    ori_smi_map = {"rct1":[],"lig":[]}
    for _ in range(max_ferr_sub_num):
        sub_info_map[f"fe_sub_{_}"] = []
    for _ in range(max_amino_sub_num):
        sub_info_map[f"amino_sub_{_}"] = []
    for amino_type_, fe_type_, fe_sub, amino_sub,rct1,lig in zip(amino_type_lst,fe_type_lst,fe_substituents_rad_align,amino_substituents_rad_align,rct1_smi_lst,lig_smi_lst):
        if amino_type_ == "NA" or fe_type_ == "NA" or fe_sub == None or amino_sub == None:
            continue
        sub_info_map["amino_type"].append(amino_type_)
        sub_info_map["fe_type"].append(fe_type_)
        for idx,sub in enumerate(fe_sub):
            sub_info_map[f"fe_sub_{idx}"].append(sub)
        for idx,sub in enumerate(amino_sub):
            sub_info_map[f"amino_sub_{idx}"].append(sub)
        ori_smi_map["rct1"].append(rct1)
        ori_smi_map["lig"].append(lig)
    sub_info_df = pd.DataFrame(sub_info_map)
    ori_smi_df = pd.DataFrame(ori_smi_map)
    sub_info_df_dep = sub_info_df.drop_duplicates()
    dep_index_map = get_dedup_index_map(sub_info_df, sub_info_df_dep)
    smi_idx_map = {}
    for key in dep_index_map:
        smi_lst = [",".join(list(ori_smi_df.iloc[idx])) for idx in dep_index_map[key]]
        for smi in smi_lst:
            smi_idx_map[smi] = key
    return sub_info_df_dep, smi_idx_map

def write_map(inf_map,file,title_lst=["smiles_pair","index"]):
    with open(file,"w") as fw:
        fw.write(",".join(title_lst)+"\n")
        for key,value in inf_map.items():
            fw.write(f"'{key}',{value}\n")

def load_map(map_file):
    inf_map = {}
    with open(map_file,'r') as f:
        lines = f.readlines()
    #print(len(lines)-1)
    for line in lines[1:]:
        smiles = line.strip().split("'")[1]
        index = line.strip().split("'")[2].split(',')[1]
        inf_map[smiles] = int(index)
    return inf_map

def read_rct_and_lig(data_df,rct_col_name="Reactant2 Smiles",lig_col_name="Catalyst2 Smiles"):
    lig_smi_lst,rct1_smi_lst = data_df[lig_col_name].to_list(),data_df[rct_col_name].to_list()
    lig_smi_lst = [canonical_smiles(smi) for smi in lig_smi_lst]
    rct1_smi_lst = [canonical_smiles(smi) for smi in rct1_smi_lst]
    lig_rct1_pair_set = set(zip(lig_smi_lst,rct1_smi_lst))
    lig_smi_lst = [item[0] for item in lig_rct1_pair_set]
    rct1_smi_lst = [item[1] for item in lig_rct1_pair_set]
    return rct1_smi_lst,lig_smi_lst
