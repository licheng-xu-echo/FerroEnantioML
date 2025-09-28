from rdkit import Chem
def clear_ferr_chirality(smi):
    mol = Chem.MolFromSmiles(smi)
    atoms = mol.GetAtoms()
    Fe_atoms = [atom for atom in atoms if atom.GetSymbol() == 'Fe' or atom.GetSymbol() == 'Ru']
    for Fe_at in Fe_atoms:
        nei_atoms = Fe_at.GetNeighbors()
        [atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_UNSPECIFIED) for atom in nei_atoms]
    return Chem.MolToSmiles(mol)
