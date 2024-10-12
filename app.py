from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import pandas as pd

class MoleculeVisualizer:
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.df['Molecule'] = self.df['SMILES'].apply(self.smiles_to_mol)
        self.df = self.df[self.df['Molecule'].notnull()] # Remove invalid molecules
    
    def smiles_to_mol(self, smiles):
        """Convert SMILES string to RDKit Mol object."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Warning: Invalid SMILES string: {smiles}")
        return mol

    def visualize(self):
        """Visualize and save the molecules with labels."""
        mols = self.df['Molecule'].tolist()
        labels = self.df['Drug Name'].tolist()
        
        img = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(200, 200), legends=labels)
        img.show()
        img.save('molecules_with_labels.png')

visualizer = MoleculeVisualizer('data/diabetes_drugs.csv')
visualizer.visualize()
