from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, rdFingerprintGenerator
import pandas as pd

class MolecularProcessor:
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.df['Molecule'] = self.df['SMILES'].apply(self.smiles_to_mol)
        self.df = self.df[self.df['Molecule'].notnull()] # Remove invalid molecules
        self.visualize()
        self.calculate_descriptors()
        print(self.df)
    
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

    def calculate_descriptors(self):
        """Calculate molecular descriptors for each molecule."""
        # Basic Molecular Properties
        self.df['Molecular Weight'] = self.df['Molecule'].apply(Descriptors.MolWt) #Overall size - drug behavior, solubility, and permeability
        # self.df['NumAtoms']=self.df['Molecule'].apply(Descriptors.HeavyAtomCount) #Compexity
        
        # Structural Complexity
        self.df['NumRotatableBonds']=self.df['Molecule'].apply(Descriptors.NumRotatableBonds)  #Flexibility and binding
        # self.df['NumRings'] = self.df['Molecule'].apply(Descriptors.RingCount) #Rigidity
        # self.df['NumAromaticRings']=self.df['Molecule'].apply(Descriptors.NumAromaticRings) #Aromaticity
        # self.df['FractionCSP3'] = self.df['Molecule'].apply(Descriptors.FractionCSP3)
        # self.df['NumHeteroAtoms']=self.df['Molecule'].apply(Descriptors.NumHeteroatoms)
        
        # Polarity and Hydrogen Bonding
        self.df['NumHBA']=self.df['Molecule'].apply(Descriptors.NumHAcceptors) #Hydrogen Bonding Acceptor
        self.df['NumHBD']=self.df['Molecule'].apply(Descriptors.NumHDonors) #Hydrogen Bonding Donor
        self.df['LogP']=self.df['Molecule'].apply(Descriptors.MolLogP) #Lipophilicity - how molecule pass through membranes
        self.df['TPSA']=self.df['Molecule'].apply(Descriptors.TPSA) #Polar Surface Area - solubility, absorption and bioavailability
        
        # Molecular Fingerprints - substructures
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        self.df['MorganFingerprint'] = self.df['Molecule'].apply(lambda mol: generator.GetFingerprintAsNumPy(mol)) 
        self.df.to_csv('molecule_descriptors.csv', index=False)

processor = MolecularProcessor('data/diabetes_drugs.csv')
