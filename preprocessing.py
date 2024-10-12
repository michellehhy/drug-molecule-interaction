from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
import datetime

from utilities import get_time

def smiles_to_mol(smiles):
      """Convert SMILES string to RDKit Mol object."""
      mol = Chem.MolFromSmiles(smiles)
      if mol is None:
          print(f"Warning: Invalid SMILES string: {smiles}")
      return mol

def calculate_descriptors(df):
        """Calculate molecular descriptors for each molecule."""
        # Basic Molecular Properties
        df['MolecularWeight'] = df['Molecule'].apply(Descriptors.MolWt)  # Overall size - drug behavior, solubility, and permeability
        # df['NumAtoms']=df['Molecule'].apply(Descriptors.HeavyAtomCount) #Compexity
        
        # Structural Complexity
        df['NumRotatableBonds']=df['Molecule'].apply(Descriptors.NumRotatableBonds)  # Flexibility and binding
        # df['NumRings'] = df['Molecule'].apply(Descriptors.RingCount) #Rigidity
        # df['NumAromaticRings']=df['Molecule'].apply(Descriptors.NumAromaticRings)  # Aromaticity
        # df['FractionCSP3'] = df['Molecule'].apply(Descriptors.FractionCSP3)
        # df['NumHeteroAtoms']=df['Molecule'].apply(Descriptors.NumHeteroatoms)
        
        # Polarity and Hydrogen Bonding
        df['NumHBA']=df['Molecule'].apply(Descriptors.NumHAcceptors)  # Hydrogen Bonding Acceptor
        df['NumHBD']=df['Molecule'].apply(Descriptors.NumHDonors)  # Hydrogen Bonding Donor
        df['LogP']=df['Molecule'].apply(Descriptors.MolLogP)  # Lipophilicity - how molecule pass through membranes
        df['TPSA']=df['Molecule'].apply(Descriptors.TPSA)  # Polar Surface Area - solubility, absorption and bioavailability
        
        # Molecular Fingerprints - substructures
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        df['MorganFingerprint'] = df['Molecule'].apply(lambda mol: generator.GetFingerprintAsNumPy(mol)) 

        df.to_csv(f'output/molecule_descriptors_{get_time()}.csv', index=False)
