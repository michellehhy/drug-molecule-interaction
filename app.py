import pandas as pd

from preprocessing import calculate_descriptors, smiles_to_mol
from random_forest import compute_similarity, predict_random_forest
from utilities import visualize


class MolecularProcessor:
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.df['Molecule'] = self.df['SMILES'].apply(smiles_to_mol)
        self.df = self.df[self.df['Molecule'].notnull()]  # Remove invalid molecules
        # visualize(self.df)
        calculate_descriptors(self.df)
        compute_similarity(self.df)
        predict_random_forest(self.df)
    
if __name__ == '__main__':
    processor = MolecularProcessor('data/diabetes_drugs.csv')
