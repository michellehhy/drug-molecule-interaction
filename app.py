from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, rdFingerprintGenerator
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import joblib
import pandas as pd
import numpy as np
import json

class MolecularProcessor:
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.df['Molecule'] = self.df['SMILES'].apply(self.smiles_to_mol)
        self.df = self.df[self.df['Molecule'].notnull()]  # Remove invalid molecules
        self.visualize()
        self.calculate_descriptors()
        self.compute_similarity()
        self.predict_random_forest()
    
    def smiles_to_mol(self, smiles):
        """Convert SMILES string to RDKit Mol object."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Warning: Invalid SMILES string: {smiles}")
        return mol

    
    def visualize(self):
        """Visualize and save the molecules with labels."""
        mols = self.df['Molecule'].tolist()
        labels = self.df['DrugName'].tolist()
        
        img = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(200, 200), legends=labels)
        img.show()
        img.save('molecules_with_labels.png')


    def calculate_descriptors(self):
        """Calculate molecular descriptors for each molecule."""
        # Basic Molecular Properties
        self.df['MolecularWeight'] = self.df['Molecule'].apply(Descriptors.MolWt)  # Overall size - drug behavior, solubility, and permeability
        # self.df['NumAtoms']=self.df['Molecule'].apply(Descriptors.HeavyAtomCount) #Compexity
        
        # Structural Complexity
        self.df['NumRotatableBonds']=self.df['Molecule'].apply(Descriptors.NumRotatableBonds)  # Flexibility and binding
        # self.df['NumRings'] = self.df['Molecule'].apply(Descriptors.RingCount) #Rigidity
        # self.df['NumAromaticRings']=self.df['Molecule'].apply(Descriptors.NumAromaticRings)  # Aromaticity
        # self.df['FractionCSP3'] = self.df['Molecule'].apply(Descriptors.FractionCSP3)
        # self.df['NumHeteroAtoms']=self.df['Molecule'].apply(Descriptors.NumHeteroatoms)
        
        # Polarity and Hydrogen Bonding
        self.df['NumHBA']=self.df['Molecule'].apply(Descriptors.NumHAcceptors)  # Hydrogen Bonding Acceptor
        self.df['NumHBD']=self.df['Molecule'].apply(Descriptors.NumHDonors)  # Hydrogen Bonding Donor
        self.df['LogP']=self.df['Molecule'].apply(Descriptors.MolLogP)  # Lipophilicity - how molecule pass through membranes
        self.df['TPSA']=self.df['Molecule'].apply(Descriptors.TPSA)  # Polar Surface Area - solubility, absorption and bioavailability
        
        # Molecular Fingerprints - substructures
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        self.df['MorganFingerprint'] = self.df['Molecule'].apply(lambda mol: generator.GetFingerprintAsNumPy(mol)) 
        self.df.to_csv('molecule_descriptors.csv', index=False)

    
    def compute_similarity(self):
        """Compute similarity between the input molecule and the molecules in the dataset."""
        def tanimoto_similarity(fp1, fp2):
            intersection = np.dot(fp1, fp2)
            union = np.sum(fp1) + np.sum(fp2) - intersection
            return intersection / union
            
        drugs = self.df['DrugName'].tolist()
        similarity_matrix = []

        for i in range(len(self.df)):
            similarities = []
            for j in range(len(self.df)):
                sim = tanimoto_similarity(self.df['MorganFingerprint'].iloc[i], self.df['MorganFingerprint'].iloc[j])
                similarities.append(sim)
            similarity_matrix.append(similarities)

        similarity_df = pd.DataFrame(similarity_matrix, index=drugs, columns=drugs)
        print(similarity_df)

    
    def predict_random_forest(self):
        X = self.df[['MolecularWeight', 'NumRotatableBonds', 'NumHBA', 'NumHBD', 'TPSA']]  # Feature set
        y = self.df['LogP']  # Target variable

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [None, 10, 20, 30, 50],
        'min_samples_split': [2, 5, 10],
        }

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=10, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)
        random_search.fit(X_train, y_train)
        print(f"Best parameters: {random_search.best_params_}")
        print(f"Best Mean Squared Error: {-random_search.best_score_}")
        
        best_model = random_search.best_estimator_
        joblib.dump(best_model, 'best_random_forest_model.pkl')
        with open('best_hyperparametres.json', 'w') as f:
            json.dump(random_search.best_params_, f)

        y_pred = best_model .predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='neg_mean_squared_error')
        cv_mse = -cv_scores.mean()

        print(f"Test Set - Mean Squared Error: {mse}")
        print(f"Test Set - R-squared: {r2}")
        print(f"Mean Absolute Error: {mae}")
        print(f"Root Mean Squared Error: {rmse}")
        print(f"Cross-Validated MSE: {cv_mse}")

        test_drugs = self.df.loc[X_test.index, 'DrugName']
        for i in range(len(y_test)):
            print(f"{test_drugs.iloc[i]} - Actual LogP: {y_test.iloc[i]}, Predicted LogP: {y_pred[i]}")
        
        plt.scatter(y_test, y_pred)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
        plt.xlabel('Actual LogP')
        plt.ylabel('Predicted LogP')
        plt.title('Grid Search - Actual vs Predicted LogP')
        plt.show()

        # Feature Importance   
        importances = best_model.feature_importances_
        features = X.columns
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances")
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.gca().invert_yaxis()
        plt.show()


if __name__ == '__main__':
    processor = MolecularProcessor('data/diabetes_drugs.csv')
