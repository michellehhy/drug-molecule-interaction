from rdkit.Chem import Draw
import datetime

def visualize(df):
        """Visualize and save the molecules with labels."""
        mols = df['Molecule'].tolist()
        labels = df['DrugName'].tolist()
        
        img = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(200, 200), legends=labels)
        img.show()
        img.save('output/molecules_with_labels.png')

def get_time():
      now = datetime.datetime.now()
      return now.strftime("%Y-%m-%d_%H-%M-%S")