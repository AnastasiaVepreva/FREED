from functools import partial
from rdkit import Chem
import pickle
import sklearn
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import MACCSkeys
import torch
from rdkit.Chem import rdMolDescriptors
import xgboost as xgb
from rdkit.Contrib.SA_Score import sascorer
from rdkit.Chem import QED as qed_module
import os
from rdkit.Avalon import pyAvalonTools
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors
from rdkit.Chem.rdMolDescriptors import GetMACCSKeysFingerprint, GetMorganFingerprintAsBitVect
from typing import List
import warnings
from rdkit import RDLogger
import joblib
RDLogger.DisableLog('rdApp.*')
pd.set_option('display.float_format', '{:.2f}'.format)
warnings.filterwarnings('ignore')
from rdkit import Chem,DataStructs
from rdkit.Chem import Descriptors, QED, rdDepictor, AllChem, Draw
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

class Reward:
    def __init__(self, property, reward, weight=1.0, preprocess=None):
        self.property = property
        self.reward = reward
        self.weight = weight
        self.preprocess = preprocess

    def __call__(self, input):
        if self.preprocess:
            input = self.preprocess(input)
        property = self.property(input)
        reward = self.weight * self.reward(property)
        return reward, property


def identity(x):
    return x


def ReLU(x):
    return max(x, 0)


def HSF(x):
    return float(x > 0)


class OutOfRange:
    def __init__(self, lower=None, upper=None, hard=True):
        self.lower = lower
        self.upper = upper
        self.func = HSF if hard else ReLU

    def __call__(self, x):
        y, u, l, f = 0, self.upper, self.lower, self.func
        if u is not None:
            y += f(x - u)
        if l is not None:
            y += f(l - x)
        return y


class PatternFilter:
    def __init__(self, patterns):
        self.structures = list(filter(None, map(Chem.MolFromSmarts, patterns)))

    def __call__(self, molecule):
        return int(any(molecule.HasSubstructMatch(struct) for struct in self.structures))


def MolLogP(m):
    return rdMolDescriptors.CalcCrippenDescriptors(m)[0]

def SA(m):
    return sascorer.calculateScore(m)

def qed_mol(m): 
    return qed_module.qed(m)

def Brenk(m):
    params_brenk = FilterCatalogParams()
    params_brenk.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
    catalog_brenk = FilterCatalog(params_brenk)
    return 1*catalog_brenk.HasMatch(m)

def smi_to_morgan(smiles):
    mols = [Chem.MolFromSmiles(smile) for smile in smiles]
    fps = []
    for mol in mols:
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
            fps.append(np.array(fp))
        else:
            fps.append(np.zeros(1024, dtype=np.int32))

    fps_array = np.array(fps, dtype=np.int32).reshape(1, -1)

    return fps_array

def smi_to_morgan_512(smiles):
    mols = [Chem.MolFromSmiles(smile) for smile in smiles]
    fps = []
    for mol in mols:
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=512)
            fps.append(np.array(fp))
        else:
            fps.append(np.zeros(512, dtype=np.int32))

    fps_array = np.array(fps, dtype=np.int32).reshape(1, -1)

    return fps_array

def smi_to_MACCS(smiles):
    mols = [Chem.MolFromSmiles(smile) for smile in smiles]
    fps = []
    for i in range(len(mols)):
        fp = [int(bit) for bit in MACCSkeys.GenMACCSKeys(mols[i]).ToBitString()][:167]
        fps.append(fp)
    fps_array = np.array(fps, dtype=np.int32).reshape(1, -1)

    return fps_array

def alzh(mol):
  model = pickle.load(open('alzheimer_clf.pkl', 'rb'))
  fps = smi_to_MACCS([Chem.MolToSmiles(mol)])
  labels = model.predict_proba(fps)[0][0]
  return labels

def sklrz(mol):
  model = pickle.load(open('ic50_btk_clf_102.pkl', 'rb'))

  fps = smi_to_morgan([Chem.MolToSmiles(mol)])
  labels = model.predict_proba(fps)[0][1]
  return labels

def cancer(mol):
  model = pickle.load(open('lung_cancer_model.pkl', 'rb'))
  fps = smi_to_morgan_512([Chem.MolToSmiles(mol)])
  labels = model.predict_proba(fps)[0][1]
  return labels

#a function that brings the transmitted smiles to the canonical form
def safe_canon_smiles(smiles):
    try:
        return Chem.CanonSmiles(smiles)
    except Exception as e:
        print(f"Bad Smiles: {smiles}")
        return None

#a function that generates descriptors to describe the smiles molecule
def get_all_descriptors(smiles):
    mols = [Chem.MolFromSmiles(i) for i in smiles]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
    desc_names = calc.GetDescriptorNames()
    mol_descriptors = []
    for mol in mols:
        mol = Chem.AddHs(mol)
        descriptors = calc.CalcDescriptors(mol)
        mol_descriptors.append(descriptors)
    return mol_descriptors, desc_names

#a function that generates fingerprints to describe the structure of a molecule
def generate_AVfpts(data):
    Avalon_fpts = []
    mols = [Chem.MolFromSmiles(x) for x in data if x is not None]
    for mol in mols:
        avfpts = pyAvalonTools.GetAvalonFP(mol, nBits=512)
        Avalon_fpts.append(avfpts)
    return pd.DataFrame(np.array(Avalon_fpts))

#a function that creates a dataframe with all the features for the transmitted smiles
def create_features_for_smiles(smiles_names):
    smiles_data = [{"Smiles": smile} for smile in smiles_names]
    df = pd.DataFrame(smiles_data)
    df['Canonical Smiles'] = df.Smiles.apply(safe_canon_smiles)
    df.drop(['Smiles'], axis=1, inplace=True)
    mol_descriptors, descriptors_names = get_all_descriptors(df['Canonical Smiles'].tolist())
    descriptors_df = pd.DataFrame(mol_descriptors, columns=descriptors_names)
    AVfpts = generate_AVfpts(df['Canonical Smiles'])
    AVfpts.columns = AVfpts.columns.astype(str)
    df.drop(["Canonical Smiles"], axis=1, inplace=True)
    X_test = pd.concat([descriptors_df, AVfpts], axis=1)
    X_array = X_test.to_numpy().reshape(1, -1)
    return X_array

def prksn(mol):
    model = joblib.load('model_tyrosine_clf.pkl')
    predictions = model.predict_proba(create_features_for_smiles([Chem.MolToSmiles(mol)]))[0][0]
    return predictions

def resist(mol):
    model = joblib.load('model_drug_clf.pkl')
    predictions = model.predict_proba(create_features_for_smiles([Chem.MolToSmiles(mol)]))[0][0]
    return predictions

def dyslipid(mol):
    model = joblib.load('model_citrate_clf.pkl')
    predictions = model.predict_proba(create_features_for_smiles([Chem.MolToSmiles(mol)]))[0][0]
    return predictions



