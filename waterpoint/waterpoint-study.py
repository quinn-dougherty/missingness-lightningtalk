from dask.distributed import Client, progress
client = Client(n_workers=2, threads_per_worker=2, memory_limit='1GB')

import pandas as pd
import numpy as np
from scipy.stats import bernoulli 
from typing import List, Callable, Dict, Tuple, Iterable
from fancyimpute import SimpleFill, KNN
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer
from category_encoders import BinaryEncoder
from tqdm import tqdm
import dask.dataframe as dd
import pickle
from argparse import ArgumentParser

DataFrame = pd.core.frame.DataFrame
Series = pd.core.series.Series
Array = np.ndarray
Imputer = Callable[[DataFrame], DataFrame]
nan = np.nan

df = dd.read_csv('data.csv').drop(['name'], axis=1)
X_ = df.drop('status_group', axis=1)
y = df.status_group

be = BinaryEncoder()
FEATS = 20

pca = PCA(n_components = FEATS)

vals= pca.fit_transform(StandardScaler().fit_transform(be.fit_transform(X_.compute())))

X = pd.DataFrame(vals, columns=[f"pc{k+1}" for k in range(FEATS)], index=y.index).assign(y=y)

def mcar_goblin(dat: DataFrame, ratio: float) -> DataFrame: 
    ''' Simulate MCAR with bernoulli '''
    def ident_or_nan(x: float) -> float:
        ''' if heads, replace value with nan. if tails, identity '''
        coin = bernoulli(ratio)
        if coin.rvs()==1: 
            return nan
        else: 
            return x
    
    return dat.assign(**{feat: [ident_or_nan(x) 
                                for x in dat[feat].values] 
                         for feat in dat.columns if feat!='y'})

def coefs(imputer: Imputer, X: DataFrame, y: Series) -> Tuple[str, Array]: 
    ''' '''
    lm = LogisticRegression(multi_class='auto', solver='lbfgs', max_iter=10000)
    
    lm.fit(imputer(X), y)
    return (imputer.__name__, 
            np.array([sum(col**2)**0.5 for col in lm.coef_.T]) # column-norms, feature importances 
           )

def idem(dat: DataFrame) -> DataFrame: 
    '''no missin'''
    return dat
    
def fill_mean(dat: DataFrame) -> DataFrame: 
    return SimpleFill(fill_method='mean').fit_transform(dat)

def fill_noise(dat: DataFrame) -> DataFrame: 
    return SimpleFill(fill_method='random').fit_transform(dat)

def iterative(dat: DataFrame) -> DataFrame: 
    return IterativeImputer().fit_transform(dat)

def knn(dat: DataFrame) -> DataFrame: 
    return KNN(verbose=False).fit_transform(dat)

def drop(dat: DataFrame) -> DataFrame: 
    return dat.dropna(axis=0)

imputers: List[Imputer] = [fill_mean, fill_noise, iterative, knn]#, drop]

def experiment(imputers: List[Imputer], 
               X_: DataFrame, 
               y: Series, 
               ratio: float, 
               passes: int) -> Iterable[Tuple[Array, Array]]: 
    
    control = coefs(idem, X_, y)
    for k in tqdm(range(passes), desc=f"goblin {ratio*100}% | {passes} passes"): 
        X = mcar_goblin(X_, ratio)
        for imputer in imputers: 
            yield control, coefs(imputer, X, y) 
        
def result(xprmnt: Iterable[Tuple[Tuple[str, Array], 
                                  Tuple[str, Array]]]) -> DataFrame: 
    deltas = [(outcome[1][0], outcome[0][1] - outcome[1][1])
              for outcome in xprmnt]
    
    return (DataFrame([[delt[0]] + list(delt[1]) for delt in deltas])
            .rename(columns={**{0: 'imputer'}}))#, **{k: f"coef_norm_{k}" for k in range(1, FEATS+1)}}))

parser = ArgumentParser(description="run experiment")
parser.add_argument('--trials', metavar='N', type=int, default=4
                    help='how many times per imputer?')

args = parser.parse_args()

if __name__=='__main__': 
    
    TRIALS = args.trials
    result_df_40percent = result(experiment(imputers, X, y, 0.4, TRIALS))
    result_df_20percent = result(experiment(imputers, X, y, 0.2, TRIALS))

    with open("results_tuple.pickle", "rb") as serialize_results: 
        serialize_results.dump((result_df_40percent, result_df_20percent))

    result_df = DataFrame().assign(twenty = result_df_20percent.drop('imputer', axis=1).T.mean(), 
                                   fourty = result_df_40percent.drop('imputer', axis=1).T.mean())

    