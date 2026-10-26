import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
import ad_tools



def size(adata):
    return (adata.n_obs, adata.n_vars)

def count_by(adata, column):
    return adata.obs[column].value_counts().to_dict()

def cells_where(adata, column, value):
    return adata[adata.obs[column] == value]

def ensembl_id(adata, symbol):
    gene = adata.var.loc[symbol, "gene_ids"]
    return gene

def gene_mean(adata, symbol):
    d = adata[:, symbol].X.toarray()
    return float(d.mean())

def embedding(adata, key):
    return adata.obsm[key]