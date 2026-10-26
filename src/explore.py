import ad_tools
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
import ad_tools
import scanpy as sc

pb = sc.datasets.pbmc3k()
pb_processed = sc.datasets.pbmc3k_processed()


print(f"size : {ad_tools.size(pb)}")
print(f"gene_id : {ad_tools.ensembl_id(pb, "CD3E")}")
print(f"mean : {round(ad_tools.gene_mean(pb, "CD3E"), 3)}")
print(f"num of cells: {ad_tools.count_by(pb_processed, "louvain")}")
print(ad_tools.size(ad_tools.cells_where(pb_processed, "louvain", "B cells")))
print(f"{ad_tools.embedding(pb_processed, "X_pca").shape}")

print(pb.var.head(), pb_processed.obs.head())