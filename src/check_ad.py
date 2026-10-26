import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
import ad_tools

results = []


def report(label, ok, detail=""):
    print(("PASS     " if ok else "FAIL     ") + label + (f"  ({detail})" if detail and not ok else ""))
    results.append(ok)


def run(label, func_name, *args):
    func = getattr(ad_tools, func_name, None)
    if func is None:
        print(f"MISSING  {func_name}() is not defined in ad_tools.py")
        results.append(False)
        return None
    try:
        return func(*args)
    except Exception as e:
        report(label, False, f"{type(e).__name__}: {e}")
        return None


# A made-up mini brain dataset: 4 cells x 3 genes
X = np.array([[0.0, 5.0, 1.0], [7.0, 0.0, 0.0], [1.0, 6.0, 0.0], [0.0, 0.0, 4.0]], dtype=np.float32)
obs = pd.DataFrame({"cell_type": ["neuron", "astrocyte", "neuron", "microglia"]},
                   index=["cell_a", "cell_b", "cell_c", "cell_d"])
var = pd.DataFrame({"gene_ids": ["ENSG00000131095", "ENSG00000132639", "ENSG00000204472"]},
                   index=["GFAP", "SNAP25", "AIF1"])
mini = ad.AnnData(X=X, obs=obs, var=var)
mini.obsm["X_fake"] = np.arange(8, dtype=float).reshape(4, 2)
mini_sparse = mini.copy()
mini_sparse.X = sparse.csr_matrix(X)

got = run("1 size", "size", mini)
if got is not None:
    report("1 size is (cells, genes) = (4, 3)", tuple(got) == (4, 3), f"got {got}")

got = run("2 count_by", "count_by", mini, "cell_type")
if got is not None:
    report("2 count_by gives a dict of counts", dict(got) == {"neuron": 2, "astrocyte": 1, "microglia": 1}, f"got {got}")

got = run("3 cells_where", "cells_where", mini, "cell_type", "neuron")
if got is not None:
    ok = got.shape == (2, 3) and list(got.obs_names) == ["cell_a", "cell_c"]
    report("3 cells_where keeps only the neurons, all genes", ok, f"got shape {getattr(got, 'shape', None)}")

got = run("4 ensembl_id", "ensembl_id", mini, "SNAP25")
if got is not None:
    report("4 ensembl_id('SNAP25') is ENSG00000132639", str(got) == "ENSG00000132639", f"got {got!r}")

got = run("5 gene_mean", "gene_mean", mini, "SNAP25")
if got is not None:
    report("5 gene_mean('SNAP25') is 2.75", np.isclose(got, 2.75), f"got {got}")
got = run("5 gene_mean sparse", "gene_mean", mini_sparse, "AIF1")
if got is not None:
    report("5 gene_mean also works when X is sparse", np.isclose(got, 1.25), f"got {got}")

got = run("6 embedding", "embedding", mini, "X_fake")
if got is not None:
    report("6 embedding returns the (4, 2) array from obsm", np.shape(got) == (4, 2) and np.allclose(got, mini.obsm["X_fake"]), f"got shape {np.shape(got)}")

print(f"\n{sum(results)} of {len(results)} checks passed")