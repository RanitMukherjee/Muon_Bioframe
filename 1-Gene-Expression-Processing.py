#!/usr/bin/env python
# coding: utf-8

# # Processing gene expression of 10k PBMCs

# This is the first chapter of the multimodal single-cell gene expression and chromatin accessibility analysis. In this notebook, scRNA-seq data processing is described, largely following [this scanpy notebook](https://scanpy-tutorials.readthedocs.io/en/latest/pbmc3k.html) on processing and clustering PBMCs.

# In[1]:


# Change directory to the root folder of the repository
import os
os.chdir("../../")


# ## Download data

# Download the data that we will use for this series of notebooks. [The data is available here](https://support.10xgenomics.com/single-cell-multiome-atac-gex/datasets/1.0.0/pbmc_granulocyte_sorted_10k).
# 
# For the tutorial, we will use the following files:
# 
# 1. Filtered feature barcode matrix (HDF5)
# 1. ATAC peak annotations based on proximal genes (TSV)
# 1. ATAC Per fragment information file (TSV.GZ)
# 1. ATAC Per fragment information index (TSV.GZ index)

# In[2]:


# This is the directory where those files are downloaded to
data_dir = "data/pbmc10k"


# In[3]:


# Remove file prefixes if any
prefix = "pbmc_granulocyte_sorted_10k_"
for file in os.listdir(data_dir):
    if file.startswith(prefix):
        new_filename = file[len(prefix):]
        os.rename(os.path.join(data_dir, file), os.path.join(data_dir, new_filename))


# ## Load libraries and data

# Import libraries:

# In[4]:


import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad


# In[5]:


import muon as mu


# We will use an HDF5 file containing gene and peak counts as input. In addition to that, when loading this data, `muon` will look for default files like `atac_peak_annotation.tsv` and `atac_fragments.tsv.gz` in the same folder and will load peak annotation table and remember the path to the fragments file if they exist.

# In[6]:


mdata = mu.read_10x_h5(os.path.join(data_dir, "filtered_feature_bc_matrix.h5"))
mdata.var_names_make_unique()
mdata


# Muon uses multimodal data (MuData) objects as containers for multimodal data.
# 
# `mdata` here is a MuData object that has been created directly from an AnnData object with multiple features types.

# ## RNA

# In this notebook, we will only work with the Gene Expression modality.
# 
# We can refer to an individual AnnData inside the MuData by defining a respective variable. All the operations will be performed on the respective AnnData object inside the MuData as you would expect.
# 
# > Please note that when AnnData is copied (e.g. `rna = rna.copy()`), there is no way for `mdata` to have a reference to this new object. It should be assigned back to a respective modality then (`mdata.mod['rna'] = rna`).

# In[7]:


rna = mdata.mod['rna']
rna


# ### Preprocessing

# #### QC

# Perform some quality control. For now, we will filter out cells that do not pass QC.

# In[8]:


rna.var['mt'] = rna.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
sc.pp.calculate_qc_metrics(rna, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)


# In[9]:


sc.pl.violin(rna, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
             jitter=0.4, multi_panel=True)


# Filter genes which expression is not detected:

# In[10]:


mu.pp.filter_var(rna, 'n_cells_by_counts', lambda x: x >= 3)
# This is analogous to
#   sc.pp.filter_genes(rna, min_cells=3)
# but does in-place filtering and avoids copying the object


# Filter cells:

# In[11]:


mu.pp.filter_obs(rna, 'n_genes_by_counts', lambda x: (x >= 200) & (x < 5000))
# This is analogous to 
#   sc.pp.filter_cells(rna, min_genes=200)
#   rna = rna[rna.obs.n_genes_by_counts < 5000, :]
# but does in-place filtering avoiding copying the object

mu.pp.filter_obs(rna, 'total_counts', lambda x: x < 15000)
mu.pp.filter_obs(rna, 'pct_counts_mt', lambda x: x < 20)


# Let's see how the data looks after filtering:

# In[12]:


sc.pl.violin(rna, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
             jitter=0.4, multi_panel=True)


# #### Normalisation

# We'll normalise the data so that we get log-normalised counts to work with.

# In[13]:


sc.pp.normalize_total(rna, target_sum=1e4)


# In[14]:


sc.pp.log1p(rna)


# #### Feature selection

# We will label highly variable genes that we'll use for downstream analysis.

# In[15]:


sc.pp.highly_variable_genes(rna, min_mean=0.02, max_mean=4, min_disp=0.5)


# In[16]:


sc.pl.highly_variable_genes(rna)


# In[17]:


np.sum(rna.var.highly_variable)


# #### Scaling

# We'll save log-normalised counts in a `.raw` slot:

# In[18]:


rna.raw = rna


# ... and scale the log-normalised counts to zero mean and unit variance:

# In[19]:


sc.pp.scale(rna, max_value=10)


# ### Analysis

# Having filtered low-quality cells, normalised the counts matrix, and performed feature selection, we can already use this data for multimodal integration.
# 
# However it is usually a good idea to study individual modalities as well.
# Below we run PCA on the scaled matrix, compute cell neighbourhood graph, and perform clustering to define cell types.

# #### PCA and neighbourhood graph

# In[20]:


sc.tl.pca(rna, svd_solver='arpack')


# To visualise the result, we will use some markers for (large-scale) cell populations we expect to see such as T cells and NK cells (CD2), B cells (CD79A), and KLF4 (monocytes).

# In[21]:


sc.pl.pca(rna, color=['CD2', 'CD79A', 'KLF4', 'IRF8'])


# The first principal component (PC1) is separating myeloid (monocytes) and lymphoid (T, B, NK) cells while B cells-related features seem to drive the second one. Also we see plasmocytoid dendritic cells (marked by IRF8) being close to B cells along the PC2.

# In[22]:


sc.pl.pca_variance_ratio(rna, log=True)


# Now we can compute a neighbourhood graph for cells:

# In[23]:


sc.pp.neighbors(rna, n_neighbors=10, n_pcs=20)


# #### Non-linear dimensionality reduction and clustering

# With the neighbourhood graph computed, we can now perform clustering. We will use `leiden` clustering as an example.

# In[24]:


sc.tl.leiden(rna, resolution=.5)


# To visualise the results, we'll first generate a 2D latent space with cells that we can colour according to their cluster assignment.

# In[25]:


sc.tl.umap(rna, spread=1., min_dist=.5, random_state=11)


# In[26]:


sc.pl.umap(rna, color="leiden", legend_loc="on data")


# #### Marker genes and celltypes

# In[27]:


sc.tl.rank_genes_groups(rna, 'leiden', method='t-test')


# In[28]:


result = rna.uns['rank_genes_groups']
groups = result['names'].dtype.names
pd.set_option('display.max_columns', 50)
pd.DataFrame(
    {group + '_' + key[:1]: result[key][group]
    for group in groups for key in ['names', 'pvals']}).head(10)


# In[29]:


sc.pl.rank_genes_groups(rna, n_genes=20, sharey=False)


# Exploring the data we notice clusters 9 and 15 seem to be composed of cells bearing markers for different cell lineages so likely to be noise (e.g. doublets). Cluster 12 has higher ribosomal gene expression when compared to other clusters. Cluster 16 seem to consist of proliferating cells.
# 
# We will remove cells from these clusters before assigning cell types names to clusters.

# In[30]:


mu.pp.filter_obs(rna, "leiden", lambda x: ~x.isin(["9", "15", "12", "16"]))
# Analogous to
#   rna = rna[~rna.obs.leiden.isin(["9", "15", "12", "16"])]
# but doesn't copy the object


# In[31]:


new_cluster_names = {
    "0": "CD4+ memory T", "1": "CD8+ naïve T", "3": "CD4+ naïve T", 
    "5": "CD8+ activated T", "7": "NK", "13": "MAIT",
    "6": "memory B", "10": "naïve B",
    "4": "CD14 mono", "2": "intermediate mono", "8": "CD16 mono",
    "11": "mDC", "14": "pDC",
}

rna.obs['celltype'] = rna.obs.leiden.astype("str").values
rna.obs.celltype = rna.obs.celltype.astype("category")
rna.obs.celltype = rna.obs.celltype.cat.rename_categories(new_cluster_names)


# We will also re-order categories for the next plots:

# In[32]:


rna.obs.celltype.cat.reorder_categories([
    'CD4+ naïve T', 'CD4+ memory T', 'MAIT',
    'CD8+ naïve T', 'CD8+ activated T', 'NK',
    'naïve B', 'memory B',
    'CD14 mono', 'intermediate mono', 'CD16 mono',
    'mDC', 'pDC'], inplace=True)


# ... and take colours from a palette:

# In[33]:


import matplotlib
import matplotlib.pyplot as plt

cmap = plt.get_cmap('rainbow')
colors = cmap(np.linspace(0, 1, len(rna.obs.celltype.cat.categories)))

rna.uns["celltype_colors"] = list(map(matplotlib.colors.to_hex, colors))


# In[34]:


sc.pl.umap(rna, color="celltype", legend_loc="on data")


# Finally, we'll visualise some marker genes across cell types.

# In[35]:


marker_genes = ['IL7R', 'TRAC',
                'ITGB1', # CD29
                'SLC4A10',
                'CD8A', 'CD8B', 'CCL5',
                'GNLY', 'NKG7',
                'CD79A', 'MS4A1', 'IGHM', 'IGHD',
                'IL4R', 'TCL1A',
                'KLF4', 'LYZ', 'S100A8', 'ITGAM', # CD11b 
                'CD14', 'FCGR3A', 'MS4A7', 
                'CST3', 'CLEC10A', 'IRF8', 'TCF4']


# In[36]:


sc.pl.dotplot(rna, marker_genes, groupby='celltype');


# ## Saving multimodal data on disk

# We will now write `mdata` object to an `.h5mu` file. It will contain all the changes we've done to the RNA modality (`mdata.mod['rna']`) inside it.

# In[37]:


mdata.write("data/pbmc10k.h5mu")


# [Next, we'll look into processing the second modality — chromatin accessibility.](2-Chromatin-Accessibility-Processing.ipynb)
# 
