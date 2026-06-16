"""t-SNE map of representative tiles across the six GMM prototypes (Pheno-MYCN).

Embeds the per-prototype representative-tile features with t-SNE and renders the
combined scatter coloured by prototype. Stand-alone script (no argparse): edit
``csv_paths`` and the ``plt.savefig`` target below to point at your own data.

Author: Dr Olga Fourkioti (https://github.com/olgarithmics). License: GPL-3.0.
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from scipy.spatial.distance import cdist
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE

csv_paths = [
    '/path/to/results/fold_9/images/top_df_proto_0.csv',
    '/path/to/results/fold_9/images/top_df_proto_1.csv',
    '/path/to/results/fold_9/images/top_df_proto_2.csv',
    '/path/to/results/fold_9/images/top_df_proto_3.csv',
    '/path/to/results/fold_9/images/top_df_proto_4.csv',
    '/path/to/results/fold_9/images/top_df_proto_5.csv',
]

# Load each prototype table and label its rows 1-6.
prototype_dfs = []
for i, path in enumerate(csv_paths, start=1):
    df = pd.read_csv(path)
    df['prototype'] = i
    prototype_dfs.append(df)
all_patches_df = pd.concat(prototype_dfs, ignore_index=True)
print("✅ Combined DataFrame shape:", all_patches_df.shape)

# Numeric feature columns only (excluding the prototype label), mean-imputed.
feature_columns = all_patches_df.select_dtypes(include=[np.number]).columns.tolist()
feature_columns = [col for col in feature_columns if col != 'prototype']
features_df = all_patches_df[feature_columns]
X_imputed = SimpleImputer(strategy='mean').fit_transform(features_df)
print("✅ NaNs in final data:", np.isnan(X_imputed).sum())

X_tsne = TSNE(n_components=2, random_state=42).fit_transform(X_imputed)

mpl.rcParams.update({'font.size': 12, 'figure.dpi': 200})
plt.figure(figsize=(7, 5.5))
cmap = get_cmap('Set2', 6)

for p in range(1, 7):
    mask = all_patches_df['prototype'] == p
    cluster_points = X_tsne[mask]
    plt.scatter(
        cluster_points[:, 0], cluster_points[:, 1],
        color=cmap(p - 1), s=25, alpha=0.7, edgecolor='white', linewidth=0.3,
    )
    # Label each cluster on the point closest to its centroid.
    centroid = cluster_points.mean(axis=0)
    closest_idx = np.argmin(cdist([centroid], cluster_points))
    label_point = cluster_points[closest_idx]
    plt.text(
        label_point[0], label_point[1], str(p),
        fontsize=10, fontweight='bold', color='black', ha='center', va='center',
    )

plt.xticks([])
plt.yticks([])
plt.xlabel('')
plt.ylabel('')
plt.box(False)
plt.tight_layout()
plt.savefig('/path/to/results/fold_9/images/tsne_cluster_labels_nobox.png', dpi=300, bbox_inches='tight')
plt.show()
