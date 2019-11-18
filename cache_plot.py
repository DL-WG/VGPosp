import pandas as pd
import gp_demo.subplots as subp
from mpl_toolkits.mplot3d import Axes3D  # REQ: projection='3d'
import numpy as np
import main_architecture_2_sampledistribution as a2sd
import main_tests.alg2_split10x10x10_50x50x10_tests as a2t

COVER_spatial, file_cov_vv, file_cov_idxs, file_delt_ch_it, SAVE_SELECTION_file, \
    GEN_LOCAL_kernel, BETA_val = a2t.get_spatial_splits(a2t.indirection)

override = True
if override:
    file_cov_idxs = 'main_datasets_/placement_algorithm_xyz_cov_idxs26x26x1_full.csv'
    file_delt_ch_it = 'main_datasets_/placement_algorithm_cache26x26x1_full.csv'

dfc = pd.read_csv(file_delt_ch_it, encoding='utf-8', engine='c')
dfxyz = pd.read_csv(file_cov_idxs, encoding='utf-8', engine='c')
dfc.drop(columns=['Unnamed: 0'], inplace=True)
dfxyz.drop(columns=['Unnamed: 0'], inplace=True)

# if there are more places than values:
if len(dfc) < len(dfxyz):
    dfxyz = dfxyz[:len(dfc)]

fig_rows, fig_cols = 2, 4
fig3, ax3 = subp.make_axis_3d(fig_rows, fig_cols,
                              figsize=(20, 10),
                              projection='3d')
# fig_i = 0
# a_1 = dfxyz.iloc[:, [1, 2]].to_numpy()[np.newaxis, ...]
# a_2 = dfc.iloc[:, 1].to_numpy()[np.newaxis, ...]

fig_i = 0
for j in range(dfc.shape[1]):
    fig_i = j
    ax_ = ax3[fig_i // fig_cols][fig_i % fig_cols]
    # drop the negative values
    dfc_np = dfc.iloc[:, fig_i].to_numpy()
    # dfc_ = np.max(dfc_np, np.zeros_like(dfc_np))
    assert 1 == len(dfc_np.shape)
    for i in range(dfc_np.shape[0]):
        dfc_np[i] = max(0., dfc_np[i])
    subp.add_3d(
        ax_, fig3,
        'cache{}(x,y)'.format(fig_i),
        dfxyz.iloc[:, [0, 1]].to_numpy()[np.newaxis, ...], None, dfc_np[np.newaxis, ...], None,
        None, None,
        None, None, edges=300)
    # ax_.set_ylim()
    # ax_.set_zlim3d(bottom=0.)

fig_i += 1
# ax3[fig_i // fig_cols, fig_i % fig_cols].remove()
# fig3.show()

while fig_i < fig_rows * fig_cols:
    # ax[fig_i // fig_cols, fig_i % fig_cols].remove()
    ax3[fig_i // fig_cols, fig_i % fig_cols].remove()
    fig_i += 1

fig3.savefig('vector_plots/mutual_information_cache.eps', format='eps')
fig3.show()
pass

if __name__ == '__main__':
    pass
