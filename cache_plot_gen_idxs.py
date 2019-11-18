import pandas as pd
import gp_demo.subplots as subp
from mpl_toolkits.mplot3d import Axes3D  # REQ: projection='3d'
import numpy as np
import main_architecture_2_sampledistribution as a2sd
import main_tests.alg2_split10x10x10_50x50x10_tests as a2t


def gen_idxs(input_splits #=COVER_spatial
             , output_file # =file_cov_idxs
            ):

    # dfxyz = pd.read_csv(file_cov_idxs, encoding='utf-8', engine='c')
    # dfc.drop(columns=['Unnamed: 0'], inplace=True)
    # dfxyz.drop(columns=['Unnamed: 0'], inplace=True)

    [I0, I1, I2] = input_splits
    N = I0 * I1 * I2

    xyz_cov_idxs_ = np.zeros([N, 3], dtype=np.int32)

    stride0 = I2 * I1
    stride1 = I2
    stride2 = 1

    for i0 in range(I0):
        for i1 in range(I1):
            for i2 in range(I2):
                line = stride0*i0 + stride1*i1 + stride2*i2
                xyz_cov_idxs_[line, :] = [i0, i1, i2]

    pd.DataFrame(xyz_cov_idxs_).to_csv(output_file)

    pass

if __name__ == '__main__':

    # COVER_spatial, file_cov_vv, file_cov_idxs, file_delt_ch_it, SAVE_SELECTION_file, \
    # GEN_LOCAL_kernel, BETA_val = a2t.get_spatial_splits()

    COVER_spatial = [50, 50, 1]
    tag = str(COVER_spatial[0])+'x'+str(COVER_spatial[1])+'x'+str(COVER_spatial[2])
    file_cov_idxs = 'main_datasets_/placement_algorithm_xyz_cov_idxs' + tag + '_gen.csv'

    gen_idxs(input_splits=COVER_spatial, output_file=file_cov_idxs)
    pass
