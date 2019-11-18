
import pandas as pd
import numpy as np

CSV = 'indoor_selection_time_series/df_timeseries_1000to1958_skip2.csv'
TIMESCALE = 480  # 10768 xyz values in each timestep !
XYZSCALE = 10768
dim1 = 'Points:0'
dim2 = 'Points:1'
dim3 = 'Points:2'
dim4 = 'Temperature'
dim5 = 'Pressure'
dim6 = 'Tracer'
FEATURES = [dim1, dim2, dim3, dim4, dim5, dim6]
COLUMNORDER = ['Index_xyz','Time', dim1, dim2, dim3, dim4, dim5, dim6]
COLUMNORDER_time = ['Index_time', 'Time',  dim1, dim2, dim3, 'Temperature', 'Pressure', 'Tracer']
# COLUMNORDER_xyz = ['Index_xyz', dim1, dim2, dim3, 'Temperature_ave', 'Pressure_ave', 'Tracer_ave']
COLUMNORDER_bkt = ['Bucket',  'Index_time', 'Time', 'Temperature', 'Pressure', 'Tracer']
COLUMNORDER_Dset_bkt_TS = ['Bucket', 'Index_time', 'Time', 'Temperature_ave', 'Pressure_ave', 'Tracer_ave']

X = 220.96300
Y = 65.15670
Z = 9.000000

def filter_df_with_xyz(df, x, y, z):
    df = df.loc[df['Points:0'] == x]
    df = df.loc[df['Points:1'] == y]
    df_filtered = df.loc[df['Points:2'] == z]
    return df_filtered

def generate_xyz_idxs(df, timescale):
    df_len = df.shape[0]
    single_xyz_len = 10768

    i = 0
    ones = np.ones(single_xyz_len)
    ones *= i
    xyz_row_idxs = np.array(np.reshape(ones, [-1, 1]))
    i += 1
    while i < timescale:
        ones = np.ones(single_xyz_len)
        ones *= i
        ones = np.array(np.reshape(ones, [-1, 1]))

        xyz_row_idxs = np.vstack((xyz_row_idxs, ones))
        i += 1
    return xyz_row_idxs


def generate_time_idxs(df):
    # take df, append a Index_time column value for each unique timestep

    df.sort_values(by=['Time'], inplace=True)
    df = df.reset_index(drop=True)

    i = 0
    j = 0
    time_val_i = pd.to_numeric(df.loc[0]['Time'])
    df_bucket_i = df.loc[df['Time']==time_val_i]  # make enough ones to append adjacent to the bucket
    ones = np.ones(df_bucket_i.shape[0])
    ones *= j
    i += len(df_bucket_i)
    j += 1
    time_row_idxs = np.array(np.reshape(ones, [-1, 1]))

    while i < len(df): # through all the rows
        time_val_i = pd.to_numeric(df.loc[0]['Time'])
        df_bucket_i = df.loc[df['Time'] == time_val_i]  # make enough ones to append adjacent to the bucket
        ones = np.reshape(np.ones(len(df_bucket_i)), [-1,1])
        ones *= j
        i += len(df_bucket_i)
        time_row_idxs = np.vstack((time_row_idxs, ones))
        j += 1

    df['Index_time'] = time_row_idxs
    df = df.reindex(columns=COLUMNORDER_time)
    return df


def generate_tp_idxs(timescale):
    tp_df = np.random.uniform(0., timescale, (2))  # pressure, tracer
    tp_df = np.ceil(tp_df)
    return tp_df[0], tp_df[1]

def generate_N_tp_from_df(df_xyz, N):
    i = 0
    t_arr = np.ones(N)
    p_arr = np.ones(N)
    while i < N:
        t_idx, p_idx = generate_tp_idxs(TIMESCALE)
        t_idx = t_idx.astype(int)
        p_idx = p_idx.astype(int)
        t_val = df_xyz.iloc[t_idx, 5]
        p_val = df_xyz.iloc[p_idx, 6]
        t_arr[i] = t_val
        p_arr[i] = p_val
        i += 1
    t_arr = np.reshape(t_arr, [-1, 1])
    p_arr = np.reshape(p_arr, [-1, 1])
    return t_arr, p_arr

def generate_xyztp(df_xyz, N):
    t_arr, p_arr = generate_N_tp_from_df(df_xyz, N)
    df_ = pd.DataFrame()
    df_['Points:0'] = df_xyz['Points:0']
    pass


def create_Dset_xyz_TS():
    df = pd.read_csv(CSV, encoding='utf-8', engine='c')
    df.drop(columns=['Unnamed: 0'], inplace=True)
    xyz_rows = generate_xyz_idxs(df, TIMESCALE)
    assert (xyz_rows.shape[0] == df.shape[0])
    df['Index_xyz'] = xyz_rows
    df = df.reindex(columns=COLUMNORDER)
    return df


def create_Dset_xyz_ave(df_):
    # 1. drop time from df
    # 2. break according to Index_xyz blocks
    #       -> for 1 Index_xyz -> 1 np.ones
    #       -> take average of lhs Temperature, Pressure
    #       -> assign to df_ave column
    # 3. append blocks as now rows one after another
    df = df_.copy()
    df.drop(columns=['Time'], inplace=True)
    df_ave = pd.DataFrame()
    i = 0
    while i <= XYZSCALE:  # iterate through each unique xyz to create its average values over time
        df_filtered = df.loc[df['Index_xyz'] == i]  # df of unique xyz
        xyz_ave = pd.DataFrame()
        xyz_ave['Index_xyz'] = df_filtered[0:1]['Index_xyz']
        xyz_ave['Points:0'] = df_filtered[0:1]['Points:0']
        xyz_ave['Points:1'] = df_filtered[0:1]['Points:1']
        xyz_ave['Points:2'] = df_filtered[0:1]['Points:2']
        xyz_ave['Temperature_ave'] = df_filtered['Temperature'].mean()
        xyz_ave['Pressure_ave'] = df_filtered['Pressure'].mean()
        xyz_ave['Tracer_ave'] = df_filtered['Tracer'].mean()
        df_ave = df_ave.append(xyz_ave)
        if i % 100 == 0:
            print("Dset_xyz_ave", i , "/", XYZSCALE)
        i += 1

    return df_ave

def average_values_for_time_per_bucket(df_bkt_):

    df_ave = pd.DataFrame()  # all Buckets
    i = 0
    while i <= df_bkt_['Bucket'].max():
        df_bkt_filt = df_bkt_.loc[df_bkt_['Bucket'] == i]  # df of unique xyz
        if len(df_bkt_filt) > 0:
            xyz_ave = pd.DataFrame()  # 1 Bucket
            j = 0
            while j <= df_bkt_filt['Index_time'].max():
                df_filtered_ = df_bkt_filt.loc[df_bkt_filt['Index_time'] == j]
                if len(df_filtered_) > 0:  # in the Bucket there is an Index_time...
                    xyz_ave.loc[0, 'Bucket'] = df_filtered_.iloc[0, 0]  # has 1 row ?
                    xyz_ave.loc[0, 'Index_time'] = df_filtered_.iloc[0, 1]
                    xyz_ave.loc[0, 'Time'] = df_filtered_.iloc[0, 2]

                    xyz_ave['Temperature_ave'] = df_filtered_['Temperature'].mean()
                    xyz_ave['Pressure_ave'] = df_filtered_['Pressure'].mean()
                    xyz_ave['Tracer_ave'] = df_filtered_['Tracer'].mean()

                    assert not np.isnan(xyz_ave.iloc[0, 0])
                    assert not np.isnan(xyz_ave.iloc[0, 1])
                    assert not np.isnan(xyz_ave.iloc[0, 2])

                    # df_ave = pd.concat([df_ave, xyz_ave], ignore_index=True, axis=1)
                    df_ave = df_ave.append(xyz_ave)

                j += 1
        print("Dset_bkt_TS, average values", i, "/", df_bkt_['Bucket'].max())
        i += 1
    return df_ave



def create_Dset_bkt_TS(df):
    # 0. WHILE loop for each bucket of values. (NOTE: each bkt may have different number of rows)
    # 1. select bucket of values
    #       -> append column of unique time values: Index_time. Same as with Index_xyz
    #       -> drop Index_xyz, x, y, z
    #       -> sort by Time (TIME takes role of Index_xyz)
    #           inner WHILE loop - create unique timesteps for each bucket
    #           j < rows_in_bkt (number of observations in that bucket in 1 time step.)
    #           -> filter to just 1 Time -> get average values for this time
    #

    # ------------------------------------------
    # CREATE TIME INDICES FOR df_bkt -> df__
    # ------------------------------------------
    df_ = generate_time_idxs(df)

    #------------------------------------------
    # SPECIFY BUCKETS
    #------------------------------------------
    # Globals to create bucket column
    split_x = 4
    split_y = 4
    split_z = 4
    MIN_x = df_['Points:0'].min()
    MIN_y = df_['Points:1'].min()
    MIN_z = df_['Points:2'].min()
    MAX_x = df_['Points:0'].max()
    MAX_y = df_['Points:1'].max()
    MAX_z = df_['Points:2'].max()
    LEN_x = MAX_x - MIN_x
    LEN_y = MAX_y - MIN_y
    LEN_z = MAX_z - MIN_z
    LEN_BKT_x = LEN_x / split_x
    LEN_BKT_y = LEN_y / split_y
    LEN_BKT_z = LEN_z / split_z

    def calc_bkt_idx_for_row(df_row):
        # take_row, return row with appended col: int32 bkt_idx
        x = df_row['Points:0']
        y = df_row['Points:1']
        z = df_row['Points:2']
        len_x = x - MIN_x
        len_y = y - MIN_y
        len_z = z - MIN_z
        BUCKET__x_ = len_x // LEN_BKT_x  # each side needs to fall into a specific bucket
        BUCKET__y_ = len_y // LEN_BKT_y
        BUCKET__z_ = len_z // LEN_BKT_z
        # NOW INDICES
        BUCKET__x = np.floor(BUCKET__x_)  # each side needs to fall into a specific bucket
        BUCKET__y = np.floor(BUCKET__y_)
        BUCKET__z = np.floor(BUCKET__z_)
        BUCKET__idx = np.floor(BUCKET__x
                            + (BUCKET__y *  split_x)
                            + (BUCKET__z * (split_x * split_y)))    # from lower corner to opposite upper corner
                                                                    # Bucket_0   = xmin, ymin, zmin
                                                                    # Bucket_max = xmax, ymax, zmax
        BUCKET__idx.astype(int)
        df_bkt_row = df_row
        df_bkt_row.loc['Bucket'] = BUCKET__idx
        df_bkt_row = df_bkt_row.reindex(columns=COLUMNORDER_bkt)
        return df_bkt_row  # returns a single row

    A = df_['Index_time'].max()
    print("max Index_time", A)
    a = 0
    while a < A:
        print("Index_time = "+str(a), len(df_.loc[df_['Index_time'] == a]))
        a+=1

    ts = 200
    df_sel = pd.DataFrame()
    while ts < 480:
        df_0 = df_.loc[df_['Index_time'] == ts].head(100)
        df_sel = pd.concat([df_sel,df_0], ignore_index=True,axis=0)
        ts += 20

    i = 0
    df_bkt = pd.DataFrame()  # df with corresponding Bucket for each xyz value
    while i < len(df_sel): # len(df_):
        df_row = df_sel[i:i+1]
        df_bkt_row = calc_bkt_idx_for_row(df_row)  #
        assert not np.isnan(df_bkt_row.iloc[0,0])
        df_bkt = df_bkt.append(df_bkt_row)
        if i % 50 == 0:
            print("Dset_bkt_TS", i, "/", len(df_sel))
        i += 1

    df_bkt.sort_values(by=['Bucket', 'Index_time'], inplace=True)
    df_bkt = df_bkt.reset_index(drop=True)

    df_bkt = average_values_for_time_per_bucket(df_bkt)

    return  df_bkt

if __name__ == '__main__':
    df_ = create_Dset_xyz_TS()
    # Dset_xyz_ave = create_Dset_xyz_ave(df_)  # for GP
    # Dset_xyz_ave.to_csv('main_datasets_/Dset_xyz_ave_small2.csv')

    Dset_bkt_TS = create_Dset_bkt_TS(df_)  # for COV
    Dset_bkt_TS.to_csv('main_datasets_/Dset_bkt_TS_mid1.csv')

    pass