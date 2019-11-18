
import pandas as pd

CSV1500 = 'roomselection_1500.csv'
CSV1800 = 'roomselection_1800.csv'
dim1 = 'Points:0'
dim2 = 'Points:1'
dim3 = 'Points:2'
dim4 = 'Temperature'
dim5 = 'Pressure'
dim6 = 'Tracer'
FEATURES = [dim1, dim2, dim3, dim4, dim5, dim6]
COLUMNORDER = ['Time', dim1, dim2, dim3, dim4, dim5, dim6]

def drop_cols(df_):
    df_.drop(columns=['TemperatureAverage', 'TracerAverage',
                      'Velocity:0', 'Velocity:1', 'Velocity:2',
                      'GravityDirection:0', 'GravityDirection:1', 'GravityDirection:2',
                      'vtkOriginalPointIds'], inplace=True)
    return df_

ITER = 2
i = 1000
df = pd.read_csv('indoor_selection_csv_08_01/1500_timestep.'+ str(i) +'.csv', encoding='utf-8', engine='c')
df = drop_cols(df)
df = df.reindex(columns=COLUMNORDER)
i += ITER

while i <= 1958:
    CSV_i ='indoor_selection_csv_08_01/1500_timestep.'+ str(i) +'.csv'
    df_1 = pd.read_csv(CSV_i, encoding='utf-8', engine='c')
    df_1 = drop_cols(df_1)
    df_1 = df_1.reindex(columns=COLUMNORDER)

    df = pd.concat([df, df_1])
    df.sort_values(by=[dim1, dim2, dim3,'Time'], inplace=True)

    i += ITER
    print(".")

df.to_csv('indoor_selection_time_series/df_timeseries_1000to2000_skip2___.csv')
pass