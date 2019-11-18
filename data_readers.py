import sys
# sys.path.append('/home/gtajn/prg/fluidity/python')
# import vtktools
import numpy as np
# from vtk import *
# from vtk.util.numpy_support import vtk_to_numpy
import matplotlib.pyplot as plt
import pandas as pd
import os
#%pylab inline

def create_dataframe(a_title, a_content,b_title, b_content,c_title,
                     c_content, d_title, d_content,e_title, e_content,
                     f_title, f_content):
    data_raw = {a_title:[a_content],b_title:[b_content],
                c_title:[c_content],d_title:[d_content],
                e_title:[e_content],f_title:[f_content]}
    columns = [a_title, b_title, c_title, d_title, e_title, f_title]
    df = pd.DataFrame(data_raw, columns)
    return df


def create_dataframe_coord_tracer(a_title, a_content,b_title, b_content,
                                           c_title, c_content,d_title, d_content):
    print("x", a_content.shape)
    print("y", b_content.shape)
    print("z", c_content.shape)
    print("tracer", d_content.shape)

    # data_raw = {a_title:[a_content],b_title:[b_content],c_title:[c_content],d_title:[d_content]}
    df = pd.DataFrame(np.array([[a_content],[b_content],[c_content],[d_content]]),
                      columns=[a_title, b_title, c_title, d_title])
    return df


def create_dataframe_coord_pressure_tracer(a_title, a_content,b_title, b_content,
                                           c_title, c_content,d_title, d_content,
                                           e_title, e_content):
    data_raw = {a_title:[a_content],b_title:[b_content],c_title:[c_content],d_title:[d_content],
                e_title:[e_content]}
    columns = [a_title, b_title, c_title, d_title, e_title]
    df = pd.DataFrame(data_raw, columns)
    return df


def load_csv(filename):
    # e = pd.read_csv(filename)
    e = pd.read_csv(open(filename, 'rU'), encoding='utf-8', engine='c')
    return e

def save_dataframe(dataframe, filename):
    dataframe.to_csv(filename)


# def load_pvtk_to_arrays_coord_tr_pr(timerange):
#     i = timerange[0]
#     FILELOC = '/Ubuntu_18/data_room/data_selections/room_selection_0001_' + str(i) + '.vtu'
#
#     dim = np.zeros([0, 3])
#     tracer = np.zeros([0, 1])
#     pressure = np.zeros([0, 1])
#
#     while (i < timerange[1]): # from 620 - 1900 #num < 5 and
#         filename = FILELOC
#         if (not os.path.isfile(filename)) or (os.stat(filename).st_size == 0):
#             i += 1
#             continue
#         # Coordinates
#         reader = vtk.vtkXMLUnstructuredGridReader()
#         reader.SetFileName(filename)
#         reader.Update()
#         data = reader.GetOutput()
#         points = data.GetPoints()
#         dim_new = vtk_to_numpy(points.GetData())
#
#         features = vtktools.vtu(filename)
#         tracer_ = features.GetScalarField('Tracer')
#         pressure_ = features.GetScalarField('Pressure')
#
#         tracer_.shape = (-1,1)
#         pressure_.shape = (-1,1)
#
#         dim = np.vstack((dim, dim_new))
#         tracer = np.vstack((tracer, tracer_))
#         pressure = np.vstack((tracer, pressure_))
#         i += 1
#         print(".")
#
#     x = dim[:,0].reshape(-1,1)
#     y = dim[:,1].reshape(-1,1)
#     z = dim[:,2].reshape(-1,1)
#     return x,y,z, tracer, pressure #, time, temperature, pressure, velocity, density


# def load_pvtk_to_arrays(timerange):
#     i = timerange[0]
#     FILELOC = '/Ubuntu_18/data_room/data_selections/room_selection_0001_' + str(i) + '.vtu'
#     # num = 0
#
#     dim = np.zeros([0, 3])
#     tracer = np.zeros([0, 1])
#     pressure = np.zeros([0, 1])
#     velocity = np.zeros([0, 1])
#     temperature = np.zeros([0, 1])
#     time = np.zeros([0, 1])
#     density = np.zeros([0, 1])
#
#     while (i < timerange[1]): # from 620 - 1900 #num < 5 and
#         filename = FILELOC
#         if (not os.path.isfile(filename)) or (os.stat(filename).st_size == 0):
#             i += 1
#             continue
#
#         # Coordinates
#         reader = vtk.vtkXMLUnstructuredGridReader()
#         reader.SetFileName(filename)
#         reader.Update()
#         data = reader.GetOutput()
#         points = data.GetPoints()
#         dim_new = vtk_to_numpy(points.GetData())
#
#         # Features from a single vtk
#         features = vtktools.vtu(filename)
#         tracer_ = features.GetScalarField('Tracer')
#
#         pressure_ = features.GetScalarField('Pressure')
#         velocity_ = features.GetVectorField('Velocity')
#         temperature_ = features.GetScalarField('Temperature')
#         time_ = features.GetScalarField('Time')
#         density_ = features.GetScalarField('Density')
#
#         tracer_.shape = (-1,1)
#         pressure_.shape = (-1, 1)
#         velocity_.shape = (-1, 1)
#         temperature_.shape = (-1, 1)
#         time_.shape = (-1, 1)
#         density_.shape = (-1, 1)
#         # if i ==0:
#         #     dim = dim_new
#         #     tracer = tracer_
#         #     pressure = pressure_
#         #     velocity = velocity_
#         #     temperature = temperature_
#         #     density = density_
#         # else:
#         dim = np.vstack((dim, dim_new))
#         tracer = np.vstack((tracer, tracer_))
#         pressure = np.vstack((pressure, pressure_))
#         velocity = np.vstack((velocity, velocity_))
#         temperature = np.vstack((temperature, temperature_))
#         time = np.vstack((time, time_))
#         density = np.vstack((density, density_))
#
#         i += 1
#         print(".")
#     return dim, tracer, time, temperature, pressure, velocity, density