import sys
# sys.path.append('/home/gtajn/prg/fluidity/python')
import vtktools
import numpy as np
from vtk import *
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib.pyplot as plt
import os

plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['grid.color'] = '#666666'


i = 620
num = 0
dim = np.zeros([1,3])
pressure = np.zeros([1,3])
velocity = np.zeros([1,3])
temperature = np.zeros([1,3])
tracer = np.zeros([1,3])
time = np.zeros([1,3])
density = np.zeros([1,3])

FILELOC = '/Ubuntu_18/data_room/data_selections/room_selection_0001_'+str(i)+'.vtu'

while( num < 5 and i < 1900 ):
    filename = FILELOC
    if (not os.path.isfile(filename)) or (os.stat(filename).st_size == 0):
        i+=1
        continue

    # Coordinates
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()
    data = reader.GetOutput()
    points = data.GetPoints()
    npts = data.GetNumberOfPoints()
    dim_new = vtk_to_numpy(points.GetData())

    # Features
    features = vtktools.vtu(filename)
    pressure_ = features.GetScalarField('Pressure')
    velocity_ = features.GetVectorField('Velocity')
    temperature_ = features.GetScalarField('Temperature')
    tracer_ = features.GetScalarField('Tracer')
    time_ = features.GetScalarField('Time')
    density_ = features.GetScalarField('Density')

    print("iteration: " + str(i))
    if num != 0:
        dim = np.vstack((dim, dim_new))
        ####
        pressure = np.vstack((pressure, pressure_))
        velocity = np.vstack((velocity, velocity_))
        temperature = np.vstack((temperature, temperature_))
        tracer = np.vstack((tracer, tracer_))
        time = np.vstack((time, time_))
        density= np.vstack((density, density_))
    else:
        dim = dim_new
        ####
        pressure = pressure_
        velocity = velocity_
        temperature = temperature_
        tracer = tracer_
        time = time_
        density = density_
    i += 1
    num+=1

#Coordinates
x = dim[:,0]
y = dim[:,1]
z = dim[:,2]

################  PRINTS ###################
PRINT_VALUES = True

def plot_tracer_histogram(fw,fh, feature):
    plt.figure(figsize=(fw, fh))
    plt.hist(feature, bins=700)
    plt.title("Histogram with 50 bins, values, ~ 16000")
    plt.show()

if PRINT_VALUES:
    print(time.shape, time)
    print(tracer.shape, tracer)
    print("start plots, tracer")
    plot_tracer_histogram(10,4, tracer.flatten())
    print("start time")
    plot_tracer_histogram(10,4, time.flatten())
    plot_tracer_histogram(10,4, temperature.flatten())
    plot_tracer_histogram(10,4, pressure.flatten())
    plot_tracer_histogram(10,4, density.flatten())



