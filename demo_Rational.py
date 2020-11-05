""" Generates data to show the effect of rescaling. Low density basisfunctions used. """

import pandas
import os
import logging
from rbf import *
import basisfunctions, testfunctions
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits import mplot3d 
import time
import mesh
import math
from random import randint
from scipy import spatial
from halton import *
import vtk
import mesh_io

class Mesh:
	"""
	A Mesh consists of:
		- Points: A list of tuples of floats representing coordinates of points
		- Cells: A list of tuples of ints representing mesh elements
		- Pointdata: A list of floats representing data values at the respective point
	"""
	def __init__(self, points = None, cells = None, cell_types = None, pointdata = None):
		if points is not None:
			self.points = points
		else:
			self.points = []
		if cells is not None:
			assert(cell_types is not None)
			self.cells = cells
			self.cell_types = cell_types
		else:
			self.cells = []
			self.cell_types = []
		if pointdata is not None:
			self.pointdata = pointdata
		else:
			self.pointdata = []

		def __str__(self):
			return "Mesh with {} Points and {} Cells ({} Cell Types)".format(len(self.points), len(self.cells), len(self.cell_types))

def read_mesh(filename):
	points, cells, cell_types, pointdata = mesh_io.read_mesh(filename)
	#print("Points: ", len(points))
	#print("Point data: ", pointdata)
	return Mesh(points, cells, cell_types, pointdata)

mesh_name = "Mesh/Plate/l1Data.vtk"
mesh = read_mesh(mesh_name)
#print("Number of points: ", mesh.points)

start = time.time()
j = 0
#nPoints = len(mesh.points)
nPoints = 100
nPointsOut = 500
#print("Number of points: ",nPoints)
inLen = 40
outLen = 60
edgeLengthX = 2
edgeLengthY = 2
xMinLength = 0.0
yMinLength = 0.0
in_size = np.linspace(xMinLength, edgeLengthX + xMinLength, inLen)
out_size = np.linspace(yMinLength, edgeLengthY + yMinLength, outLen)
in_mesh = np.random.random((pow(inLen,2),2))
out_mesh = np.random.random((pow(outLen,2),2))
out_mesh_Combined = np.random.random((pow(outLen,2),2))
out_mesh_Split = np.random.random((pow(outLen,2),2))
out_mesh_Combined_value = []
out_mesh_Split_value = []

for i in range(0,inLen):
	for j in range(0,inLen):
		in_mesh[j+i*inLen,0] = (edgeLengthX/inLen)*j + xMinLength
		in_mesh[j+i*inLen,1] = (edgeLengthY/inLen)*i + yMinLength

for i in range(0,outLen):
	for j in range(0,outLen):
		out_mesh[j+i*outLen,0] = (edgeLengthX/outLen)*j + xMinLength
		out_mesh[j+i*outLen,1] = (edgeLengthY/outLen)*i + yMinLength
		out_mesh_Combined[j+i*outLen,0] = (edgeLengthX/outLen)*j + xMinLength
		out_mesh_Combined[j+i*outLen,1] = (edgeLengthY/outLen)*i + yMinLength
		out_mesh_Split[j+i*outLen,0] = (edgeLengthX/outLen)*j + xMinLength
		out_mesh_Split[j+i*outLen,1] = (edgeLengthY/outLen)*i + yMinLength

#mesh_size = 1/math.sqrt(nPoints)
mesh_size = edgeLengthX/inLen
shape_parameter = 4.55228/((4.0)*mesh_size)
bf = basisfunctions.Gaussian(shape_parameter)
func = lambda x,y: np.sin(5*x*y)+(0.0000001*y)
funcTan = lambda x,y: np.arctan(125*(pow(pow(x-1.5,2) + pow(y-0.25,2),0.5) - 0.92))
one_func = lambda x: np.ones_like(x)
in_vals = func(in_mesh[:,0],in_mesh[:,1])
out_vals = func(out_mesh[:,0],out_mesh[:,1])

start = time.time()
interpRational = Rational(bf, in_mesh, in_vals, rescale = False)
end = time.time()
print("Time for inversion: ", end-start)	
start = time.time()
fr = interpRational(in_vals, out_mesh)
end = time.time()
print("Time for eigen decomposition: ", end-start)



interp = NoneConsistent(bf, in_mesh, in_vals, rescale = False)
fr_regular = interp(out_mesh)


#out_vals = funcTan(out_mesh[:,0], out_mesh[:,1])
print("out_vals: ", max(fr))
print("Error fr= ", np.linalg.norm(out_vals - fr, 2))
print("Error fr_regular= ", np.linalg.norm(out_vals - fr_regular, 2))
maxRegError = max(out_vals - fr_regular)
print("max fr: ", max(out_vals - fr))
print("max regular: ", maxRegError)



#plt.scatter(in_mesh[:,0], in_mesh[:,1], label = "In Mesh", s=2)
#plt.scatter(out_mesh[:,0], out_mesh[:,1], label = "Out Mesh", s=2)



# Make data.
#X = np.arange(-5, 5, 0.25)
Xtotal = np.linspace(xMinLength, edgeLengthX + xMinLength, outLen)
Ytotal = np.linspace(yMinLength, edgeLengthY + yMinLength, outLen)
#Y = np.arange(-5, 5, 0.25)
Xtotal, Ytotal = np.meshgrid(Xtotal, Ytotal)
#R = np.sqrt(X**2 + Y**2)
#Z = np.sin(R)
Z = np.arctan(125*(pow(pow(Xtotal-1.5,2) + pow(Ytotal-0.25,2),0.5) - 0.92))
Z_combined = np.arctan(125*(pow(pow(Xtotal-1.5,2) + pow(Ytotal-0.25,2),0.5) - 0.92))
Z_split = np.arctan(125*(pow(pow(Xtotal-1.5,2) + pow(Ytotal-0.25,2),0.5) - 0.92))
Z_regular = np.arctan(125*(pow(pow(Xtotal-1.5,2) + pow(Ytotal-0.25,2),0.5) - 0.92))
Z_regular_error = np.arctan(125*(pow(pow(Xtotal-1.5,2) + pow(Ytotal-0.25,2),0.5) - 0.92))
Z_rational = np.arctan(125*(pow(pow(Xtotal-1.5,2) + pow(Ytotal-0.25,2),0.5) - 0.92))
Z_rational_error = np.arctan(125*(pow(pow(Xtotal-1.5,2) + pow(Ytotal-0.25,2),0.5) - 0.92))
#print(Z)
k=0
for i in range(0,outLen):
	for j in range(0,outLen):
		Z[i,j] = out_vals[k]
		Z_combined[i,j] = out_vals[k]
		Z_split[i,j] = 0
		Z_rational[i,j] = fr[k]
		Z_rational_error[i,j] = out_vals[k]- fr[k]
		Z_regular[i,j] = fr_regular[k]
		Z_regular_error[i,j] = out_vals[k]- fr_regular[k]
		k += 1

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('Actual')
ax.plot_surface(Xtotal, Ytotal, Z_combined,cmap='viridis',linewidth=0,edgecolor='black')
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('Regular')
ax.plot_surface(Xtotal, Ytotal, Z_regular,cmap='viridis',linewidth=0,edgecolor='black')
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('Rational')
ax.plot_surface(Xtotal, Ytotal, Z_rational,cmap='viridis',linewidth=0)
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('Regular')
ax.plot_surface(Xtotal, Ytotal, Z_regular_error,cmap='viridis',linewidth=0)
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('Rational')
ax.plot_surface(Xtotal, Ytotal, Z_rational_error,cmap='viridis',linewidth=0)
plt.show()

#Z = in_vals
# Plot the surface.
#surf = ax.plot_surface(X, Y, Z,cmap='viridis',linewidth=0)
#ax.plot_surface(X, Y, Z_regular,cmap='viridis',linewidth=0)
# Customize the z axis.
#ax.set_zlim(-4.0, 4.0)
#ax.zaxis.set_major_locator(LinearLocator(10))
#ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
#fig.colorbar(surf, shrink=0.5, aspect=5)

#ax.plot_surface(X, Y, Z_regular,cmap='viridis',linewidth=0)
#ax.plot_surface(X, Y, Z_rational,cmap='viridis',linewidth=0)
#plt.show()



#fig, axs = plt.subplots(2, 2)
#axs[0, 0].plot_surface(X, Y, Z,cmap='viridis',linewidth=0)
#axs[0, 0].set_title('Axis [0, 0]')
#axs[0, 1].plot_surface(X, Y, Z_regular,cmap='viridis',linewidth=0)
#axs[0, 1].set_title('Axis [0, 1]')
#axs[1, 0].plot_surface(X, Y, Z_rational,cmap='viridis',linewidth=0)
#axs[1, 0].set_title('Axis [1, 0]')

inLen = 40
outLen = 30
edgeLengthX = 1
edgeLengthY = 1
xMinLength = 0.0
yMinLength = 0.0
in_size = np.linspace(xMinLength, edgeLengthX + xMinLength, inLen)
out_size = np.linspace(yMinLength, edgeLengthY + yMinLength, outLen)
in_mesh = np.random.random((pow(inLen,2),2))
out_mesh = np.random.random((pow(outLen,2),2))
for i in range(0,inLen):
	for j in range(0,inLen):
		in_mesh[j+i*inLen,0] = (edgeLengthX/inLen)*j + xMinLength
		in_mesh[j+i*inLen,1] = (edgeLengthY/inLen)*i + yMinLength

for i in range(0,outLen):
	for j in range(0,outLen):
		out_mesh[j+i*outLen,0] = (edgeLengthX/outLen)*j + xMinLength
		out_mesh[j+i*outLen,1] = (edgeLengthY/outLen)*i + yMinLength

#mesh_size = 1/math.sqrt(nPoints)
mesh_size = edgeLengthX/inLen
shape_parameter = 4.55228/((4.0)*mesh_size)
bf = basisfunctions.Gaussian(shape_parameter)
func = lambda x,y: np.sin(5*x*y)+(0.0000001*y)
funcTan = lambda x,y: np.arctan(125*(pow(pow(x-1.5,2) + pow(y-0.25,2),0.5) - 0.92))
one_func = lambda x: np.ones_like(x)
in_vals = func(in_mesh[:,0],in_mesh[:,1])
out_vals = func(out_mesh[:,0],out_mesh[:,1])

start = time.time()
interpRational = Rational(bf, in_mesh, in_vals, rescale = False)
end = time.time()
print("Time for inversion: ", end-start)	
start = time.time()
fr = interpRational(in_vals, out_mesh)
end = time.time()
print("Time for eigen decomposition: ", end-start)



interp = NoneConsistent(bf, in_mesh, in_vals, rescale = False)
fr_regular = interp(out_mesh)


#out_vals = funcTan(out_mesh[:,0], out_mesh[:,1])
print("out_vals: ", max(fr))
print("Error fr= ", np.linalg.norm(out_vals - fr, 2))
print("Error fr_regular= ", np.linalg.norm(out_vals - fr_regular, 2))
maxRegError = max(out_vals - fr_regular)
print("max fr: ", max(out_vals - fr))
print("max regular: ", maxRegError)



#plt.scatter(in_mesh[:,0], in_mesh[:,1], label = "In Mesh", s=2)
#plt.scatter(out_mesh[:,0], out_mesh[:,1], label = "Out Mesh", s=2)



# Make data.
#X = np.arange(-5, 5, 0.25)
X = np.linspace(xMinLength, edgeLengthX + xMinLength, outLen)
Y = np.linspace(yMinLength, edgeLengthY + yMinLength, outLen)
#Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
#R = np.sqrt(X**2 + Y**2)
#Z = np.sin(R)
Z = np.arctan(125*(pow(pow(X-1.5,2) + pow(Y-0.25,2),0.5) - 0.92))
Z_regular = np.arctan(125*(pow(pow(X-1.5,2) + pow(Y-0.25,2),0.5) - 0.92))
Z_regular_error = np.arctan(125*(pow(pow(X-1.5,2) + pow(Y-0.25,2),0.5) - 0.92))
Z_rational = np.arctan(125*(pow(pow(X-1.5,2) + pow(Y-0.25,2),0.5) - 0.92))
Z_rational_error = np.arctan(125*(pow(pow(X-1.5,2) + pow(Y-0.25,2),0.5) - 0.92))
#print(Z)
k=0
for i in range(0,outLen):
	for j in range(0,outLen):
		Z[i,j] = out_vals[k]
		Z_split[i,j] = out_vals[k]
		Z_rational[i,j] = fr[k]
		Z_rational_error[i,j] = out_vals[k]- fr[k]
		Z_regular[i,j] = fr_regular[k]
		Z_regular_error[i,j] = out_vals[k]- fr_regular[k]
		k += 1

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('Actual')
ax.plot_surface(Xtotal, Ytotal, Z_split,cmap='viridis',linewidth=0,edgecolor='black')
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('Regular')
ax.plot_surface(X, Y, Z_regular,cmap='viridis',linewidth=0,edgecolor='black')
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('Rational')
ax.plot_surface(X, Y, Z_rational,cmap='viridis',linewidth=0)
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('Regular')
ax.plot_surface(X, Y, Z_regular_error,cmap='viridis',linewidth=0)
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('Rational')
ax.plot_surface(X, Y, Z_rational_error,cmap='viridis',linewidth=0)
plt.show()



## ------------------------------------------------------------

inLen = 40
outLen = 30
edgeLengthX = 1
edgeLengthY = 1
xMinLength = 1.0
yMinLength = 0.0
in_size = np.linspace(xMinLength, edgeLengthX + xMinLength, inLen)
out_size = np.linspace(yMinLength, edgeLengthY + yMinLength, outLen)
in_mesh = np.random.random((pow(inLen,2),2))
out_mesh = np.random.random((pow(outLen,2),2))
for i in range(0,inLen):
	for j in range(0,inLen):
		in_mesh[j+i*inLen,0] = (edgeLengthX/inLen)*j + xMinLength
		in_mesh[j+i*inLen,1] = (edgeLengthY/inLen)*i + yMinLength

for i in range(0,outLen):
	for j in range(0,outLen):
		out_mesh[j+i*outLen,0] = (edgeLengthX/outLen)*j + xMinLength
		out_mesh[j+i*outLen,1] = (edgeLengthY/outLen)*i + yMinLength

#mesh_size = 1/math.sqrt(nPoints)
mesh_size = edgeLengthX/inLen
shape_parameter = 4.55228/((4.0)*mesh_size)
bf = basisfunctions.Gaussian(shape_parameter)
func = lambda x,y: np.sin(5*x*y)+(0.0000001*y)
funcTan = lambda x,y: np.arctan(125*(pow(pow(x-1.5,2) + pow(y-0.25,2),0.5) - 0.92))
one_func = lambda x: np.ones_like(x)
in_vals = func(in_mesh[:,0],in_mesh[:,1])
out_vals = func(out_mesh[:,0],out_mesh[:,1])

start = time.time()
interpRational = Rational(bf, in_mesh, in_vals, rescale = False)
end = time.time()
print("Time for inversion: ", end-start)	
start = time.time()
fr = interpRational(in_vals, out_mesh)
end = time.time()
print("Time for eigen decomposition: ", end-start)



interp = NoneConsistent(bf, in_mesh, in_vals, rescale = False)
fr_regular = interp(out_mesh)


#out_vals = func(out_mesh[:,0], out_mesh[:,1])
print("out_vals: ", max(fr))
print("Error fr= ", np.linalg.norm(out_vals - fr, 2))
print("Error fr_regular= ", np.linalg.norm(out_vals - fr_regular, 2))
maxRegError = max(out_vals - fr_regular)
print("max fr: ", max(out_vals - fr))
print("max regular: ", maxRegError)



#plt.scatter(in_mesh[:,0], in_mesh[:,1], label = "In Mesh", s=2)
#plt.scatter(out_mesh[:,0], out_mesh[:,1], label = "Out Mesh", s=2)



# Make data.
#X = np.arange(-5, 5, 0.25)
X = np.linspace(xMinLength, edgeLengthX + xMinLength, outLen)
Y = np.linspace(yMinLength, edgeLengthY + yMinLength, outLen)
#Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
#R = np.sqrt(X**2 + Y**2)
#Z = np.sin(R)
Z = np.arctan(125*(pow(pow(X-1.5,2) + pow(Y-0.25,2),0.5) - 0.92))
Z_regular = np.arctan(125*(pow(pow(X-1.5,2) + pow(Y-0.25,2),0.5) - 0.92))
Z_regular_error = np.arctan(125*(pow(pow(X-1.5,2) + pow(Y-0.25,2),0.5) - 0.92))
Z_rational = np.arctan(125*(pow(pow(X-1.5,2) + pow(Y-0.25,2),0.5) - 0.92))
Z_rational_error = np.arctan(125*(pow(pow(X-1.5,2) + pow(Y-0.25,2),0.5) - 0.92))
#print(Z)
k=0
for i in range(0,outLen):
	for j in range(0,outLen):
		Z[i,j] = out_vals[k]
		Z_split[i,j+30] = out_vals[k]
		Z_rational[i,j] = fr[k]
		Z_rational_error[i,j] = out_vals[k]- fr[k]
		Z_regular[i,j] = fr_regular[k]
		Z_regular_error[i,j] = out_vals[k]- fr_regular[k]
		k += 1
'''
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('Actual')
ax.plot_surface(Xtotal, Ytotal, Z_split,cmap='viridis',linewidth=0,edgecolor='black')
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('Regular')
ax.plot_surface(X, Y, Z_regular,cmap='viridis',linewidth=0,edgecolor='black')
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('Rational')
ax.plot_surface(X, Y, Z_rational,cmap='viridis',linewidth=0)
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('Regular')
ax.plot_surface(X, Y, Z_regular_error,cmap='viridis',linewidth=0)
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('Rational')
ax.plot_surface(X, Y, Z_rational_error,cmap='viridis',linewidth=0)
plt.show()

'''
## ------------------------------------------------------------

inLen = 40
outLen = 30
edgeLengthX = 1
edgeLengthY = 1
xMinLength = 0.0
yMinLength = 1.0
in_size = np.linspace(xMinLength, edgeLengthX + xMinLength, inLen)
out_size = np.linspace(yMinLength, edgeLengthY + yMinLength, outLen)
in_mesh = np.random.random((pow(inLen,2),2))
out_mesh = np.random.random((pow(outLen,2),2))
for i in range(0,inLen):
	for j in range(0,inLen):
		in_mesh[j+i*inLen,0] = (edgeLengthX/inLen)*j + xMinLength
		in_mesh[j+i*inLen,1] = (edgeLengthY/inLen)*i + yMinLength

for i in range(0,outLen):
	for j in range(0,outLen):
		out_mesh[j+i*outLen,0] = (edgeLengthX/outLen)*j + xMinLength
		out_mesh[j+i*outLen,1] = (edgeLengthY/outLen)*i + yMinLength

#mesh_size = 1/math.sqrt(nPoints)
mesh_size = edgeLengthX/inLen
shape_parameter = 4.55228/((4.0)*mesh_size)
bf = basisfunctions.Gaussian(shape_parameter)
func = lambda x,y: np.sin(5*x*y)+(0.0000001*y)
funcTan = lambda x,y: np.arctan(125*(pow(pow(x-1.5,2) + pow(y-0.25,2),0.5) - 0.92))
one_func = lambda x: np.ones_like(x)
in_vals = func(in_mesh[:,0],in_mesh[:,1])
out_vals = func(out_mesh[:,0],out_mesh[:,1])

start = time.time()
interpRational = Rational(bf, in_mesh, in_vals, rescale = False)
end = time.time()
print("Time for inversion: ", end-start)	
start = time.time()
fr = interpRational(in_vals, out_mesh)
end = time.time()
print("Time for eigen decomposition: ", end-start)



interp = NoneConsistent(bf, in_mesh, in_vals, rescale = False)
fr_regular = interp(out_mesh)


#out_vals = funcTan(out_mesh[:,0], out_mesh[:,1])
print("out_vals: ", max(fr))
print("Error fr= ", np.linalg.norm(out_vals - fr, 2))
print("Error fr_regular= ", np.linalg.norm(out_vals - fr_regular, 2))
maxRegError = max(out_vals - fr_regular)
print("max fr: ", max(out_vals - fr))
print("max regular: ", maxRegError)



#plt.scatter(in_mesh[:,0], in_mesh[:,1], label = "In Mesh", s=2)
#plt.scatter(out_mesh[:,0], out_mesh[:,1], label = "Out Mesh", s=2)



# Make data.
#X = np.arange(-5, 5, 0.25)
X = np.linspace(xMinLength, edgeLengthX + xMinLength, outLen)
Y = np.linspace(yMinLength, edgeLengthY + yMinLength, outLen)
#Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
#R = np.sqrt(X**2 + Y**2)
#Z = np.sin(R)
Z = np.arctan(125*(pow(pow(X-1.5,2) + pow(Y-0.25,2),0.5) - 0.92))
Z_regular = np.arctan(125*(pow(pow(X-1.5,2) + pow(Y-0.25,2),0.5) - 0.92))
Z_regular_error = np.arctan(125*(pow(pow(X-1.5,2) + pow(Y-0.25,2),0.5) - 0.92))
Z_rational = np.arctan(125*(pow(pow(X-1.5,2) + pow(Y-0.25,2),0.5) - 0.92))
Z_rational_error = np.arctan(125*(pow(pow(X-1.5,2) + pow(Y-0.25,2),0.5) - 0.92))
#print(Z)
k=0
for i in range(0,outLen):
	for j in range(0,outLen):
		Z[i,j] = out_vals[k]
		Z_split[i+30,j] = out_vals[k]
		Z_rational[i,j] = fr[k]
		Z_rational_error[i,j] = out_vals[k]- fr[k]
		Z_regular[i,j] = fr_regular[k]
		Z_regular_error[i,j] = out_vals[k]- fr_regular[k]
		k += 1
'''
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('Actual')
ax.plot_surface(Xtotal, Ytotal, Z_split,cmap='viridis',linewidth=0,edgecolor='black')
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('Regular')
ax.plot_surface(X, Y, Z_regular,cmap='viridis',linewidth=0,edgecolor='black')
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('Rational')
ax.plot_surface(X, Y, Z_rational,cmap='viridis',linewidth=0)
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('Regular')
ax.plot_surface(X, Y, Z_regular_error,cmap='viridis',linewidth=0)
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('Rational')
ax.plot_surface(X, Y, Z_rational_error,cmap='viridis',linewidth=0)
plt.show()

'''
## ------------------------------------------------------------

inLen = 40
outLen = 30
edgeLengthX = 1
edgeLengthY = 1
xMinLength = 1.0
yMinLength = 1.0
in_size = np.linspace(xMinLength, edgeLengthX + xMinLength, inLen)
out_size = np.linspace(yMinLength, edgeLengthY + yMinLength, outLen)
in_mesh = np.random.random((pow(inLen,2),2))
out_mesh = np.random.random((pow(outLen,2),2))
for i in range(0,inLen):
	for j in range(0,inLen):
		in_mesh[j+i*inLen,0] = (edgeLengthX/inLen)*j + xMinLength
		in_mesh[j+i*inLen,1] = (edgeLengthY/inLen)*i + yMinLength

for i in range(0,outLen):
	for j in range(0,outLen):
		out_mesh[j+i*outLen,0] = (edgeLengthX/outLen)*j + xMinLength
		out_mesh[j+i*outLen,1] = (edgeLengthY/outLen)*i + yMinLength

#mesh_size = 1/math.sqrt(nPoints)
mesh_size = edgeLengthX/inLen
shape_parameter = 4.55228/((4.0)*mesh_size)
bf = basisfunctions.Gaussian(shape_parameter)
func = lambda x,y: np.sin(5*x*y)+(0.0000001*y)
funcTan = lambda x,y: np.arctan(125*(pow(pow(x-1.5,2) + pow(y-0.25,2),0.5) - 0.92))
one_func = lambda x: np.ones_like(x)
in_vals = func(in_mesh[:,0],in_mesh[:,1])
out_vals = func(out_mesh[:,0],out_mesh[:,1])

start = time.time()
interpRational = Rational(bf, in_mesh, in_vals, rescale = False)
end = time.time()
print("Time for inversion: ", end-start)	
start = time.time()
fr = interpRational(in_vals, out_mesh)
end = time.time()
print("Time for eigen decomposition: ", end-start)



interp = NoneConsistent(bf, in_mesh, in_vals, rescale = False)
fr_regular = interp(out_mesh)


#out_vals = funcTan(out_mesh[:,0], out_mesh[:,1])
print("out_vals: ", max(fr))
print("Error fr= ", np.linalg.norm(out_vals - fr, 2))
print("Error fr_regular= ", np.linalg.norm(out_vals - fr_regular, 2))
maxRegError = max(out_vals - fr_regular)
print("max fr: ", max(out_vals - fr))
print("max regular: ", maxRegError)



#plt.scatter(in_mesh[:,0], in_mesh[:,1], label = "In Mesh", s=2)
#plt.scatter(out_mesh[:,0], out_mesh[:,1], label = "Out Mesh", s=2)



# Make data.
#X = np.arange(-5, 5, 0.25)
X = np.linspace(xMinLength, edgeLengthX + xMinLength, outLen)
Y = np.linspace(yMinLength, edgeLengthY + yMinLength, outLen)
#Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
#R = np.sqrt(X**2 + Y**2)
#Z = np.sin(R)
Z = np.arctan(125*(pow(pow(X-1.5,2) + pow(Y-0.25,2),0.5) - 0.92))
Z_regular = np.arctan(125*(pow(pow(X-1.5,2) + pow(Y-0.25,2),0.5) - 0.92))
Z_regular_error = np.arctan(125*(pow(pow(X-1.5,2) + pow(Y-0.25,2),0.5) - 0.92))
Z_rational = np.arctan(125*(pow(pow(X-1.5,2) + pow(Y-0.25,2),0.5) - 0.92))
Z_rational_error = np.arctan(125*(pow(pow(X-1.5,2) + pow(Y-0.25,2),0.5) - 0.92))
#print(Z)
k=0
for i in range(0,outLen):
	for j in range(0,outLen):
		Z[i,j] = out_vals[k]
		Z_split[i+30,j+30] = out_vals[k]
		Z_rational[i,j] = fr[k]
		Z_rational_error[i,j] = out_vals[k]- fr[k]
		Z_regular[i,j] = fr_regular[k]
		Z_regular_error[i,j] = out_vals[k]- fr_regular[k]
		k += 1

'''
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('Actual')
ax.plot_surface(Xtotal, Ytotal, Z_split,cmap='viridis',linewidth=0,edgecolor='black')
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('Regular')
ax.plot_surface(X, Y, Z_regular,cmap='viridis',linewidth=0,edgecolor='black')
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('Rational')
ax.plot_surface(X, Y, Z_rational,cmap='viridis',linewidth=0)
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('Regular')
ax.plot_surface(X, Y, Z_regular_error,cmap='viridis',linewidth=0)
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('Rational')
ax.plot_surface(X, Y, Z_rational_error,cmap='viridis',linewidth=0)
plt.show()
'''

Z_split_error = np.arctan(125*(pow(pow(Xtotal-1.5,2) + pow(Ytotal-0.25,2),0.5) - 0.92))
for i in range(0,60):
	for j in range(0,60):
		Z_split_error[i,j] = Z_combined[i,j] - Z_split[i,j]

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('Actual')
ax.plot_surface(Xtotal, Ytotal, Z_split_error,cmap='viridis',linewidth=0,edgecolor='black')
plt.show()




