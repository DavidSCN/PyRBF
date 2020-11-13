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
#from mpi4py import MPI

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

#print("Hi, I'm process: ", rank)
#print(MPI.COMM_WORLD.rank)
#if (MPI.COMM_WORLD.rank == 0):
#	print("This is the master rank")

#Xtest = np.outer(np.linspace(-2, 2, 100), np.ones(100))
#Ytest = Xtest.copy().T # transpose
#Xtest, Ytest = np.meshgrid(Xtest, Ytest)
#Ztest = np.cos(Xtest ** 2 + Ytest ** 2)
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#ax.set_title('Test grid')
#ax.plot_surface(Xtest, Ytest, Ztest,cmap='viridis',linewidth=0,edgecolor='black')
#plt.show()


mesh_name = "Mesh/Plate/l1Data.vtk"
mesh = read_mesh(mesh_name)
#print("Number of points: ", mesh.points)

start = time.time()
j = 0
#nPoints = len(mesh.points)
nPoints = 100
nPointsOut = 500
#print("Number of points: ",nPoints)
inLenTotal = 60
outLenTotal = 45
InedgeLengthX = 3.0
InedgeLengthY = 3.0
OutedgeLengthX = 3.0
OutedgeLengthY = 3.0
InxMinLength = 0.0
InyMinLength = 0.0
OutxMinLength = 0.0
OutyMinLength = 0.0
alpha = InedgeLengthX/inLenTotal

domainXLenghtMin = 0.0
domainXLenghtMax = 3.0
domainLength = domainXLenghtMax - domainXLenghtMin
#in_size = np.linspace(xMinLength, edgeLengthX + xMinLength, inLenTotal)
#out_size = np.linspace(yMinLength, edgeLengthY + yMinLength, outLenTotal)
in_mesh = np.random.random((pow(inLenTotal,2),2))
out_mesh = np.random.random((pow(outLenTotal,2),2))
out_mesh_Combined = np.random.random((pow(outLenTotal,2),2))
out_mesh_Split = np.random.random((pow(outLenTotal,2),2))
out_mesh_Combined_value = []
out_mesh_Split_value = []

for i in range(0,inLenTotal):
	for j in range(0,inLenTotal):
		in_mesh[j+i*inLenTotal,0] = (InedgeLengthX/inLenTotal)*j 
		in_mesh[j+i*inLenTotal,1] = (InedgeLengthY/inLenTotal)*i

for i in range(0,outLenTotal):
	for j in range(0,outLenTotal):
		out_mesh[j+i*outLenTotal,0] = (OutedgeLengthX/outLenTotal)*j + OutxMinLength
		out_mesh[j+i*outLenTotal,1] = (OutedgeLengthY/outLenTotal)*i + OutyMinLength
		out_mesh_Combined[j+i*outLenTotal,0] = (OutedgeLengthX/outLenTotal)*j + OutxMinLength
		out_mesh_Combined[j+i*outLenTotal,1] = (OutedgeLengthY/outLenTotal)*i + OutyMinLength
		out_mesh_Split[j+i*outLenTotal,0] = (OutedgeLengthX/outLenTotal)*j + OutxMinLength
		out_mesh_Split[j+i*outLenTotal,1] = (OutedgeLengthY/outLenTotal)*i + OutyMinLength 

#mesh_size = 1/math.sqrt(nPoints)
mesh_size = InedgeLengthX/inLenTotal
shape_parameter = 4.55228/((3.0)*mesh_size)
print("shape_parameter: ", shape_parameter)
bf = basisfunctions.Gaussian(shape_parameter)
func = lambda x,y: 0.5*np.sin(0.2*x*y)+(0.0000001*y)
funcTan = lambda x,y: np.arctan(125*(pow(pow(x-1.5,2) + pow(y-0.25,2),0.5) - 0.92))
one_func = lambda x: np.ones_like(x)
#rosenbrock_func = lambda x,y: pow(1-((x-0.5)*4),2) + (100*pow(((y-0.5)*4)-pow((x-0.5)*4,2),2))
rosenbrock_func = lambda x,y: pow(1-x,2) + 100*pow(y-pow(x,2),2)
in_vals = func(in_mesh[:,0],in_mesh[:,1])
out_vals = func(out_mesh[:,0],out_mesh[:,1])

start = time.time()
interpRational = Rational(bf, in_mesh, in_vals, rescale = False)
end = time.time()
print("Time for inversion: ", end-start)	
start = time.time()
fr = interpRational(in_vals, out_mesh)
#fr = func(out_mesh[:,0],out_mesh[:,1])
end = time.time()
print("Time for eigen decomposition: ", end-start)



#interp = NoneConsistent(bf, in_mesh, in_vals, rescale = False)
#fr_regular = interp(out_mesh)
fr_regular = func(out_mesh[:,0],out_mesh[:,1])

#out_vals = funcTan(out_mesh[:,0], out_mesh[:,1])
#print("out_vals: ", max(fr))
print("Error fr= ", np.linalg.norm(out_vals - fr, 2))
print("max fr: ", max(out_vals - fr))
#print("Error fr_regular= ", np.linalg.norm(out_vals - fr_regular, 2))
#maxRegError = max(out_vals - fr_regular)
#print("max fr: ", max(out_vals - fr))
#print("max regular: ", maxRegError)


Xtotal = np.linspace(OutxMinLength, OutedgeLengthX + OutxMinLength, outLenTotal)
Ytotal = np.linspace(OutyMinLength, OutedgeLengthY + OutyMinLength, outLenTotal)
#Y = np.arange(-5, 5, 0.25)
Xtotal, Ytotal = np.meshgrid(Xtotal, Ytotal)
#R = np.sqrt(X**2 + Y**2)
#Z = np.sin(R)
X = np.linspace(InxMinLength, InedgeLengthX + InxMinLength, inLenTotal)
Y = np.linspace(InyMinLength, InedgeLengthY + InyMinLength, inLenTotal)
X, Y = np.meshgrid(X, Y)

Zin = np.arctan(125*(pow(pow(X-1.5,2) + pow(Y-0.25,2),0.5) - 0.92))
Z = np.arctan(125*(pow(pow(Xtotal-1.5,2) + pow(Ytotal-0.25,2),0.5) - 0.92))
Z_combined = np.arctan(125*(pow(pow(Xtotal-1.5,2) + pow(Ytotal-0.25,2),0.5) - 0.92))
Z_split = np.arctan(125*(pow(pow(Xtotal-1.5,2) + pow(Ytotal-0.25,2),0.5) - 0.92))
Z_regular = np.arctan(125*(pow(pow(Xtotal-1.5,2) + pow(Ytotal-0.25,2),0.5) - 0.92))
Z_regular_error = np.arctan(125*(pow(pow(Xtotal-1.5,2) + pow(Ytotal-0.25,2),0.5) - 0.92))
Z_rational = np.arctan(125*(pow(pow(Xtotal-1.5,2) + pow(Ytotal-0.25,2),0.5) - 0.92))
Z_rational_global = np.arctan(125*(pow(pow(Xtotal-1.5,2) + pow(Ytotal-0.25,2),0.5) - 0.92))
Z_rational_error = np.arctan(125*(pow(pow(Xtotal-1.5,2) + pow(Ytotal-0.25,2),0.5) - 0.92))
Z_rational_error_final = np.arctan(125*(pow(pow(Xtotal-1.5,2) + pow(Ytotal-0.25,2),0.5) - 0.92))
Z_regular_error_global = np.arctan(125*(pow(pow(Xtotal-1.5,2) + pow(Ytotal-0.25,2),0.5) - 0.92))
Z_rational_error_global = np.arctan(125*(pow(pow(Xtotal-1.5,2) + pow(Ytotal-0.25,2),0.5) - 0.92))
#print(Z)
k=0
for i in range(0,inLenTotal):
	for j in range(0,inLenTotal):
		Zin[i,j] = in_vals[k]
		k += 1
k=0
for i in range(0,outLenTotal):
	for j in range(0,outLenTotal):
		Z[i,j] = out_vals[k]
		Z_combined[i,j] = out_vals[k]
		Z_split[i,j] = 0
		Z_rational[i,j] = fr[k]
		Z_rational_global[i,j] = fr[k]
		Z_rational_error[i,j] = out_vals[k]- fr[k]
		Z_rational_error_global[i,j] = out_vals[k] - fr[k]
		Z_regular[i,j] = fr_regular[k]
		Z_regular_error[i,j] = out_vals[k] - fr_regular[k]
		Z_regular_error_global[i,j] = out_vals[k]- fr_regular[k]
		k += 1


fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_title('In Grid')
ax.plot_surface(X, Y, Zin,cmap='viridis',linewidth=0,edgecolor='black')
plt.show()

#save_plot(fileName='plot_01.py',obj=sys.argv[0],sel='plot',ctx=libscript.get_ctx(ctx_global=globals(),ctx_local=locals()))


fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_title('Actual - out Grid')
ax.plot_surface(Xtotal, Ytotal, Z_combined,cmap='viridis',linewidth=0,edgecolor='black')
plt.show()

'''

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_title('Regular')
ax.plot_surface(Xtotal, Ytotal, Z_regular,cmap='viridis',linewidth=0,edgecolor='black')
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_title('Rational')
ax.plot_surface(Xtotal, Ytotal, Z_rational,cmap='viridis',linewidth=0)
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_title('Regular - error')
ax.plot_surface(Xtotal, Ytotal, Z_regular_error,cmap='viridis',linewidth=0)
plt.show()
'''
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_title('Rational - error')
#ax.set_zlim(-0.001, 0.001)
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

'''
How many blocks in each direction to break problem into
'''

domainDecomposition = 3

inLen = int(inLenTotal/domainDecomposition)
outLen = int(outLenTotal/domainDecomposition)
#inLen = 20
#outLen = 30
boundaryExtension = 10
edgeLengthX = InedgeLengthX/domainDecomposition
edgeLengthY = InedgeLengthY/domainDecomposition
xMinLength = 0.0
yMinLength = 0.0
if domainDecomposition > 1:
	#shift = (InedgeLengthX - (edgeLengthX + (edgeLengthX*boundaryExtension/inLen)))/(domainDecomposition - 1)
	shift = edgeLengthX*boundaryExtension/(2*inLen)
else:
	shift = 1

print("Shift value:", InedgeLengthX,edgeLengthX + (edgeLengthX*boundaryExtension/inLen), shift)

start = time.time()
domainCount= 0
for dd1 in range(0,domainDecomposition):
	for dd2 in range(0,domainDecomposition):
		if (dd1 == 0):
			shiftX = 0.0
		elif (dd1 == domainDecomposition-1):
			shiftX = (dd1)*edgeLengthX - boundaryExtension*alpha
		else:
			shiftX = (dd1)*(edgeLengthX) - (boundaryExtension/2)*alpha

		if (dd2 == 0):
			shiftY = 0.0
		elif (dd2 == domainDecomposition-1):
			shiftY = (dd2)*edgeLengthX - boundaryExtension*alpha
		else:
			shiftY = (dd2)*(edgeLengthX) - (boundaryExtension/2)*alpha

		xMinLength = 0.0 + shiftX
		yMinLength = 0.0 + shiftY
		xMinLengthOut = 0.0 + dd1*edgeLengthY
		yMinLengthOut = 0.0 + dd2*edgeLengthY

		print("Properties: ",inLen, outLen,xMinLength,yMinLength,xMinLengthOut, yMinLengthOut,dd1,dd2)
		print("domain count: ",domainCount)
		domainCount += 1

		in_size = np.linspace(xMinLength, edgeLengthX + xMinLength, inLen+boundaryExtension)
		#in_size = np.linspace(xMinLength, edgeLengthX + xMinLength, inLen)
		out_size = np.linspace(yMinLength, edgeLengthY + yMinLength, outLen)
		in_mesh = np.random.random((pow(inLen+boundaryExtension,2),2))
		out_mesh = np.random.random((pow(outLen,2),2))
		for i in range(0,inLen+boundaryExtension):
			for j in range(0,inLen+boundaryExtension):
				#in_mesh[j+i*(inLen),0] = (edgeLengthX/inLen)*j + xMinLength
				#in_mesh[j+i*(inLen),1] = (edgeLengthX/inLen)*i + yMinLength
				in_mesh[j+i*(inLen+boundaryExtension),0] = (edgeLengthX/inLen)*j + xMinLength
				in_mesh[j+i*(inLen+boundaryExtension),1] = (edgeLengthX/inLen)*i + yMinLength
				#if i == 0:
				#	print("in_mesh: ",in_mesh[j+i*(inLen+boundaryExtension),0])

		for i in range(0,outLen):
			for j in range(0,outLen):
				out_mesh[j+i*outLen,0] = (edgeLengthY/outLen)*j + xMinLengthOut
				out_mesh[j+i*outLen,1] = (edgeLengthY/outLen)*i + yMinLengthOut
				#if i == 0:
				#	print("out_mesh: ",out_mesh[j+i*outLen,0])

		#mesh_size = 1/math.sqrt(nPoints)
		#mesh_size = edgeLengthX/inLen
		#shape_parameter = 4.55228/((4.0)*mesh_size)
		print("Min in: ", in_mesh[0,0], in_mesh[0,1])
		print("Max in: ", in_mesh[(inLen+boundaryExtension)*(inLen+boundaryExtension)-1,0], in_mesh[(inLen+boundaryExtension)*(inLen+boundaryExtension)-1,1])
		print("Min in: ", out_mesh[0,0], out_mesh[0,1])
		print("Max in: ", out_mesh[outLen*outLen-1,0], out_mesh[outLen*outLen-1,1])
		print("shape_parameter: ", shape_parameter)
		bf = basisfunctions.Gaussian(shape_parameter)
		func = lambda x,y: 0.5*np.sin(0.2*x*y)+(0.0000001*y)
		funcTan = lambda x,y: np.arctan(125*(pow(pow(x-1.5,2) + pow(y-0.25,2),0.5) - 0.92))
		#rosenbrock_func = lambda x,y: pow(1-((x-0.5)*4),2) + (100*pow(((y-0.5)*4)-pow((x-0.5)*4,2),2)) 
		rosenbrock_func = lambda x,y: pow(1-x,2) + (100*pow(y-pow(x,2),2)) 
		one_func = lambda x: np.ones_like(x)
		in_vals = func(in_mesh[:,0],in_mesh[:,1])
		out_vals = func(out_mesh[:,0],out_mesh[:,1])
		
		interpRational = Rational(bf, in_mesh, in_vals, rescale = False)	
		fr = interpRational(in_vals, out_mesh)
		#fr = func(out_mesh[:,0],out_mesh[:,1])
		
		interp = NoneConsistent(bf, in_mesh, in_vals, rescale = False)
		fr_regular = interp(out_mesh)
		#fr_regular = func(out_mesh[:,0],out_mesh[:,1])

		#out_vals = funcTan(out_mesh[:,0], out_mesh[:,1])
		print("out_vals: ", max(fr))
		print("Error fr= ", np.linalg.norm(out_vals - fr, 2))
		print("Error fr_regular= ", np.linalg.norm(out_vals - fr_regular, 2))
		maxRegError = max(out_vals - fr_regular)
		print("max fr: ", max(out_vals - fr))
		print("max regular: ", maxRegError)

		X = np.linspace(xMinLength, edgeLengthX + xMinLength, outLen)
		Y = np.linspace(yMinLength, edgeLengthY + yMinLength, outLen)

		X, Y = np.meshgrid(X, Y)

		Z = np.arctan(125*(pow(pow(X-1.5,2) + pow(Y-0.25,2),0.5) - 0.92))
		Z_regular = np.arctan(125*(pow(pow(X-1.5,2) + pow(Y-0.25,2),0.5) - 0.92))
		Z_regular_error = np.arctan(125*(pow(pow(X-1.5,2) + pow(Y-0.25,2),0.5) - 0.92))
		Z_rational = np.arctan(125*(pow(pow(X-1.5,2) + pow(Y-0.25,2),0.5) - 0.92))
		Z_rational_error = np.arctan(125*(pow(pow(X-1.5,2) + pow(Y-0.25,2),0.5) - 0.92))

		k=0
		for i in range(0,outLen):
			for j in range(0,outLen):
				Z[i,j] = out_vals[k]
				Z_split[i+(outLen*dd2),j+(outLen*dd1)] = fr[k]
				Z_rational[i,j] = fr[k]
				Z_rational_error[i,j] = out_vals[k]- fr[k]
				Z_regular[i,j] = fr_regular[k]
				Z_regular_error[i,j] = out_vals[k]- fr_regular[k]
				k += 1

		#fig = plt.figure()
		#ax = fig.gca(projection='3d')
		#ax.set_xlabel('Actual')
		#ax.plot_surface(Xtotal, Ytotal, Z_split,cmap='viridis',linewidth=0,edgecolor='black')
		#plt.show()

end = time.time()
print("Time for decomposed problem eigen decomposition: ", end-start)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('Regular')
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
'''
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('Regular Error')
ax.plot_surface(X, Y, Z_regular_error,cmap='viridis',linewidth=0)
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('Rational Error')
ax.plot_surface(X, Y, Z_rational_error,cmap='viridis',linewidth=0)
plt.show()
'''
Z_split_error = np.arctan(125*(pow(pow(Xtotal-1.5,2) + pow(Ytotal-0.25,2),0.5) - 0.92))
Z_error_diff = np.arctan(125*(pow(pow(Xtotal-1.5,2) + pow(Ytotal-0.25,2),0.5) - 0.92))
Z_rational_diff = np.arctan(125*(pow(pow(Xtotal-1.5,2) + pow(Ytotal-0.25,2),0.5) - 0.92))
max_global_rational_error = []
for i in range(0,outLenTotal):
	for j in range(0,outLenTotal):
		Z_split_error[i,j] = Z_combined[i,j] - Z_split[i,j]
		Z_rational_diff[i,j] = Z_rational_global[i,j] - Z_split[i,j]
		#if (Z_split_error[i,j] > 0.004):
		#	Z_split_error[i,j] = 0.004
		#if (Z_split_error[i,j] < -0.004):
		#	Z_split_error[i,j] = -0.004
		Z_error_diff[i,j] = Z_rational_error_global[i,j] - Z_split_error[i,j]
		max_global_rational_error.append(Z_rational_error_global[i,j])

print("Error of Global rational RBF: ", np.linalg.norm(Z_rational_error_global, 2))
print("Error of Rational RBF sub-domains combined: ", np.linalg.norm(Z_split_error, 2))
print("Max Global: ", max(max_global_rational_error))
#print("Max Local: ", max(Z_split_error))

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_title('Rational split mesh error when combined onto full grid')
#ax.set_zlim(-0.00025, 0.00025)
ax.plot_surface(Xtotal, Ytotal, Z_split_error,cmap='viridis',linewidth=0,edgecolor='black')
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_title('Rational RBF Global - Local')
#ax.set_zlim(-0.00025, 0.00025)
ax.plot_surface(Xtotal, Ytotal, Z_rational_diff,cmap='viridis',linewidth=0,edgecolor='black')
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_title('Difference between the Global vs Local Rational RBF error magnitudes')
#ax.set_zlim(-0.00025, 0.00025)
ax.plot_surface(Xtotal, Ytotal, Z_error_diff,cmap='viridis',linewidth=0,edgecolor='black')
plt.show()




