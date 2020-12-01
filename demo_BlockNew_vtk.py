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
import matplotlib.tri as mtri
import mesh_io
#from mpi4py import MPI

'''
############################################################
IMPORTANT!
1. Eigenvalue decomposition does not work with 100x100 input matrix - too large for memory
2. Difficult to run global input mesh with 100x100 


############################################################

'''

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


input_mesh_name = "Mesh/Plate/l1Data.vtk"
input_mesh = read_mesh(input_mesh_name)
print("Number of input points: ", len(input_mesh.points))

output_mesh_name = "Mesh/Plate/l2Data.vtk"
output_mesh = read_mesh(output_mesh_name)
print("Number of output points: ", len(output_mesh.points))

start = time.time()
j = 0
#nPoints = len(mesh.points)
nPoints = 100
nPointsOut = 500
#print("Number of points: ",nPoints)

######################################################
######################################################
'''
Define the parameters of in and out meshes
'''
######################################################
######################################################

#inLenTotal = 60 #now xMesh	
#outLenTotal = 45 # now yMesh
xInMesh = 60
yInMesh = 60

print("Total number in input mesh vertices: ", xInMesh*yInMesh)

xOutMesh = 90
yOutMesh = 90

print("Total number in output mesh vertices: ", xOutMesh*yOutMesh)

xMin = -0.5
xMax = 0.5
yMin = -0.5
yMax = 0.5

TotalXLength = xMax - xMin
TotalYLength = yMax - yMin

alphaInX = TotalXLength/xInMesh
alphaInY = TotalYLength/yInMesh

alphaOutX = TotalXLength/xOutMesh
alphaOutY = TotalYLength/yOutMesh

InedgeLengthX = 3.0
InedgeLengthY = 3.0
OutedgeLengthX = 3.0
OutedgeLengthY = 3.0
InxMinLength = 0.0
InyMinLength = 0.0
OutxMinLength = 0.0
OutyMinLength = 0.0
#alpha = TotalXLength/inLenTotal

domainXLenghtMin = 0.0
domainXLenghtMax = 3.0
domainLength = domainXLenghtMax - domainXLenghtMin

######################################################
######################################################
'''
Define which problems to solve:

Global - Regular
Global - Rational
Local - Regular
Local - Rational
'''

regularGlobal = 1
rationalGlobal = 1
regularLocal = 1
rationalLocal = 1

######################################################
######################################################


######################################################
######################################################
'''
How many blocks in each direction to break problem into
'''
######################################################
######################################################

# Domain decomposition. Grid mesh size/domainDecompo must be integer value
xDomainDecomposition = 2
yDomainDecomposition = 2

xStep = TotalXLength/xDomainDecomposition
yStep = TotalYLength/yDomainDecomposition

xGridStepIn = xInMesh/xDomainDecomposition
yGridStepIn = yInMesh/yDomainDecomposition

xGridStepOut = xOutMesh/xDomainDecomposition
yGridStepOut = yOutMesh/yDomainDecomposition

#inLen = int(inLenTotal/domainDecomposition)
#outLen = int(outLenTotal/domainDecomposition)
#inLen = 20
#outLen = 30
#### Even numbers only!!!!
xBoundaryExtension = 1
yBoundaryExtension = 1

#edgeLengthX = InedgeLengthX/domainDecomposition
#edgeLengthY = InedgeLengthY/domainDecomposition
xMinLength = xMin
yMinLength = yMin

globalRegularL2Error = 0
globalRationalL2Error = 0


######################################################
######################################################
######################################################

xInMesh += 1
yInMesh += 1
xOutMesh += 1
yOutMesh += 1

#in_size = np.linspace(xMinLength, edgeLengthX + xMinLength, inLenTotal)
#out_size = np.linspace(yMinLength, edgeLengthY + yMinLength, outLenTotal)
in_mesh = np.random.random((len(input_mesh.points),2))
in_mesh_global = np.random.random((len(input_mesh.points),2))
out_mesh = np.random.random((len(output_mesh.points),2))
out_mesh_global = np.random.random((len(output_mesh.points),2))
out_mesh_Combined = np.random.random((len(output_mesh.points),2))
out_mesh_Split = np.random.random((len(output_mesh.points),2))
out_mesh_Combined_value = []
out_mesh_Split_value = []

for j in range(0,len(input_mesh.points)):
	in_mesh[j,0] = input_mesh.points[j][0]
	in_mesh[j,1] = input_mesh.points[j][1]
	in_mesh_global[j,0] = input_mesh.points[j][0]
	in_mesh_global[j,1] = input_mesh.points[j][1]

#print("Original inmesh length: ", jj)

for j in range(0,len(output_mesh.points)):
	#out_mesh[j+i*outLenTotal,0] = (OutedgeLengthX/outLenTotal)*j + OutxMinLength
	out_mesh[j,0] = output_mesh.points[j][0]
	out_mesh[j,1] = output_mesh.points[j][1]
	out_mesh_global[j,0] = output_mesh.points[j][0]
	out_mesh_global[j,1] = output_mesh.points[j][1]
	out_mesh_Combined[j,0] = output_mesh.points[j][0]
	out_mesh_Combined[j,1] = output_mesh.points[j][1]
	out_mesh_Split[j,0] = output_mesh.points[j][0]
	out_mesh_Split[j,1] = output_mesh.points[j][1]

#print("Original inmesh: ", in_mesh)
#print("Original outmesh: ", out_mesh)
#print("Original outmesh length: ", kk)


#mesh_size = 1/math.sqrt(nPoints)
mesh_size = 0.1
shape_parameter = 4.55228/((5.0)*mesh_size)
print("shape_parameter: ", shape_parameter)
bf = basisfunctions.Gaussian(shape_parameter)

##########################################################
##########################################################
'''
Functions to test
'''
func = lambda x,y: 0.5*np.sin(2*x*y)+(0.0000001*y)

## Complex sin function
lambda x,y: 0.5*np.sin(2*x*y)+(0.0000001*y)

## Rosenbrock function
lambda x,y: pow(1-x,2) + 100*pow(y-pow(x,2),2)

## Arctan function (STEP FUNCTION) 
lambda x,y: np.arctan(125*(pow(pow(x-1.5,2) + pow(y-0.25,2),0.5) - 0.92))

## Unit values
lambda x: np.ones_like(x)

stepFunction = 0	# Apply step function values if = 1

##########################################################
##########################################################

in_vals = func(in_mesh[:,0],in_mesh[:,1])
in_vals_global = func(in_mesh[:,0],in_mesh[:,1])
out_vals = func(out_mesh[:,0],out_mesh[:,1])
out_vals_global = func(out_mesh[:,0],out_mesh[:,1])

k = 0
if (stepFunction == 1):
	for j in range(0,yInMesh):
		for i in range(0,xInMesh):
			if (in_mesh[i+j*xInMesh,0] <= 0.5 and in_mesh[i+j*xInMesh,1] <= 0.5):
				in_vals[k] = 2
			if (in_mesh[i+j*xInMesh,0] <= 0.5 and in_mesh[i+j*xInMesh,1] > 0.5):
				in_vals[k] = 5
			if (in_mesh[i+j*xInMesh,0] > 0.5 and in_mesh[i+j*xInMesh,1] > 0.5):
				in_vals[k] = 7
			if (in_mesh[i+j*xInMesh,0] > 0.5 and in_mesh[i+j*xInMesh,1] <= 0.5):
				in_vals[k] = 9
			k += 1
	k = 0
	for j in range(0,yOutMesh):
		for i in range(0,xOutMesh):
			if (out_mesh[i+j*xOutMesh,0] <= 0.5 and out_mesh[i+j*xOutMesh,1] <= 0.5):
				out_vals[k] = 2
				out_vals_global[k] = 2
			if (out_mesh[i+j*xOutMesh,0] <= 0.5 and out_mesh[i+j*xOutMesh,1] > 0.5):
				out_vals[k] = 5
				out_vals_global[k] = 5
			if (out_mesh[i+j*xOutMesh,0] > 0.5 and out_mesh[i+j*xOutMesh,1] > 0.5):
				out_vals[k] = 7
				out_vals_global[k] = 7
			if (out_mesh[i+j*xOutMesh,0] > 0.5 and out_mesh[i+j*xOutMesh,1] <= 0.5):
				out_vals[k] = 9
				out_vals_global[k] = 9
			k += 1




out_vals_global_rational = 0*func(out_mesh[:,0],out_mesh[:,1])
out_vals_global_regular = 0*func(out_mesh[:,0],out_mesh[:,1])
out_vals_split_rational = 0*func(out_mesh[:,0],out_mesh[:,1])
out_vals_split_regular = 0*func(out_mesh[:,0],out_mesh[:,1])
out_vals_split_rational_error = 0*func(out_mesh[:,0],out_mesh[:,1])
out_vals_split_regular_error = 0*func(out_mesh[:,0],out_mesh[:,1])
out_vals_global_rational_error = 0*func(out_mesh[:,0],out_mesh[:,1])
out_vals_global_regular_error = 0*func(out_mesh[:,0],out_mesh[:,1])

if (rationalGlobal == 1):
	start = time.time()
	interpRational = Rational(bf, in_mesh, in_vals, rescale = False)
	end = time.time()
	print("Time for Global rational inversion: ", end-start)	
	start = time.time()
	fr = interpRational(in_vals, out_mesh)
	end = time.time()
	print("Time for Global eigen decomposition: ", end-start)
else:
	print("Not running the Global Rational RBF")
	fr = func(out_mesh[:,0],out_mesh[:,1])

if (regularGlobal == 1):
	start = time.time()
	interp = NoneConsistent(bf, in_mesh, in_vals, rescale = False)
	fr_regular = interp(out_mesh)
	end = time.time()
	print("Time for Global regular solve: ", end-start)
else:
	print("Not running the Global Regular RBF")
	fr_regular = func(out_mesh[:,0],out_mesh[:,1])

#out_vals = funcTan(out_mesh[:,0], out_mesh[:,1])
#print("out_vals: ", max(fr))
#print("Error fr= ", np.linalg.norm(out_vals - fr, 2))
#print("max fr: ", max(out_vals - fr))
#print("Error fr_regular= ", np.linalg.norm(out_vals - fr_regular, 2))
maxRegError = max(out_vals - fr_regular)
#print("max fr: ", max(out_vals - fr))
#print("max regular: ", maxRegError)
globalRegularL2Error = np.linalg.norm(out_vals - fr_regular, 2)
globalRationalL2Error = np.linalg.norm(out_vals - fr, 2)


k=0
for k in range(0,len(fr)):
		out_vals_global_rational[k] = fr[k]
		out_vals_global_regular[k] = fr_regular[k]


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

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_title('Rational - error')
#ax.set_zlim(-0.001, 0.001)
ax.plot_surface(Xtotal, Ytotal, Z_rational_error,cmap='viridis',linewidth=0)
plt.show()

'''
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



######################################################
######################################################
######################################################
'''
Begin local RBF
'''
print("#############################################")
print("Beginning local RBF")
print("#############################################")
######################################################
######################################################
######################################################
xGridStepInSet = xGridStepIn
yGridStepInSet = yGridStepIn

xGridStepOutSet = xGridStepOut
yGridStepOutSet = yGridStepOut

start = time.time()
domainCount= 0
for dd2 in range(0,yDomainDecomposition):
	for dd1 in range(0,xDomainDecomposition):
		if (dd1 == 0):
			shiftX = 0.0
		elif (dd1 == xDomainDecomposition-1):
			#shiftX = (dd1)*xStep - xBoundaryExtension*alphaInX
			shiftX = (dd1)*xStep - xBoundaryExtension*xStep
		else:
			#shiftX = (dd1)*xStep- (xBoundaryExtension/2)*alphaInX
			shiftX = (dd1)*xStep- (xBoundaryExtension/2)*xStep

		if (dd2 == 0):
			shiftY = 0.0
		elif (dd2 == yDomainDecomposition-1):
			#shiftY = (dd2)*yStep - yBoundaryExtension*alphaInY
			shiftY = (dd2)*yStep - yBoundaryExtension*yStep
		else:
			#shiftY = (dd2)*yStep - (yBoundaryExtension/2)*alphaInY
			shiftY = (dd2)*yStep - (yBoundaryExtension/2)*yStep

		xMinLength = xMin + shiftX
		xMaxLength = xMin + shiftX + (1+xBoundaryExtension)*xStep
		yMinLength = yMin + shiftY
		yMaxLength = yMin + shiftY + (1+xBoundaryExtension)*yStep
		xMinLengthOut = xMin + dd1*xStep
		xMaxLengthOut = xMin + dd1*xStep + xStep
		yMinLengthOut = yMin + dd2*yStep
		yMaxLengthOut = yMin + dd2*yStep + yStep

		#print("Properties: ",xMinLength,yMinLength,xMinLengthOut, yMinLengthOut,dd1,dd2)
		#print("Alpha X: ", alphaOutX, alphaOutY)
		print("Local Domain Number: ",domainCount + 1)
		domainCount += 1

		if (dd1 == xDomainDecomposition-1):
			xGridStepIn = xGridStepInSet + 1
			xGridStepOut = xGridStepOutSet + 1
		else:
			xGridStepIn = xGridStepInSet + 1
			xGridStepOut = xGridStepOutSet 

		if (dd2 == yDomainDecomposition-1):
			yGridStepIn = yGridStepInSet + 1
			yGridStepOut = yGridStepOutSet + 1
		else:
			yGridStepIn = yGridStepInSet + 1
			yGridStepOut = yGridStepOutSet 

		# To create boxes for mesh
		# xMinLength is the point at lowest X position, alphaInX*int(xGridStepIn+xBoundaryExtension) is the length of the block


		#in_size = np.linspace(xMinLength, alphaInX*(xGridStepIn+xBoundaryExtension+1), int(xGridStepIn+xBoundaryExtension))
		#print("in_size: ", in_size)
		#in_size = np.linspace(xMinLength, edgeLengthX + xMinLength, inLen)
		#out_size = np.linspace(yMinLength, alphaInY*(yGridStepIn+yBoundaryExtension+1), int(yGridStepIn+yBoundaryExtension))
		#print("out_size: ", out_size)
		#in_mesh = np.random.random((int(xGridStepIn+xBoundaryExtension)*int(yGridStepIn+yBoundaryExtension),2))
		#out_mesh = np.random.random((int(xGridStepOut)*int(yGridStepOut),2))
		in_mesh_list = []
		out_mesh_list = []

		#print("Local domain input vertices: ", int(yGridStepIn+yBoundaryExtension)*int(xGridStepIn+xBoundaryExtension))
		#for j in range(0,int(yGridStepIn+yBoundaryExtension)):
		#	for i in range(0,int(xGridStepIn+xBoundaryExtension)):
		#		in_mesh[i+j*int(xGridStepIn+xBoundaryExtension),0] = alphaInX*i + xMinLength
		#		in_mesh[i+j*int(xGridStepIn+xBoundaryExtension),1] = alphaInY*j + yMinLength
				#if i == 0:
				#print("in_mesh: ",in_mesh[i+j*int(xGridStepIn+xBoundaryExtension),0])
		#print("Local domain output vertices: ", int(yGridStepOut)*int(xGridStepOut))
		#for j in range(0,int(yGridStepOut)):
		#	for i in range(0,int(xGridStepOut)):
		#		out_mesh[i+j*int(xGridStepOut),0] = alphaOutX*i + xMinLengthOut
		#		out_mesh[i+j*int(xGridStepOut),1] = alphaOutY*j + yMinLengthOut
				#if i == 0:
				#	print("out_mesh: ",out_mesh[j+i*outLen,0])
		#print(len(out_mesh))
		inCount = 0
		for i in range(0,len(input_mesh.points)):
			if ((in_mesh_global[i,0] >= xMinLength) and (in_mesh_global[i,0] <= xMaxLength) and (in_mesh_global[i,1] >= yMinLength) and (in_mesh_global[i,1] <= yMaxLength)):
				in_mesh_list.append(i)
				inCount += 1


		in_mesh = np.random.random((inCount,2))
		for i in range(0,inCount):
			in_mesh[i,0] = in_mesh_global[in_mesh_list[i],0]
			in_mesh[i,1] = in_mesh_global[in_mesh_list[i],1]



		outCount = 0
		for i in range(0,len(output_mesh.points)):
			if ((out_mesh_global[i,0] >= xMinLengthOut) and (out_mesh_global[i,0] <= xMaxLengthOut) and (out_mesh_global[i,1] >= yMinLengthOut) and (out_mesh_global[i,1] <= yMaxLengthOut)):
				out_mesh_list.append(i)
				outCount += 1


		out_mesh = np.random.random((outCount,2))
		for i in range(0,outCount):
			out_mesh[i,0] = out_mesh_global[out_mesh_list[i],0]
			out_mesh[i,1] = out_mesh_global[out_mesh_list[i],1]
			

		in_vals = func(in_mesh[:,0],in_mesh[:,1])
		out_vals = func(out_mesh[:,0],out_mesh[:,1])

		k = 0
		
		
		if (rationalLocal == 1):
			print("Using local Rational RBFs")
			interpRational = Rational(bf, in_mesh, in_vals, rescale = False)	
			fr = interpRational(in_vals, out_mesh)
		else:
			print("NOT Using local Rational RBFs")
			fr = func(out_mesh[:,0],out_mesh[:,1])
		
		if (regularLocal == 1):
			print("Using local Regular RBFs")
			interp = NoneConsistent(bf, in_mesh, in_vals, rescale = False)
			fr_regular = interp(out_mesh)
		else:	
			print("NOT Using local Regular RBFs")	
			fr_regular = func(out_mesh[:,0],out_mesh[:,1])

		#out_vals = funcTan(out_mesh[:,0], out_mesh[:,1])
		#print("out_vals: ", max(fr))
		#print("Error fr= ", np.linalg.norm(out_vals - fr, 2))
		#print("Error fr_regular= ", np.linalg.norm(out_vals - fr_regular, 2))
		maxRegError = max(out_vals - fr_regular)
		#print("max fr: ", max(out_vals - fr))
		#print("max regular: ", maxRegError)


		for i in range(0,outCount):
			out_vals_split_rational[out_mesh_list[i]] =fr[k]
			out_vals_split_regular[out_mesh_list[i]] = fr_regular[k]
			k += 1
			

		#for j in range(0,int(yGridStepOut)):
		#	for i in range(0,int(xGridStepOut)):
				#Z[i,j] = out_vals[k]
				#Z_split[i+int(xGridStepOutSet*dd1),j+int(yGridStepOutSet*dd2)] = fr[k]
				###w = int((i+(dd1*xGridStepOutSet)) + ((j+(dd2*yGridStepOutSet))*(xOutMesh)))
				#print(w)
				###out_vals_split_rational[w] = fr[k]
				###out_vals_split_regular[w] = fr_regular[k]
				#Z_rational[i,j] = fr[k]
				#Z_rational_error[i,j] = out_vals[k]- fr[k]
				#Z_regular[i,j] = fr_regular[k]
				#Z_regular_error[i,j] = out_vals[k]- fr_regular[k]
		#		k += 1
			#print("j: ", j)

		'''
		fig = plt.figure()
		ax = fig.gca(projection='3d')
		ax.set_xlabel('x axis')
		ax.set_ylabel('y axis')
		ax.set_title('Split mesh - Rational')
		ax.plot_surface(Xtotal, Ytotal, Z_rational,cmap='viridis',linewidth=0,edgecolor='black')
		plt.show()

		fig = plt.figure()
		ax = fig.gca(projection='3d')
		ax.set_xlabel('x axis')
		ax.set_ylabel('y axis')
		ax.set_title('Split mesh - Regular')
		ax.plot_surface(Xtotal, Ytotal, Z_regular,cmap='viridis',linewidth=0,edgecolor='black')
		plt.show()
		'''

end = time.time()
print("Time for decomposed problem eigen decomposition: ", end-start)

#fig = plt.figure()
#ax = fig.gca(projection='3d')
#ax.set_xlabel('Regular')
#ax.plot_surface(Xtotal, Ytotal, Z_split,cmap='viridis',linewidth=0,edgecolor='black')
#plt.show()


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

global_local_rational_difference = []
global_local_regular_difference = []

k = 0
for j in range(0,len(output_mesh.points)):
		#Z_split_error[i,j] = Z_combined[i,j] - Z_split[i,j]
		#Z_rational_diff[i,j] = Z_rational_global[i,j] - Z_split[i,j]
		#Z_error_diff[i,j] = Z_rational_error_global[i,j] - Z_split_error[i,j]
		out_vals_split_rational_error[k] = out_vals_split_rational[k] - out_vals_global[k]
		out_vals_split_regular_error[k] = out_vals_split_regular[k] - out_vals_global[k]
		out_vals_global_rational_error[k] = out_vals_global_rational[k] - out_vals_global[k]
		out_vals_global_regular_error[k] = out_vals_global_regular[k] - out_vals_global[k]
		global_local_rational_difference.append(out_vals_split_rational[k] - out_vals_global_rational[k])
		global_local_regular_difference.append(out_vals_split_regular[k] - out_vals_global_regular[k])
		k += 1 

print("Error of Global Rational RBF: ", np.linalg.norm(out_vals_global_rational_error, 2))
print("Error of Local Rational RBF sub-domains: ", np.linalg.norm(out_vals_split_rational_error, 2))
print("Max Global Rational RBF Error: ", max(abs(out_vals_global_rational_error)))
print("Max Local Rational RBF Error: ", max(abs(out_vals_split_rational_error)))

print("Error of Global Regular RBF: ", np.linalg.norm(out_vals_global_regular_error, 2))
print("Error of Local Regular RBF sub-domains: ", np.linalg.norm(out_vals_split_regular_error, 2))
print("Max Global Regular RBF Error: ", max(abs(out_vals_global_regular_error)))
print("Max Local Regular RBF Error: ", max(abs(out_vals_split_regular_error)))



#fig = plt.figure()
#ax = fig.add_subplot(1,1,1)

#ax.triplot(triang, c="#D3D3D3", marker='.', markerfacecolor="#DC143C",markeredgecolor="black", markersize=10)

#ax.set_xlabel('X')
#ax.set_ylabel('Y')
#plt.show()

triang = mtri.Triangulation(out_mesh_global[:,0], out_mesh_global[:,1])
fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')

ax.plot_trisurf(triang, out_vals_global, cmap='jet')
ax.scatter(out_mesh_global[:,0], out_mesh_global[:,1],out_vals_global, marker='.', s=10, c="black", alpha=0.5)
ax.view_init(elev=60, azim=-45)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

triang = mtri.Triangulation(in_mesh_global[:,0], in_mesh_global[:,1])
fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')

ax.plot_trisurf(triang, in_vals_global, cmap='jet')
ax.scatter(in_mesh_global[:,0], in_mesh_global[:,1],in_vals_global, marker='.', s=10, c="black", alpha=0.5)
ax.view_init(elev=60, azim=-45)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

triang = mtri.Triangulation(out_mesh_global[:,0], out_mesh_global[:,1])
fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')

ax.plot_trisurf(triang, out_vals_global_regular, cmap='jet')
ax.scatter(out_mesh_global[:,0], out_mesh_global[:,1],out_vals_global_regular, marker='.', s=10, c="black", alpha=0.5)
ax.view_init(elev=60, azim=-45)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Regular RBF on Global Grid')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')

ax.plot_trisurf(triang, out_vals_global_rational, cmap='jet')
ax.scatter(out_mesh_global[:,0], out_mesh_global[:,1],out_vals_global_rational, marker='.', s=10, c="black", alpha=0.5)
ax.view_init(elev=60, azim=-45)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Rational RBF on Global Grid')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')

ax.plot_trisurf(triang, out_vals_split_regular, cmap='jet')
ax.scatter(out_mesh_global[:,0], out_mesh_global[:,1],out_vals_split_regular, marker='.', s=10, c="black", alpha=0.5)
ax.view_init(elev=60, azim=-45)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Regular RBF on Local Grids')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')

ax.plot_trisurf(triang, out_vals_split_rational, cmap='jet')
ax.scatter(out_mesh_global[:,0], out_mesh_global[:,1],out_vals_split_rational, marker='.', s=10, c="black", alpha=0.5)
ax.view_init(elev=60, azim=-45)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Rational RBF on Local Grids')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')
#(ax1, ax2) = plt.subplots(1, 2, projection='3d')

ax.plot_trisurf(triang, out_vals_split_regular_error, cmap='jet')
ax.scatter(out_mesh_global[:,0], out_mesh_global[:,1],out_vals_split_regular_error, marker='.', s=10, c="black", alpha=0.5)
ax.view_init(elev=60, azim=-45)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Error - Regular RBF on Local Grids')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')

ax.plot_trisurf(triang, out_vals_global_regular_error, cmap='jet')
ax.scatter(out_mesh_global[:,0], out_mesh_global[:,1],out_vals_global_regular_error, marker='.', s=10, c="black", alpha=0.5)
ax.view_init(elev=60, azim=-45)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Error - Regular RBF on Global Grid')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')

ax.plot_trisurf(triang, out_vals_split_rational_error, cmap='jet')
ax.scatter(out_mesh_global[:,0], out_mesh_global[:,1],out_vals_split_rational_error, marker='.', s=10, c="black", alpha=0.5)
ax.view_init(elev=60, azim=-45)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Error - Rational RBF on Local Grids')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')

ax.plot_trisurf(triang, out_vals_global_rational_error, cmap='jet')
ax.scatter(out_mesh_global[:,0], out_mesh_global[:,1],out_vals_global_rational_error, marker='.', s=10, c="black", alpha=0.5)
ax.view_init(elev=60, azim=-45)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Error - Rational RBF on Global Grid')
plt.show()
