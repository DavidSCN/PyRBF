""" Generates data to show the effect of rescaling. Low density basisfunctions used. """

import pandas
import os
import logging
from rbf import *
import basisfunctions, testfunctions
import matplotlib.pyplot as plt
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

#mesh_name = "Mesh/Plate/l1Data.vtk"
#mesh = read_mesh(mesh_name)
#print("Points: ", mesh.points)

dimension_M = 2			# dimension of problem
nPoints = pow(25,2)
#nPoints = len(mesh.points)	# number of points
nPatches = nPoints / pow(2,dimension_M) # number of subdomains

print("n patches: ", nPatches)

start = time.time()
j = 0

nPointsOut = 1
print("Number of points: ",nPoints)
in_mesh = np.random.random((nPoints,2))

haltonPoints = halton_sequence(nPoints, 2)
for i in range(0,nPoints):
	#in_mesh[i,0] = mesh.points[i][0]
	#in_mesh[i,1] = mesh.points[i][1]
	in_mesh[i,0] = haltonPoints[0][i]
	in_mesh[i,1] = haltonPoints[1][i]

# Find Lbox 
x_min = min(in_mesh[:,0])
x_max = max(in_mesh[:,0])
y_min = min(in_mesh[:,1])
y_max = max(in_mesh[:,1])
l_min = min(x_min,y_min)
l_max = max(x_max,y_max)

print("x min: ", x_min)
print("x max: ", x_max)
print("y min: ", y_min)
print("y max: ", y_max)

hyperVolume = (x_max-x_min)*(y_max-y_min)
print("Hypervolume: ", hyperVolume)
# Number of patches per dimension
nPatchCentres = math.floor(0.5*(l_max - l_min)*pow(nPoints/(6*hyperVolume),1/dimension_M))
#nPatchCentres = math.floor(pow(nPoints/2,1/dimension_M))
print("n Patch Centres: ", nPatchCentres)
#Radius of the patch (half the diameter)
patchRadii = (l_max - l_min)/nPatchCentres	
print("n Patch Radii: ", patchRadii)

puCentres = np.random.random((pow(nPatchCentres,dimension_M),2))
j = 0
k = 0
for i in range(0,pow(nPatchCentres,dimension_M)):
	if (j == nPatchCentres):
		j = 0
		k += 1
	puCentres[i,0] = j*1.1*patchRadii + 0.1*patchRadii
	puCentres[i,1] = k*1.1*patchRadii + 0.1*patchRadii
	print("PU coords: ", puCentres[i,0], puCentres[i,1])
	j += 1

# find q^M blocks
q = math.ceil((l_max - l_min)/patchRadii)
print("q: ", q)

### Find the block where the PU domain centre lies in. Similar to linked cell method
km = 0
kthBlock = 0
puKBlocks = []
for i in range(0,pow(nPatchCentres,2)):
	kthBlock = 0
	for k in range(0,dimension_M-1):
		km = math.ceil(puCentres[i,k]/patchRadii)
		if (km < 1):
			km = 1
		kthBlock += (km - 1)*pow(q,dimension_M-k-1)
	kthBlock += math.ceil(puCentres[i,dimension_M-1]/patchRadii)
	#if (kthBlock < 1):
	#	kthBlock += q
	print("kthBlock: ", kthBlock)
	puKBlocks.append(kthBlock)
	
### Repeat the method for all in_mesh points
km = 0
kthBlock = 0
in_mesh_KBlocks = []
for i in range(0,nPoints):
	kthBlock = 0
	for k in range(0,dimension_M-1):
		km = math.ceil(in_mesh[i,k]/patchRadii)
		kthBlock += (km - 1)*pow(q,dimension_M-k-1)
	kthBlock += math.ceil(in_mesh[i,dimension_M-1]/patchRadii)
	in_mesh_KBlocks.append(kthBlock)
print("in_mesh_KBlocks: ", in_mesh_KBlocks)

out_mesh = np.random.random((nPointsOut,2))
#for i in range(0,nPointsOut):
#	out_mesh[i,0] = haltonPoints[0][i] #+ 0.0001
#	out_mesh[i,1] = haltonPoints[1][i] #+ 0.01

#in_mesh[0,0] = out_mesh[0,0]
#in_mesh[0,1] = out_mesh[0,1]

tree = spatial.KDTree(list(zip(in_mesh[:,0],in_mesh[:,1])))
nearest_neighbors = []
shape_params = []

plt.scatter(puCentres[:,0], puCentres[:,1], label = "In Mesh", s=20)
plt.scatter(in_mesh[:,0], in_mesh[:,1], label = "Out Mesh", s=2)
plt.show()


'''
for j in range(0,1):
	queryPt = (in_mesh[j,0],in_mesh[j,1])
	nnArray = tree.query(queryPt,50)
	#print(nnArray[0][1])
	nearest_neighbors.append(nnArray[0][1])
	shape_params.append(0)

for i in range(0,5):
	ntesting = 100 + i*0
	#print("nearest_nighbors: ",nearest_neighbors)
	maxNN = max(nearest_neighbors)
	random_point_removal = [randint(0, nPoints) for p in range(0, ntesting)]
	print(random_point_removal)
	basis_mesh = np.random.random((nPoints,2))
	evaluate_mesh_intermediate = np.random.random((ntesting,2))
	#evaluate_mesh = []
	evalAppend = 0
	basisAppend = 0
	for j in range(0,nPoints):
		if j in random_point_removal:
			if in_mesh[j,0] > 0.1 and in_mesh[j,0] < 0.9: 
				if in_mesh[j,1] > 0.1 and in_mesh[j,1] < 0.9:
					evaluate_mesh_intermediate[evalAppend,0] = in_mesh[j,0]
					evaluate_mesh_intermediate[evalAppend,1] = in_mesh[j,1]
					evalAppend += 1
		else:
			basis_mesh[basisAppend,0] = in_mesh[j,0]
			basis_mesh[basisAppend,1] = in_mesh[j,1]
			basisAppend += 1
	#print(evaluate_mesh)
	evaluate_mesh = np.random.random((evalAppend,2))
	for j in range(0,evalAppend):
		evaluate_mesh[j,0] = evaluate_mesh_intermediate[j,0]
		evaluate_mesh[j,1] = evaluate_mesh_intermediate[j,1]

	mesh_size = maxNN
	plt.scatter(basis_mesh[:,0], basis_mesh[:,1], label = "In Mesh", s=2)
	plt.scatter(evaluate_mesh[:,0], evaluate_mesh[:,1], label = "Out Mesh")
	plt.show()
	func = lambda x,y: np.sin(10*x)+(0.0000001*y)
	one_func = lambda x: np.ones_like(x)
	in_vals = func(in_mesh[:,0],in_mesh[:,1])
	out_vals = func(out_mesh[:,0],out_mesh[:,1])
	removalLowestValue = 100
	LOOCVLowestValue = 100
	removalLowestPosition = 0
	LOOCVLowestPosition = 0
	DoubleMeshLowestValue = 100
	DoubleMeshLowestPosition = 0
	for k in range(3,15):
		shape_parameter = 4.55228/((k)*mesh_size)
		#print("shape_parameter: ",shape_parameter)
		bf = basisfunctions.Gaussian(shape_parameter)
		#func = lambda x: (x-0.1)**2 + 1
	
		#in_meshChange = [0, 0.02, 0.03, 0.1,0.23,0.25,0.52,0.83,0.9,0.95,1]	
	#for j in range(0,11):

		#	in_mesh[j] = in_meshChange[j]
		#print(in_mesh)
		#plot_mesh = np.linspace(0, 1, 250)
		
		evaluate_vals = func(evaluate_mesh[:,0],evaluate_mesh[:,1])
		basis_vals = func(basis_mesh[:,0],basis_mesh[:,1])
		if (i == 0):
			error_LOOCV = LOOCV(bf, in_mesh, in_vals, rescale = False)
			errorsLOOCV = error_LOOCV() 
			interpFull = NoneConsistent(bf, in_mesh, in_vals, rescale = False)	
		
		interp = NoneConsistent(bf, basis_mesh, basis_vals, rescale = False)	
		
		#print("Error: ", max(evaluate_vals - interp(evaluate_mesh)))
	
		#resc_interp = NoneConsistent(bf, in_mesh, in_vals, rescale = True)
		#one_interp = NoneConsistent(bf, in_mesh, one_func(in_mesh), rescale = False)
	
		#plt.plot(plot_mesh, func(plot_mesh), label = "Target $f$")
		#plt.plot(evaluate_mesh, interp(evaluate_mesh), "--", label = "Interpolant $S_f$")
		#plt.plot(evaluate_mesh, evaluate_vals, "--", label = "Interpolant $S_r$ of $g(x) 	= 1$")
		#plt.plot(evaluate_mesh, evaluate_vals - interp(evaluate_mesh), label = "Error on selected points")
		errors = evaluate_vals - interp(evaluate_mesh)
		errorsFull = out_vals - interpFull(out_mesh)
		plt.plot(in_mesh, errorsFull, label = "Error on selected points")
		plt.show()

		#print("Testing error: ",errors)
		print("Random removal - Average Testing error with k = ",k,": ", abs(np.average(errors)), " - Max Testing error: ", abs(max(errors)))
		print("LOOCV - Average Error with k = ", k, ": ", abs(np.average(errorsLOOCV)), " - and max error: ", abs(max(errorsLOOCV)))
		print("Double Mesh - Average Error with k = ", k, ": ", abs(np.average(errorsFull)), " - and max error: ", abs(max(errorsFull)))

		if abs(np.average(errors)) < removalLowestValue:
			removalLowestPosition = k
			removalLowestValue = abs(np.average(errors))
		if abs(np.average(errorsLOOCV)) < LOOCVLowestValue:
			LOOCVLowestPosition = k
			LOOCVLowestValue = abs(np.average(errorsLOOCV))
		if abs(np.average(errorsFull)) < DoubleMeshLowestValue:
			DoubleMeshLowestPosition = k
			DoubleMeshLowestValue = abs(np.average(errorsFull))
		

	#plt.tight_layout()
	#plt.plot(in_mesh, in_vals, label = "Rescaled Interpolant")

	#rint("RMSE no rescale =", interp.RMSE(func, plot_mesh))
	#print("RMSE rescaled   =", resc_interp.RMSE(func, plot_mesh))
	print("Random removal place: ", removalLowestPosition, " - LOOCV place: ", LOOCVLowestPosition, " - Double Mesh place: ", DoubleMeshLowestPosition)
	print("Random removal value: ", removalLowestValue, " - LOOCV value: ", LOOCVLowestValue, " - Double Mesh value: ", DoubleMeshLowestValue)
end = time.time()
print("Elapsed time for optimization: ", end - start)

'''
