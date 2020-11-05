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
#nPoints = pow(90,2)
nPoints = 40
#nPoints = len(mesh.points)	# number of points

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
x_range = x_max - x_min
y_range = y_max - y_min
max_values = [x_max, y_max]

print("x min: ", x_min)
print("x max: ", x_max)
print("y min: ", y_min)
print("y max: ", y_max)

hyperVolume = (x_max-x_min)*(y_max-y_min)
print("Hypervolume: ", hyperVolume)
# Number of patches per dimension
#hpParameter = 60		# Find value to automate parameter
hpParameter = 0.5*nPoints/pow(nPoints,0.5)
#nPatchCentres = math.floor(0.5*(l_max - l_min)*pow(nPoints/(hpParameter*hyperVolume),1/dimension_M))
nPatchCentres = math.floor(1*pow(nPoints/2,1/dimension_M))
print("n Patch Centres: ", nPatchCentres)
#Radius of the patch (half the diameter)
patchRadii = (l_max - l_min)/nPatchCentres	
print("n Patch Radii: ", patchRadii)

puCentres = np.random.random((pow(nPatchCentres,dimension_M),2))
j = 0
k = 0

patchOffset = 1/nPatchCentres
for i in range(0,pow(nPatchCentres,dimension_M)):
	if (j == nPatchCentres):
		j = 0
		k += 1
	puCentres[i,0] = j*1.*patchRadii + patchOffset*patchRadii
	puCentres[i,1] = k*1.*patchRadii + patchOffset*patchRadii
	#print("PU coords: ", puCentres[i,0], puCentres[i,1])
	j += 1

# find q^M blocks
q = math.ceil((l_max - l_min)/patchRadii)
print("q: ", q)

### Find the block where the PU domain centre lies in. Similar to linked cell method
'''
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
'''
	
### Repeat the method for all in_mesh points
'''
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
'''

### Old method works fine. Add new linked cell algorithm from - Linked-List Cell Molecular Dynamics- 
'''
km = 0
kthBlock = 0
puKBlocks = []
for i in range(0,pow(nPatchCentres,dimension_M)):
	km = 0
	km = math.floor(puCentres[i,1]/patchRadii)*nPatchCentres
	km += math.floor(puCentres[i,0]/patchRadii)
	puKBlocks.append(km)
print("puKBlocks: ", puKBlocks)

km = 0
kthBlock = 0
in_mesh_KBlocks = []
for i in range(0,nPoints):
	km = 0
	km = math.floor(in_mesh[i,1]/patchRadii)*nPatchCentres
	km += math.floor(in_mesh[i,0]/patchRadii)
	in_mesh_KBlocks.append(km)
print("in_mesh_KBlocks: ", in_mesh_KBlocks)
'''

#headPatches = -1*np.ones(pow(nPatchCentres,2))
#print("headPatches: ", headPatches)
#lsclPatches = -1*np.ones(nPoints)
#print("lsclPatches: ", lsclPatches)
headPatches = []
lsclPatches = []
for k in range(0,pow(nPatchCentres,dimension_M)):
	headPatches.append(-1)
for k in range(0,pow(nPatchCentres,dimension_M)):
	lsclPatches.append(-1)

#print("headPatches: ", headPatches)

for i in range(0,pow(nPatchCentres,dimension_M)):
	mc = [0,0,0]
	c = 0
	for j in range(0,dimension_M):
		mc[j] = math.floor(puCentres[i,j]/patchRadii)
		if puCentres[i,j] >= max_values[j]:
			mc[j] -= 1
	c = mc[0]*pow(nPatchCentres,0) + mc[1]*nPatchCentres + mc[2]
	#print("c: ", c,mc[0], mc[1], mc[2])
	lsclPatches[i] = headPatches[int(c)]
	headPatches[int(c)] = i

#print("headPatches: ", headPatches)
#print("lsclPatches: ", lsclPatches)

headInterpPoints = []
lsclInterpPoints = []
for k in range(0,pow(nPatchCentres,dimension_M)):
	headInterpPoints.append(-1)
for k in range(0,nPoints):
	lsclInterpPoints.append(-1)

#print("headInterpPoints: ", headInterpPoints)

regionWidth = patchRadii + 0.001*patchRadii
print("regionWidth: ", regionWidth)
for i in range(0,nPoints):
	mc = [0,0,0]
	c = 0
	for j in range(0,dimension_M):
		mc[j] = math.floor(in_mesh[i,j]/regionWidth)
		if in_mesh[i,j] >= max_values[j]:
			mc[j] -= 1
	c = mc[0]*1 + mc[1]*nPatchCentres + mc[2]
	#print("c: ", c,mc[0], mc[1], mc[2])
	lsclInterpPoints[i] = headInterpPoints[int(c)]
	headInterpPoints[int(c)] = i

#print("headInterpPoints: ", headInterpPoints)
#print("lsclInterpPoints: ", lsclInterpPoints)

# -------------------------------
# 
# Loop through all blocks defined by q. Loop through each vertex in blocks k-q-1, k-q, k-q+1,
# k-1, k, k+1, k+q-1, k+q, k+q+1.
#
# -------------------------------

surroundBlocks = [-q-1, -q, -q+1, -1, 0, 1, q-1, q, q+1]
print("surroundBlocks: ",surroundBlocks)
# Loop through each patch
#for i in range(0,nPatchCentres):	# loops in x direction
#	for j in range(0,nPatchCentres):	# loops in y direction
#pointsInPatchWithinRadius = []
for i in range(0,len(headPatches)):
	pointsInPatchWithinRadius = []
	xPatchCentre = puCentres[i,0]	
	yPatchCentre = puCentres[i,1]
	for k in range(0,9):
		blockNum = surroundBlocks[k] + i
		#print("Block num: ", blockNum)
		if (blockNum >= 0):
			if (blockNum < len(headPatches)):
				vertexID = lsclInterpPoints[headInterpPoints[blockNum]]		
				while (vertexID > 0):
					#print("Vertex : ", vertexID)
					x = in_mesh[vertexID,0]
					y = in_mesh[vertexID,1] 
					r = math.sqrt(pow(x-xPatchCentre,2) + pow(y-yPatchCentre,2))
					if (r <= patchRadii*1):
						pointsInPatchWithinRadius.append(vertexID)
					vertexID = lsclInterpPoints[vertexID]
	#print("pointsInPatchWithinRadius: ", pointsInPatchWithinRadius)



### Find points in each cells using the arrays
'''
for i in range(0,pow(nPatchCentres,dimension_M)):
	index = headInterpPoints[i]
	indexLSCL = lsclInterpPoints[index]
	while (indexLSCL > -1):
		print("Value in cell: ", i, " - is: ", indexLSCL)
		indexLSCL = lsclInterpPoints[indexLSCL]
'''
end = time.time()
print("Elapsed time to allocate linked list data structures: ", end - start)


out_mesh = np.random.random((nPointsOut,2))
#for i in range(0,nPointsOut):
#	out_mesh[i,0] = haltonPoints[0][i] #+ 0.0001
#	out_mesh[i,1] = haltonPoints[1][i] #+ 0.01

#in_mesh[0,0] = out_mesh[0,0]
#in_mesh[0,1] = out_mesh[0,1]

'''
km = 0
kthBlock = 0
out_mesh_KBlocks = []
for i in range(0,nPointsOut):
	km = 0
	km = math.floor(out_mesh[i,1]/patchRadii)*nPatchCentres
	km += math.floor(out_mesh[i,0]/patchRadii)
	out_mesh_KBlocks.append(km)
print("out_mesh_KBlocks: ", out_mesh_KBlocks)
'''
tree = spatial.KDTree(list(zip(in_mesh[:,0],in_mesh[:,1])))
nearest_neighbors = []
shape_params = []

plt.scatter(puCentres[:,0], puCentres[:,1], label = "In Mesh", s=20)
plt.scatter(in_mesh[:,0], in_mesh[:,1], label = "Out Mesh", s=2)
plt.show()
