""" Generates data to show the effect of rescaling. Low density basisfunctions used. """
""" Halton sequence from https://laszukdawid.com/2017/02/04/halton-sequence-in-python/  """

"""Finds the 5 nearest neighbors for each vertex. A gaussian is applied using the domain width.
	Points closer together has a larger summation of the gaussian.
"""

import pandas
from rbf import *
from rbf_2d import *
import basisfunctions, testfunctions
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import time
import mesh
import math
from scipy import spatial
from halton import *

nPoints = 2000
nPointsOut = 20

haltonPoints = halton_sequence(nPoints, 2)
print("Halton: ",haltonPoints[0][4])


x = np.linspace(0,1,nPointsOut)
out_vals = []
print("Number of points on input mesh: ",nPoints)
print("Number of points on output mesh: ",nPointsOut*nPointsOut)
#in_mesh = np.linspace((1,2),(10,20),nPoints)
in_mesh = np.random.random((nPoints,2))
out_mesh = np.random.random((nPointsOut*nPointsOut,2))

for i in range(0,nPoints):
	#in_mesh[i,0] = in_mesh[i,0]*in_mesh[i,0]
	#in_mesh[i,1] = in_mesh[i,1]
	in_mesh[i,0] = haltonPoints[0][i]
	in_mesh[i,1] = haltonPoints[1][i]

#To focus points in 1 dimension
for i in range(0,nPoints):
	for j in range(0,9):
		if in_mesh[i,0] < 0.5:
			in_mesh[i,0] = 0.05*(0.25 - in_mesh[i,0]) + in_mesh[i,0]
		if in_mesh[i,0] > 0.5:
			in_mesh[i,0] = 0.05*(0.75 - in_mesh[i,0]) + in_mesh[i,0]
		if in_mesh[i,1] < 0.5:
			in_mesh[i,1] = 0.05*(0.25 - in_mesh[i,1]) + in_mesh[i,1]
		if in_mesh[i,1] > 0.5:
			in_mesh[i,1] = 0.05*(0.75 - in_mesh[i,1]) + in_mesh[i,1]

start = time.time()	# Start timing

tree = spatial.KDTree(list(zip(in_mesh[:,0],in_mesh[:,1])))
#
nearest_neighbors = []
shape_params = []

#for j in range(0,nPoints):
#		queryPt = (in_mesh[j,0],in_mesh[j,1])
#		nnArray = tree.query(queryPt,2)
#		nearest_neighbors.append(nnArray[0][1])

for i in range(0,nPointsOut):
	for j in range(0,nPointsOut):
		out_mesh[j + i*nPointsOut,1] = x[j]
		out_mesh[j + i*nPointsOut,0] = x[i]
		out_vals.append(0)

#treeOut = spatial.KDTree(list(zip(out_mesh[:,0],out_mesh[:,1])))

cutOff = 0.2
s = 4.55228 / (cutOff)
gaussValueXVec = []
gaussValueYVec = []

# Searches all points within a radius 
for j in range(0,nPointsOut*nPointsOut):
		queryPt = (out_mesh[j,0],out_mesh[j,1])
		nnArray = tree.query_ball_point(queryPt,cutOff)
		gaussValue = 0
		gaussValueX = 0
		gaussValueY = 0
		#print(nnArray)
		#print(len(nnArray))
		for k in range(0,len(nnArray)):
			radius = math.sqrt(np.power(in_mesh[nnArray[k],0]-out_mesh[j,0],2) + np.power(in_mesh[nnArray[k],1]-out_mesh[j,1],2))
			gaussValue += np.exp( -np.power(s * radius,2))
			gaussValueX += np.exp( -np.power(s * (in_mesh[nnArray[k],0]-out_mesh[j,0]),2))
			gaussValueY += np.exp( -np.power(s * (in_mesh[nnArray[k],1]-out_mesh[j,1]),2))
		shape_params.append(gaussValue)
		gaussValueXVec.append(gaussValueX)
		gaussValueYVec.append(gaussValueY)

# Searches a set number of nearest neighbors
"""
for j in range(0,nPointsOut*nPointsOut):
		queryPt = (in_mesh[j,0],in_mesh[j,1])
		nnArray = tree.query(queryPt,4)
		gaussValue = 0
		gaussValueX = 0
		gaussValueY = 0
		#print(nnArray)
		#print(len(nnArray))
		for k in range(0,4):
			radius = math.sqrt(np.power(in_mesh[nnArray[k],0]-out_mesh[j,0],2) + np.power(in_mesh[nnArray[k],1]-out_mesh[j,1],2))
			gaussValue += np.exp( -np.power(s * radius,2))
			gaussValueX += np.exp( -np.power(s * (in_mesh[nnArray[k],0]-out_mesh[j,0]),2))
			gaussValueY += np.exp( -np.power(s * (in_mesh[nnArray[k],1]-out_mesh[j,1]),2))
		#nearest_neighbors.append(nnArray[0][1])
		shape_params.append(gaussValue)
		gaussValueXVec.append(gaussValueX)
		gaussValueYVec.append(gaussValueY)
"""

#maxValue = max(shape_params)
#shape_params = shape_params/maxValue
sMin = min(shape_params)
#shape_params -= sMin

func = lambda x,y: np.sin(2*x)+(0.0000001*y)
in_vals = func(in_mesh[:,0],in_mesh[:,1])

#fig = plt.figure()
#ax = Axes3D(fig)
#surf = ax.plot_trisurf(in_mesh[:,0], in_mesh[:,1], in_vals, cmap=cm.jet, linewidth=0.1)
#fig.colorbar(surf, shrink=0.5, aspect=5)
#plt.show()



"""
treeOut = spatial.KDTree(list(zip(out_mesh[:,0],out_mesh[:,1])))

nnAmount = 10

for j in range(0,nPoints):
	queryPt = (in_mesh[j,0],in_mesh[j,1])
	nnArray = treeOut.query(queryPt,nnAmount)
	#print(nnArray)
	for k in range(0,nnAmount):
		#print(nnArray[1][k])
		out_vals[nnArray[1][k]] += 1 - min(nnArray[0][k]*10,1)	
"""

#for i in range(0,nPoints):
#	for j in range(0,nPointsOut):
#		distanceX = in_mesh[i,0] - out_mesh[j,0]
#		distanceY = in_mesh[i,1] - out_mesh[j,1]
#		distance = math.sqrt(math.pow(distanceX,2) + math.pow(distanceY,2))
#		out_vals[j] += 1 - min(distance*10,1)

end = time.time()
print("Elapsed time: ", end - start)

plt.scatter(in_mesh[:,0], in_mesh[:,1], label = "In Mesh",s=2)
plt.scatter(out_mesh[:,0], out_mesh[:,1], label = "Out Mesh")
plt.show()

fig = plt.figure()
ax = Axes3D(fig)
#surf = ax.plot_trisurf(out_mesh[:,0], out_mesh[:,1], out_vals, cmap=cm.jet, linewidth=0.1)
#surf = ax.plot_trisurf(out_mesh[:,0], out_mesh[:,1], shape_params, cmap=cm.jet, linewidth=0.1)
surf = ax.scatter(out_mesh[:,0], out_mesh[:,1], shape_params, cmap=cm.jet, linewidth=0.1)
fig.colorbar(surf, shrink=0.5, aspect=5)
#plt.show()

fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_trisurf(out_mesh[:,0], out_mesh[:,1], gaussValueXVec, cmap=cm.jet, linewidth=0.1)
fig.colorbar(surf, shrink=0.5, aspect=5)
#plt.show()

fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_trisurf(out_mesh[:,0], out_mesh[:,1], gaussValueYVec, cmap=cm.jet, linewidth=0.1)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

"""
for i in range(0,1):
	nPoints = 16000
	print("Number of points: ",nPoints)
	#in_mesh = np.linspace((1,2),(10,20),nPoints)
	in_mesh = np.random.random((nPoints,2))
	tree = spatial.KDTree(list(zip(in_mesh[:,0],in_mesh[:,1])))
	nearest_neighbors = []
	shape_params = []
	for j in range(0,nPoints):
		queryPt = (in_mesh[j,0],in_mesh[j,1])
		nnArray = tree.query(queryPt,2)
		#print(nnArray[0][1])
		nearest_neighbors.append(nnArray[0][1])
		shape_params.append(0)
	#print("nearest_nighbors: ",nearest_neighbors)
	maxNN = max(nearest_neighbors)
	
	func = lambda x,y: np.sin(2*x)+(0.0000001*y)
	one_func = lambda x: np.ones_like(x)
	in_vals = func(in_mesh[:,0],in_mesh[:,1])

	#print(tree.query(pts))
	#print("in_mesh: ", in_mesh)
	for k in range(5,6):	
		for j in range(0,nPoints):
			shape_params[j]=4.55228/(k*maxNN)
			#shape_params[j]=4.55228/(i*nearest_neighbors[j])
		#print("shape_params: ", shape_params)
		#mesh_size = 1/math.sqrt(nPoints)
		#print("mesh_size: ",mesh_size)
		#shape_parameter = 4.55228/((5)*mesh_size)
		#print("shape_parameter: ",shape_parameter)
		bf = basisfunctions.Gaussian(list(shape_params))
		#print("BF: ", bf)
		#func = lambda x: (x-0.1)**2 + 1
	
		#in_meshChange = [0, 0.02, 0.03, 0.1,0.23,0.25,0.52,0.83,0.9,0.95,1]
		#for j in range(0,11):
		#	in_mesh[j] = in_meshChange[j]
		#print(in_mesh)
		
		#plot_mesh = np.random.random((nPoints,2))
		
		#print(in_vals)
	#	evaluatine_vals = func(evaluate_mesh)
	#	basis_vals = func(basis_mesh)
	
		#interp = NoneConsistent(bf, in_mesh, in_vals, rescale = False)
		error_LOOCV = LOOCV(bf, in_mesh, in_vals, rescale = False)
		errors = error_LOOCV() 
		#print("Error with i = ", i, ": ", errors)
		print("LOOCV - Max Error with k = ", k, ": ", np.average(errors), " - and max error: ", max(errors))
	
	#error_LOOCVSVD = LOOCVSVD(bf, in_mesh, in_vals, rescale = False)
	#errorsSVD = error_LOOCVSVD() 
	#print("Error SVD: ", max(errorsSVD))
	#print("Error Difference: ", errors - errorsSVD)

	#end = time.time()
	#print("Elapsed time: ", end - start)

	#resc_interp = NoneConsistent(bf, in_mesh, in_vals, rescale = True)
	#one_interp = NoneConsistent(bf, in_mesh, one_func(in_mesh), rescale = False)
	'''
	plt.scatter(in_mesh[:,0], in_mesh[:,1], label = "In Mesh")
	plt.scatter(plot_mesh[:,0], plot_mesh[:,1], label = "Out Mesh")
	
	#plt.legend()
	#plt.show()

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	ax.scatter(in_mesh[:,0],in_mesh[:,1], in_vals, c='r', marker='o')

	ax.set_xlabel('X coordinate')
	ax.set_ylabel('Y ycoordinate')
	ax.set_zlabel('Magnitude')

	#plt.show()
	'''

	#fig = plt.figure()
	#ax = Axes3D(fig)
	#plt.scatter(in_mesh[:,0],in_mesh[:,1], errors, c = 'b', marker='o')

	#plt.plot(evaluate_mesh, interp(evaluate_mesh), "--", label = "Interpolant $S_f$")
	#plt.plot(evaluate_mesh, evaluate_vals, "--", label = "Interpolant $S_r$ of $g(x) = 1$")
	#plt.plot(evaluate_mesh, evaluate_vals - interp(evaluate_mesh), "--", label = "Error on selected points")


	#plt.tight_layout()
	#plt.plot(in_mesh, in_vals, label = "Rescaled Interpolant")

	#fig = plt.figure()
	#ax = Axes3D(fig)
	#surf = ax.plot_trisurf(in_mesh[:,0], in_mesh[:,1], in_vals)
	#fig.colorbar(surf, shrink=0.5, aspect=5)
	#plt.savefig('testSurrogate.pdf')
	#plt.show()

	#fig = plt.figure()
	#ax = fig.add_subplot(111, projection='3d')

	#ax.scatter(in_mesh[:,0],in_mesh[:,1], errors, c='r', marker='o')

	#ax.set_xlabel('X coordinate')
	#ax.set_ylabel('Y ycoordinate')
	#ax.set_zlabel('Error Magnitude')

	#plt.show()

	#rint("RMSE no rescale =", interp.RMSE(func, plot_mesh))
	#print("RMSE rescaled   =", resc_interp.RMSE(func, plot_mesh))

end = time.time()
print("Elapsed time: ", end - start)

"""
