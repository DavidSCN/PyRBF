""" Generates data to show the effect of rescaling. Low density basisfunctions used. """

import pandas
from rbf import *
import basisfunctions, testfunctions
import matplotlib.pyplot as plt
import time
import mesh
import math
from random import randint
from scipy import spatial

start = time.time()
j = 0
nPoints = 2000
nPointsOut = 100
print("Number of points: ",nPoints)
in_mesh = np.random.random((nPoints,2))
out_mesh = np.random.random((nPointsOut,2))
tree = spatial.KDTree(list(zip(in_mesh[:,0],in_mesh[:,1])))
nearest_neighbors = []
shape_params = []

plt.scatter(in_mesh[:,0], in_mesh[:,1], label = "In Mesh")
plt.scatter(out_mesh[:,0], out_mesh[:,1], label = "Out Mesh")
plt.show()

for j in range(0,nPoints):
	queryPt = (in_mesh[j,0],in_mesh[j,1])
	nnArray = tree.query(queryPt,2)
	#print(nnArray[0][1])
	nearest_neighbors.append(nnArray[0][1])
	shape_params.append(0)

for i in range(0,5):
	ntesting = 10 + i*10
	#print("nearest_nighbors: ",nearest_neighbors)
	maxNN = max(nearest_neighbors)
	random_point_removal = [randint(0, nPoints) for p in range(0, ntesting)]
	print(random_point_removal)
	basis_mesh = np.random.random((nPoints,2))
	evaluate_mesh = np.random.random((ntesting,2))
	evalAppend = 0
	basisAppend = 0
	for i in range(0,nPoints):
		if i in random_point_removal:
			evaluate_mesh[evalAppend,0] = in_mesh[i,0]
			evaluate_mesh[evalAppend,1] = in_mesh[i,1]
			evalAppend += 1
		else:
			basis_mesh[basisAppend,0] = in_mesh[i,0]
			basis_mesh[basisAppend,1] = in_mesh[i,1]
			basisAppend += 1
	#print(evaluate_mesh)
	mesh_size = maxNN
	plt.scatter(basis_mesh[:,0], basis_mesh[:,1], label = "In Mesh")
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
	for k in range(1,10):
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
	
		interp = NoneConsistent(bf, basis_mesh, basis_vals, rescale = False)	
		error_LOOCV = LOOCV(bf, in_mesh, in_vals, rescale = False)
		errorsLOOCV = error_LOOCV() 
		interpFull = NoneConsistent(bf, in_mesh, in_vals, rescale = False)	
		#print("Error: ", max(evaluate_vals - interp(evaluate_mesh)))
	
		#resc_interp = NoneConsistent(bf, in_mesh, in_vals, rescale = True)
		#one_interp = NoneConsistent(bf, in_mesh, one_func(in_mesh), rescale = False)
	
		#plt.plot(plot_mesh, func(plot_mesh), label = "Target $f$")
		#plt.plot(evaluate_mesh, interp(evaluate_mesh), "--", label = "Interpolant $S_f$")
		#plt.plot(evaluate_mesh, evaluate_vals, "--", label = "Interpolant $S_r$ of $g(x) 	= 1$")
		#plt.plot(evaluate_mesh, evaluate_vals - interp(evaluate_mesh), label = "Error on selected points")
		errors = evaluate_vals - interp(evaluate_mesh)
		errorsFull = out_vals - interpFull(out_mesh)

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
