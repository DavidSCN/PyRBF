""" Generates data to show the effect of rescaling. Low density basisfunctions used. """

import pandas
from rbf import *
import basisfunctions, testfunctions
import matplotlib.pyplot as plt
import time
import mesh
import math
from random import randint
import random
from scipy import spatial
from numpy import linalg

start = time.time()
j = 0

nPoints = 2000

print("Number of points: ",nPoints)
in_mesh = np.random.random((nPoints,2))
for i in range(0,nPoints):
	in_mesh[i,0] = in_mesh[i,0]*in_mesh[i,0]
	in_mesh[i,1] = in_mesh[i,1]*in_mesh[i,1]


plt.scatter(in_mesh[:,0], in_mesh[:,1], label = "In Mesh", s=2)
plt.show()

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
#random_point_removal = [randint(0, nPoints) for p in range(0, ntesting)]
ntesting = 150
random_point_removal_all = random.sample(range(0, nPoints), ntesting)
random_point_removal = []
for i in range(0,ntesting):
	random_point_removal.append(random_point_removal_all[i])

for i in range(0,2):
	#ntesting = int(ntesting/2)
	ntesting -= 50
	for g in range(0,50):
		random_point_removal.pop()
	print(random_point_removal)
	basis_mesh = np.random.random((nPoints,2))
	partial_mesh = np.random.random((nPoints-ntesting,2))
	evaluate_mesh = np.random.random((ntesting,2))

	evalAppend = 0
	partialAppend = 0
	for i in range(0,nPoints):
		if i in random_point_removal:
			evaluate_mesh[evalAppend,0] = in_mesh[i,0]
			evaluate_mesh[evalAppend,1] = in_mesh[i,1]	
			basis_mesh[i,0] = 0
			basis_mesh[i,1] = 0
			evalAppend += 1
		else:
			partial_mesh[partialAppend,0] = in_mesh[i,0]
			partial_mesh[partialAppend,1] = in_mesh[i,1]
			basis_mesh[i,0] = in_mesh[i,0]
			basis_mesh[i,1] = in_mesh[i,1]
			partialAppend += 1
	#print(evaluate_mesh)
	mesh_size = maxNN
	func = lambda x,y: np.sin(10*x)+(0.0000001*y)
	one_func = lambda x: np.ones_like(x)
	for k in range(2,15):
		for j in range(0,nPoints):
			shape_params[j]=4.55228/(k*maxNN)
		shape_parameter = 4.55228/((k)*mesh_size)
		#print("shape_parameter: ",shape_parameter)
		bf = basisfunctions.Gaussian(shape_parameter)
		#func = lambda x: (x-0.1)**2 + 1
	
		#in_meshChange = [0, 0.02, 0.03, 0.1,0.23,0.25,0.52,0.83,0.9,0.95,1]	
	#for j in range(0,11):

		#	in_mesh[j] = in_meshChange[j]
		#print(in_mesh)
		#plot_mesh = np.linspace(0, 1, 250)
		in_vals = func(in_mesh[:,0],in_mesh[:,1])
		evaluate_vals = func(evaluate_mesh[:,0],evaluate_mesh[:,1])
		partial_vals = func(partial_mesh[:,0],partial_mesh[:,1])
	
		interp = NoneConsistent(bf, partial_mesh, partial_vals, rescale = False)	
		#error_LOOCV = LOOCV(bf, in_mesh, in_vals, rescale = False)
		#errors = error_LOOCV() 
		#print("Error: ", max(evaluate_vals - interp(evaluate_mesh)))
	
		#resc_interp = NoneConsistent(bf, in_mesh, in_vals, rescale = True)
		#one_interp = NoneConsistent(bf, in_mesh, one_func(in_mesh), rescale = False)
	
		#plt.plot(plot_mesh, func(plot_mesh), label = "Target $f$")
		#plt.plot(evaluate_mesh, interp(evaluate_mesh), "--", label = "Interpolant $S_f$")
		#plt.plot(evaluate_mesh, evaluate_vals, "--", label = "Interpolant $S_r$ of $g(x) 	= 1$")
		plt.plot(evaluate_mesh, evaluate_vals - interp(evaluate_mesh), label = "Error on selected points")
		errors = evaluate_vals - interp(evaluate_mesh)
		#print("Testing error: ",errors)
		print("k = ", k)
		print("Random removal - Average Testing error: ", np.linalg.norm(errors), " - Max Testing error: ", max(errors))

	#plt.tight_layout()
	#plt.plot(in_mesh, in_vals, label = "Rescaled Interpolant")

	#rint("RMSE no rescale =", interp.RMSE(func, plot_mesh))
	#print("RMSE rescaled   =", resc_interp.RMSE(func, plot_mesh))
end = time.time()
print("Elapsed time for optimization: ", end - start)

#plt.legend()
#plt.show()
'''
	plt.plot(plot_mesh, interp.error(func, plot_mesh))
	plt.plot(plot_mesh, resc_interp.error(func, plot_mesh))
	plt.grid()
	plt.show()

	df = pandas.DataFrame(data = { "Target" : func(plot_mesh),
                               "Interpolant" : interp(plot_mesh),
                               "RescaledInterpolant" : resc_interp(plot_mesh),
                               "OneInterpolant" : one_interp(plot_mesh),
                               "Error" : interp.error(func, plot_mesh),
                               "RescaledError" : resc_interp.error(func, plot_mesh)},
                      index = plot_mesh)

	df.to_csv("rescaled_demo.csv", index_label = "x")
'''
