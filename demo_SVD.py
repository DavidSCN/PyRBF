""" Generates data to show the effect of rescaling. Low density basisfunctions used. """

import pandas
from rbf import *
import basisfunctions, testfunctions
import matplotlib.pyplot as plt
import time
import mesh
import math
import numpy as np
from scipy.sparse import csr_matrix

startTotal = time.time()
j = 0
#CinvTotal = csr_matrix((10, 10), dtype=np.int8).toarray()
rowArray = []
columnArray = []
invData = []
totalData = []

#for j in range(0,10):
#	for k in range(0,10):
#		rowArray.append(j)
#		columnArray.append(k)
#print(rowArray)
#print(columnArray)
nPoints = 5		#The larger nPoints, the less overlap and therefore, less iterations
iterations = 1
for i in range(0,iterations):
	start = time.time()
	print("Number of points: ",nPoints)
	in_mesh = np.linspace(0, 1, nPoints)
	mesh_size = 1
	shape_parameter = 4.55228/((2)*mesh_size)
	print("shape_parameter: ",shape_parameter)
	bf = basisfunctions.Gaussian(shape_parameter)
	#func = lambda x: (x-0.1)**2 + 1
	func = lambda x: np.sin(5*x)
	one_func = lambda x: np.ones_like(x)
	#in_meshChange = [0, 0.02, 0.03, 0.1,0.23,0.25,0.52,0.83,0.9,0.95,1]
	#for j in range(0,11):
	#	in_mesh[j] = in_meshChange[j]
	#print(in_mesh)
	plot_mesh = np.linspace(0, 1, 250)
	in_vals = func(in_mesh)
	#evaluate_vals = func(plot_mesh)
#	basis_vals = func(basis_mesh)
	
	#interp = NoneConsistent(bf, in_mesh, in_vals, rescale = False)	
	full_SVD = fullSVD(bf, in_mesh, in_vals, rescale = False)
	#US,S,V = 
	C, U, s, Vh = full_SVD() 
	#for j in range(0,nPoints):
#		for k in range(0,nPoints):
#			rowArray.append(i*nPoints + j)
#			columnArray.append(i*nPoints + k)
#			invData.append(Cinv[j][k])
#		totalData.append(i)
	#print("Error: ", Cinv)
	print('u,s,v', C)
	#Vt = np.transpose(Vh)
	#Sinv = np.linalg.inv(s)
	#Ut = np.transpose(U)
	Cinv = np.transpose(Vh) @ np.diag(1/s) @ np.transpose(U)
	print("Cinv:",Cinv)
	print("In vals:",in_vals)
	gamma = Cinv @ np.transpose(in_vals)
	print("gamma: ", gamma)
	new_vals = np.transpose(gamma) @ C
	print("Erro: ",in_vals - new_vals)
	
	end = time.time()
	print("Elapsed time: ", end - start)

	#resc_interp = NoneConsistent(bf, in_mesh, in_vals, rescale = True)
	#one_interp = NoneConsistent(bf, in_mesh, one_func(in_mesh), rescale = False)
	
	#plt.plot(in_mesh, errors, label = "Interpolation error with LOOCV")
	#plt.legend()
	#plt.show()
	#plt.plot(evaluate_mesh, interp(evaluate_mesh), "--", label = "Interpolant $S_f$")
	#plt.plot(evaluate_mesh, evaluate_vals, "--", label = "Interpolant $S_r$ of $g(x) = 1$")
	#plt.plot(evaluate_mesh, evaluate_vals - interp(evaluate_mesh), "--", label = "Error on selected points")


	#plt.tight_layout()
	#plt.plot(in_mesh, in_vals, label = "Rescaled Interpolant")

	#rint("RMSE no rescale =", interp.RMSE(func, plot_mesh))
	#print("RMSE rescaled   =", resc_interp.RMSE(func, plot_mesh))
	#some_functionMesh = np.linspace(0, 3, nPoints)
	#some_func = lambda x: np.sin(10*x)*x*x
	#some_func_values = some_func(some_functionMesh)

	#evaluation = Cinv*some_func_values
	#print("evaluation", evaluation)

endTotal = time.time()
print("Elapsed time: ", endTotal - startTotal)

#CinvTotal = csr_matrix((invData, (rowArray, columnArray)), #shape=(iterations*nPoints,iterations*nPoints)).toarray()
#print(CinvTotal)

#start = time.time()
#for ll in range(0,1):
#	sumNewTotal = CinvTotal*totalData
#end = time.time()
#print("Matrix Multiplication time: ", end - start)

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
