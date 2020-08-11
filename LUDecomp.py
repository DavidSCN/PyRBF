""" Generates data to show the effect of rescaling. Low density basisfunctions used. """

import pandas
from rbf import *
import basisfunctions, testfunctions
import matplotlib.pyplot as plt
import time
import mesh
import math

start = time.time()
j = 0
for i in range(0,1):
	nPoints = 5
	print("Number of points: ",nPoints)
	in_mesh = np.linspace(0, 1, nPoints)
	mesh_size = 1/nPoints
	shape_parameter = 4.55228/((2+i)*mesh_size)
	print("shape_parameter: ",shape_parameter)
	bf = basisfunctions.Gaussian(shape_parameter)
	#func = lambda x: (x-0.1)**2 + 1
	func = lambda x: np.sin(5*x)
	one_func = lambda x: np.ones_like(x)
	#in_meshChange = [0, 0.02, 0.03, 0.1,0.23,0.25,0.52,0.83,0.9,0.95,1]
	#for j in range(0,11):
	#	in_mesh[j] = in_meshChange[j]
	#print(in_mesh)
	#plot_mesh = np.linspace(0, 1, 250)
	in_vals = func(in_mesh)
#	evaluate_vals = func(evaluate_mesh)
#	basis_vals = func(basis_mesh)

	interp = NoneConsistent(bf, in_mesh, in_vals, rescale = False)	
	LUofC = LUDecomp(bf, in_mesh, in_vals, rescale = False)
	p, l, u = LUofC()
	#errors = error_LOOCV() 
	print("p ", p)
	print("l ", l)
	print("u ", u)

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
end = time.time()
print("Elapsed time: ", end - start)


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
