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

'''
def read_dataset(filename):
	extension = os.path.splitext(filename)[1]
	if (extension == ".vtk"): # VTK Legacy format
		reader = vtk.vtkDataSetReader()
	else:
		raise MeshFormatError()
	reader.SetFileName(filename)
	reader.Update()
	return reader.GetOutput()



vtkmesh = read_dataset('l1.vtk')
points = []
cells = []
pointdata = []
cell_types = []
points = [vtkmesh.GetPoint(i) for i in range(vtkmesh.GetNumberOfPoints())]
for i in range(vtkmesh.GetNumberOfCells()):
    cell = vtkmesh.GetCell(i)
    cell_type = cell.GetCellType()
    if cell_type not in [vtk.VTK_LINE, vtk.VTK_TRIANGLE]:
        continue
    cell_types.append(cell_type)
    entry = ()
    for j in range(cell.GetNumberOfPoints()):
        entry += (cell.GetPointId(j),)
    cells.append(entry)
if not tag:
    # vtk Python utility method. Same as tag=="scalars"
    fieldData = vtkmesh.GetPointData().GetScalars()
else:
    fieldData = vtkmesh.GetPointData().GetAbstractArray(tag)
if fieldData:
    for i in range(vtkmesh.GetNumberOfPoints()):
        pointdata.append(fieldData.GetTuple1(i))

print(pointdata)
'''

start = time.time()
j = 0
nPoints = 500
nPointsOut = 200
print("Number of points: ",nPoints)
in_mesh = np.random.random((nPoints,2))

haltonPoints = halton_sequence(nPoints, 2)
for i in range(0,nPoints):
	in_mesh[i,0] = haltonPoints[0][i]
	in_mesh[i,1] = haltonPoints[1][i]

haltonPoints = halton_sequence(nPointsOut, 2)
out_mesh = np.random.random((nPointsOut,2))
#for i in range(0,nPointsOut):
#	out_mesh[i,0] = haltonPoints[0][i] + 0.1*randint(0, 1)
#	out_mesh[i,1] = haltonPoints[1][i] + 0.1*randint(0, 1)

tree = spatial.KDTree(list(zip(in_mesh[:,0],in_mesh[:,1])))
nearest_neighbors = []
shape_params = []

plt.scatter(in_mesh[:,0], in_mesh[:,1], label = "In Mesh", s=2)
plt.scatter(out_mesh[:,0], out_mesh[:,1], label = "Out Mesh", s=2)
plt.show()

for j in range(0,nPoints):
	queryPt = (in_mesh[j,0],in_mesh[j,1])
	nnArray = tree.query(queryPt,2)
	#print(nnArray[0][1])
	nearest_neighbors.append(nnArray[0][1])
	shape_params.append(0)

for i in range(0,1):
	ntesting = 100 + i*0
	#print("nearest_nighbors: ",nearest_neighbors)
	maxNN = max(nearest_neighbors)
	random_point_removal = [randint(0, nPoints) for p in range(0, ntesting)]
	#print("Indices of randomly removed points: ", random_point_removal)
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
	for k in range(7,20):
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
		#plt.plot(in_mesh, errorsFull, label = "Error on selected points")
		#plt.show()

		#print("Testing error: ",errors)
		#print("Random removal - Average Testing error with k = ",k,": ", abs(np.average(errors)), " - Max Testing error: ", abs(max(errors)))
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
