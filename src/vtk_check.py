import pyvista as pv

file = "./data/vortices_vtk/strain_1.0strain_3.0_Z40C_093003b-192.vtk"
data = pv.read(file)
print(data)
