
"""
make_tube_mesh.py — parametric 3D tube mesh generator for FEniCSx
Generates: tube.xdmf + tube.h5 (if needed)
Physical tags:
  Volume: "TUBE" (id 1)
  Surfaces: "TOP"=101, "BOTTOM"=102, "LATERAL"=103, ("INNER"=104 if hollow)
"""

import gmsh
import meshio
import numpy as np
import os

# ---------------- Parameters ----------------
L = 0.04          # tube length [m]
D = 0.005         # outer diameter [m]
R = D / 2
Ri = 0
res_axial = 4   # number of divisions along length was 4
res_circ  = 8   # around circumference

mesh_scale = 0.2 # global refinement factor (smaller -> finer) 0.15 ( 0.2 0.4 with res axial 4 )

# Compute element size heuristics
lc_axial = L / res_axial * mesh_scale
lc_circ  = (2 * np.pi * R) / res_circ * mesh_scale
lc = min(lc_axial, lc_circ)

gmsh.initialize()
gmsh.model.add("tube")

# ---------------- Geometry ----------------
# Build outer and inner cylinders along +Z
outer = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, L, R)
if Ri > 0:
    inner = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, L, Ri)
    vol = gmsh.model.occ.cut([(3, outer)], [(3, inner)], removeObject=True, removeTool=True)
    vtag = vol[0][1]
else:
    vtag = outer
gmsh.model.occ.synchronize()

# Tag surfaces
surf_all = gmsh.model.getBoundary([(3, vtag)], oriented=False, recursive=False)
top = gmsh.model.getEntitiesInBoundingBox(-R-1e-6, -R-1e-6, L-1e-6,
                                          R+1e-6, R+1e-6, L+1e-6, 2)
bot = gmsh.model.getEntitiesInBoundingBox(-R-1e-6, -R-1e-6, -1e-6,
                                          R+1e-6, R+1e-6, 1e-6, 2)
inner = []
if Ri > 0:
    inner = gmsh.model.getEntitiesInBoundingBox(-Ri-1e-6, -Ri-1e-6, 1e-6,
                                                Ri+1e-6, Ri+1e-6, L-1e-6, 2)
# lateral = all boundary - top - bottom - inner
top_ids = {s[1] for s in top}
bot_ids = {s[1] for s in bot}
inner_ids = {s[1] for s in inner}
lat = [(2, s[1]) for s in surf_all if s[1] not in top_ids | bot_ids | inner_ids]

# ---------------- Physical groups ----------------
gmsh.model.addPhysicalGroup(3, [vtag], 1)
gmsh.model.setPhysicalName(3, 1, "TUBE")

if top:     gmsh.model.addPhysicalGroup(2, [s[1] for s in top], 101)
if bot:     gmsh.model.addPhysicalGroup(2, [s[1] for s in bot], 102)
if lat:     gmsh.model.addPhysicalGroup(2, [s[1] for s in lat], 103)
if inner:   gmsh.model.addPhysicalGroup(2, [s[1] for s in inner], 104)

# ---------------- Mesh settings ----------------
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
gmsh.option.setNumber("Mesh.ElementOrder", 1)

gmsh.model.mesh.generate(3)

# ---------------- Write MSH ----------------
gmsh.write("tube.msh")

# ---------------- Convert to XDMF ----------------

print("Converting tube.msh → tube.xdmf / tube_facets.xdmf …")

msh = meshio.read("tube.msh")

cells = []
facets = []
cell_data = {"cell_tags": []}
facet_data = {"facet_tags": []}

# Each entry in msh.cells has a corresponding entry in
# msh.cell_data["gmsh:physical"], in the same order.
phys_list = msh.cell_data["gmsh:physical"]

for cell_block, tag_array in zip(msh.cells, phys_list):
    ctype = cell_block.type
    if ctype in ("tetra", "tetra10"):
        cells.append((ctype, cell_block.data))
        cell_data["cell_tags"].append(tag_array)
    elif ctype in ("triangle", "triangle6"):
        facets.append((ctype, cell_block.data))
        facet_data["facet_tags"].append(tag_array)

# Volume mesh
if cells:
    meshio.write(
        "tube.xdmf",
        meshio.Mesh(
            points=msh.points,
            cells=cells,
            cell_data=cell_data,
        ),
        file_format="xdmf",
    )

# Facet mesh
if facets:
    meshio.write(
        "tube_facets.xdmf",
        meshio.Mesh(
            points=msh.points,
            cells=facets,
            cell_data=facet_data,
        ),
        file_format="xdmf",
    )
    
m = meshio.read("tube_facets.xdmf")

# Keep only triangle blocks
tri_blocks = []
tri_data   = []
for cells, tags in zip(m.cells, m.cell_data["facet_tags"]):
    if cells.type.startswith("triangle"):
        tri_blocks.append((cells.type, cells.data))
        tri_data.append(tags)

m_tri = meshio.Mesh(
    points=m.points,
    cells=tri_blocks,
    cell_data={"facet_tags": tri_data}
)
meshio.write("tube_facets_linear.xdmf", m_tri)

print(" Wrote tube.xdmf and tube_facets.xdmf successfully.")
