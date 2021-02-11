from sect.triangulation import constrained_delaunay_triangles
import numpy as np

def polygon2mesh(polygon):#,mesh_vertices,mesh_faces,mesh_vertices_id,mesh_vertices_color,color=[0,0,0]):
    polygon = [(x,y) for x,y in polygon]
    tri = np.array(constrained_delaunay_triangles(polygon))
    mesh_vertice = tri.reshape(tri.shape[0] * 3, 2)
    mesh_vertice = np.concatenate((mesh_vertice,np.zeros(mesh_vertice.shape[0]).reshape(-1,1)),1)
    mesh_face = np.arange(mesh_vertice.shape[0]).reshape(tri.shape[0], 3)
    return mesh_vertice,mesh_face#,mesh_vertices_id,mesh_vertices_color