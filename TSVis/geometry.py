# from sect.triangulation import constrained_delaunay_triangles
import numpy as np

def polygon2mesh(upper_line,lower_line):#,mesh_vertices,mesh_faces,mesh_vertices_id,mesh_vertices_color,color=[0,0,0]):
    # polygon = [(x,y) for x,y in polygon]
    # tri = np.array(constrained_delaunay_triangles(polygon))
    size_ul = upper_line.shape[0]
    size_ll = lower_line.shape[0]
    upper_line = upper_line[np.argsort(upper_line[:, 0])]
    lower_line = lower_line[np.argsort(lower_line[:, 0])]
    upper_line_idx = np.arange(size_ul)
    lower_line_idx = np.arange(size_ll) + size_ul

    i = 0
    j = 0
    faces = []
    turn = 'u'
    while i != size_ul - 1 or j != size_ll - 1:
        if i == size_ul - 1:
            turn = 'u'
        if j == size_ll - 1:
            turn = 'l'
        if turn == 'u':
            p1 = upper_line_idx[i]
            p2 = lower_line_idx[j+1]
            p3 = lower_line_idx[j]
            j += 1
            turn = 'l'
        else:
            p1 = lower_line_idx[j]
            p2 = upper_line_idx[i]
            p3 = upper_line_idx[i+1]
            i += 1
            turn = 'u'
        faces.append([p1,p2,p3])
    # print(len(np.unique(faces)),len(np.unique(np.concatenate((upper_line_idx,lower_line_idx)))))
    # print(np.max(faces),size_ul + size_ll - 1)
    mesh_vertice = np.concatenate((upper_line,lower_line))
    mesh_vertice = np.concatenate((mesh_vertice,np.zeros(mesh_vertice.shape[0]).reshape(-1,1)),1)
    mesh_face = np.array(faces)
    # mesh_vertice = tri.reshape(tri.shape[0] * 3, 2)
    # mesh_vertice = np.concatenate((mesh_vertice,np.zeros(mesh_vertice.shape[0]).reshape(-1,1)),1)
    # mesh_face = np.arange(mesh_vertice.shape[0]).reshape(tri.shape[0], 3)
    return mesh_vertice,mesh_face#,mesh_vertices_id,mesh_vertices_color

def line2mesh(line,thickness):
    x0 = line[:-1,0]
    x1 = line[1:,0]
    y0 = line[:-1,1]
    y1 = line[1:,1]
    dx = x1 - x0
    dy = y1 - y0
    linelength = np.sqrt(dx * dx + dy * dy)
    dx = dx /  linelength
    dy = dy / linelength
    px = 0.5 * thickness * (-dy)
    py = 0.5 * thickness * dx
    p1 = np.vstack((x0 + px, y0 + py)).T
    p2 = np.vstack((x1 + px, y1 + py)).T
    p3 = np.vstack((x1 - px, y1 - py)).T
    p4 = np.vstack((x0 - px, y0 - py)).T
    mesh_vertice = np.concatenate((p1,p2,p3,p4))
    mesh_vertice = np.concatenate((mesh_vertice,np.zeros(mesh_vertice.shape[0]).reshape(-1,1)),1)
    p1_idx = np.arange(p1.shape[0])
    p2_idx = np.arange(p2.shape[0]) + p1_idx[-1] + 1
    p3_idx = np.arange(p3.shape[0]) + p2_idx[-1] + 1
    p4_idx = np.arange(p4.shape[0]) + p3_idx[-1] + 1
    t1 = np.vstack((p1_idx,p2_idx,p3_idx)).T
    t2 = np.vstack((p3_idx,p4_idx,p1_idx)).T
    mesh_face = np.concatenate((t1,t2))
    return mesh_vertice, mesh_face
