import numpy as np
from scipy import io
import os
cimport numpy as np
from libc.math cimport sin, cos

# use the Numpy-C-API from Cython
np.import_array()

# cdefine the signature of our c function
cdef extern from "mesh_core.h":
    void _render_colors_core(
        float* image, float* vertices, int* triangles, 
        float* colors, 
        float* depth_buffer,
        int nver, int ntri,
        int h, int w, int c)

cdef float clamp(float value, float min, float max):
    if value < min:
        return min
    if value > max:
        return max
    return value

cdef class PcaModel:
    cdef:
        public np.ndarray mean
        public np.ndarray orthonormal_pca_basis
        public np.ndarray eigenvalues
        public np.ndarray rescaled_pca_basis
        public list triangle_list
    
    def __init__(self, np.ndarray[float, ndim=1] mean, np.ndarray[float, ndim=2] orthonormal_pca_basis, np.ndarray[float, ndim=1] eigenvalues, list triangle_list):
        self.mean = mean
        self.orthonormal_pca_basis = orthonormal_pca_basis
        self.eigenvalues = eigenvalues
        self.triangle_list = triangle_list
        self.rescaled_pca_basis = rescale_pca_basis(orthonormal_pca_basis, eigenvalues)
    
    cpdef int get_data_dimension(self):
        return self.rescaled_pca_basis.shape[0]
    
    cpdef list get_triangle_list(self):
        return self.triangle_list
    
    cpdef np.ndarray[float, ndim=1] get_mean(self):
        return self.mean
    
    # Projects the given data instance into the model space, and returns the PCA coefficients
    cpdef np.ndarray[float, ndim=1] project(self, np.ndarray[float, ndim=1] instance):
        cdef np.ndarray[float, ndim=1] coeffs
        coeffs = (instance - self.mean).T * self.rescaled_pca_basis
        coeffs = coeffs / self.eigenvalues
        return coeffs
    
    cpdef np.ndarray[float, ndim=2] draw_sample(self, np.ndarray[float, ndim=1] coeffs):
        cdef np.ndarray[float, ndim=1] temp_coeffs
        cdef np.ndarray[float, ndim=2] model_sample

        if len(coeffs) < self.rescaled_pca_basis.shape[1]:
            temp_coeffs = np.zeros(self.rescaled_pca_basis.shape[1])
            temp_coeffs[:len(coeffs)] = coeffs
            coeffs = temp_coeffs

        model_sample = self.mean + self.rescaled_pca_basis * coeffs
        
        return model_sample

cdef class Blendshape:
    cdef:
        public str name
        public np.ndarray deformation
    def __init__(self, np.ndarray[float, ndim=1] deformation, str name):
        self.deformation = deformation
        self.name = name

cdef class MorphableModel:
    cdef:
        public PcaModel shape_model
        public PcaModel color_model
        public PcaModel expression_model
        public list texture_coordinates
        public list texture_triangle_indices

    def __init__(self, PcaModel shape_model, PcaModel color_model, PcaModel expression_model, list texture_coordinates=[], list texture_triangle_indices=[]):
        self.shape_model = shape_model
        self.color_model = color_model
        self.expression_model = expression_model
        self.texture_coordinates = texture_coordinates
        self.texture_triangle_indices = texture_triangle_indices

    cpdef int has_color_model(self):
        if self.color_model==None:
            return 0
        return 1
    
    cpdef int has_texture_coordinates(self):
        if len(self.texture_coordinates) > 0:
            return 1
        else:
            return 0
    
    cpdef Mesh get_mean(self):
        cdef np.ndarray[float, ndim=1] shape, color
        cdef Mesh mesh

        mesh = Mesh()

        assert (self.shape_model.get_data_dimension() == self.color_model.get_data_dimension() or self.has_color_model()==0)
        shape = self.shape_model.get_mean()
        
        if self.has_color_model():
            color = self.color_model.get_mean()

        if self.expression_model != None:
            shape += self.expression_model.get_mean()
        
        if self.has_texture_coordinates():
            mesh = sample_to_mesh(shape, color, self.shape_model.get_triangle_list(), self.color_model.get_triangle_list(), self.texture_coordinates, self.texture_triangle_indices)
        else:
            mesh = sample_to_mesh(shape, color, self.shape_model.get_triangle_list(), self.color_model.get_triangle_list())

        return mesh


cdef class Mesh:
    cdef:
        # 3D vertex positions
        public list vertices
        # Colour information for each vertex. Expected to be in RGB order
        public list colors
        # Texture coordinates
        public list texcoords

        # Triangle vertex indices
        public list tvi
        # Triangle colour indices
        public list tci
        # Triangle texture indices
        public list tti
    
    def __init__(self):
        self.vertices = []
        self.colors = []
        self.texcoords = []
        self.tvi = []
        self.tci = []
        self.tti = []

cpdef tuple parse_vertex(str line):
    cdef list tokens, vertex, vertex_color = []
    tokens = line.split()
    if len(tokens)!=3 and len(tokens)!=6:
        #raise Exception("Encountered a vertex ('v') line that does not consist of either 3 ('x y z') ")
        tokens = tokens[:3]
    
    vertex = [float(tokens[0]), float(tokens[1]), float(tokens[2])]
    if len(tokens) == 6:
        vertex_color = [float(tokens[3]), float(tokens[4]), float(tokens[5])]
    
    return vertex, vertex_color

cpdef tuple parse_face(str line):
    cdef list vertex_indices=[], texture_indices=[], normal_indices=[], tokens, subtokens
    cdef int i
    tokens = line.split()
    if len(tokens)!=3 and len(tokens)!=4:
        raise Exception("Encountered a faces ('f') line that does not consist of three or four ")

    for i in range(len(tokens)):
        subtokens = tokens[i].split('/')
        vertex_indices.append(int(subtokens[0])-1)
        if len(subtokens)==2:
            texture_indices.append(int(subtokens[1])-1)
        if len(subtokens)==3:
            if subtokens[1]!='':
                texture_indices.append(int(subtokens[1])-1)
            normal_indices.append(int(subtokens[2])-1)
    
    return vertex_indices, texture_indices, normal_indices


cpdef list parse_texcoords(str line):
    cdef list tokens, texcoords
    tokens = line.split()
    if len(tokens)!=2:
        if len(tokens)==3 and float(tokens[2])==0.0:
            pass
        else:
            raise Exception("Encountered a texture coordinates ('vt') line that does not consist of two")
    texcoords = [float(tokens[0]), float(tokens[1])]
    return texcoords


cpdef Mesh read_obj(str filename):
    cdef tuple vertex_data, face_data
    cdef list texcoords
    cdef Mesh mesh

    mesh = Mesh()

    with open(filename, 'rt') as f:
        while True:
            line = f.readline()
            if not line:
                break
            if line[0]=='#':
                continue
            if line[:2]=='v ':
                vertex_data = parse_vertex(line[2:])
                mesh.vertices.append(vertex_data[0])
                if len(vertex_data[1])!=0:
                    mesh.colors.append(vertex_data[1])
            if line[:3]=='vt ':
                texcoords = parse_texcoords(line[3:])
                mesh.texcoords.append(texcoords)
            if line[:3]=='vn ':
                pass
            if line[:2]=='f ':
                face_data = parse_face(line[2:])
                if len(face_data[0])==3:
                    mesh.tvi.append(face_data[0])
                elif len(face_data[0])==4:
                    mesh.tvi.append([face_data[0][0], face_data[0][1], face_data[0][2]])
                    mesh.tvi.append([face_data[0][0], face_data[0][2], face_data[0][3]])
    return mesh

cpdef save_color_map(Mesh mesh, str filename):
    cdef np.ndarray[double, ndim=2] colors
    assert (len(mesh.vertices)==len(mesh.colors))
    
    colors = np.array(mesh.colors)
    np.save(filename, colors)
    

cpdef write_obj(Mesh mesh, str filename):
    cdef int i
    assert (len(mesh.vertices)==len(mesh.colors) or len(mesh.tti)==0)

    with open(filename, 'w') as obj_f:
        if len(mesh.colors)==0:
            for i in range(len(mesh.vertices)):
                obj_f.write("v {} {} {}\n".format(mesh.vertices[i][0],mesh.vertices[i][1],mesh.vertices[i][2]))
        else:
            for i in range(len(mesh.vertices)):
                obj_f.write("v {} {} {} {} {} {}\n".format(mesh.vertices[i][0],mesh.vertices[i][1],mesh.vertices[i][2], mesh.colors[i][0], mesh.colors[i][1], mesh.colors[i][2]))
        if len(mesh.texcoords)!=0:
            for i in range(len(mesh.texcoords)):
                obj_f.write("vt {} {}\n".format(mesh.texcoords[i][0], 1.0-mesh.texcoords[i][1]))
        for i in range(len(mesh.tvi)):
            if len(mesh.texcoords)==0:
                obj_f.write("f {} {} {}\n".format(mesh.tvi[i][0]+1, mesh.tvi[i][1]+1, mesh.tvi[i][2]+1))
            else:
                if len(mesh.tti)==0:
                    assert (len(mesh.texcoords)==len(mesh.vertices))
                    obj_f.write("f {}/{} {}/{} {}/{}\n".format(mesh.tvi[i][0]+1, mesh.tvi[i][0]+1, mesh.tvi[i][1]+1, mesh.tvi[i][1]+1, mesh.tvi[i][2]+1, mesh.tvi[i][2]+1))
                else:
                    obj_f.write("f {}/{} {}/{} {}/{}\n".format(mesh.tvi[i][0]+1, mesh.tti[i][0]+1, mesh.tvi[i][1]+1, mesh.tti[i][1]+1, mesh.tvi[i][2]+1, mesh.tti[i][2]+1))

cpdef write_textured_obj(Mesh mesh, str filename):
    cdef str mtl_filename, texture_filename
    cdef list vi, ti
    cdef int i
    assert (len(mesh.vertices)==len(mesh.colors) or len(mesh.colors)==0)
    assert (len(mesh.texcoords)!=0)
    print(len(mesh.tvi), len(mesh.tti))
    assert (len(mesh.tvi)==len(mesh.tti) or len(mesh.tti)==0)
    
    mtl_filename = filename[:-3]+'mtl'
    
    with open(filename, 'w') as obj_f:
        obj_f.write("mtllib %s\n" % os.path.basename(mtl_filename))

        if len(mesh.colors) == 0:
            for i in range(len(mesh.vertices)):
                obj_f.write("v {} {} {} \n".format(mesh.vertices[i][0], mesh.vertices[i][1], mesh.vertices[i][2]))
        else:
            for i in range(len(mesh.vertices)):
                obj_f.write("v {} {} {} {} {} {} \n".format(mesh.vertices[i][0], mesh.vertices[i][1], mesh.vertices[i][2], mesh.colors[i][0], mesh.colors[i][1], mesh.colors[i][2]))
        for i in range(len(mesh.texcoords)):
            obj_f.write("vt {} {}\n".format(mesh.texcoords[i][0], 1.0-mesh.texcoords[i][1]))
        
        obj_f.write("usemtl FaceTexture\n")
        for i in range(len(mesh.tvi)):
            vi = mesh.tvi[i]
            if len(mesh.tti)==0:
                assert (len(mesh.texcoords)==len(mesh.vertices))
                obj_f.write("f {}/{} {}/{} {}/{}\n".format(vi[0]+1,vi[0]+1,vi[1]+1,vi[1]+1,vi[2]+1,vi[2]+1))
            else:
                ti = mesh.tti[i]
                obj_f.write("f {}/{} {}/{} {}/{}\n".format(vi[0]+1,ti[0]+1,vi[1]+1,ti[1]+1,vi[2]+1,ti[2]+1))
        
        texture_filename = filename[:-3]+'texture.png'
        with open(mtl_filename, 'w') as mtl_f:
            mtl_f.write("newmtl FaceTexture\n")
            mtl_f.write("map_Kd %s\n" % texture_filename)

cpdef np.ndarray[float, ndim=2] get_rotation_matrix(float pitch, float yaw, float roll):
    cdef np.ndarray[float, ndim=2] Rx, Ry, Rz, R

    pitch = np.deg2rad(pitch)
    yaw = np.deg2rad(yaw)
    roll = np.deg2rad(roll)

    Rx = np.array([[1,0,0],[0,cos(pitch),-1*sin(pitch)],[0,sin(pitch),cos(pitch)]], dtype=np.float32)
    Ry = np.array([[cos(yaw),0,sin(yaw)],[0,1,0],[-1*sin(yaw),0,cos(yaw)]], dtype=np.float32)
    Rz = np.array([[cos(roll),-1*sin(roll),0],[sin(roll),cos(roll),0],[0,0,1]], dtype=np.float32)
    R = Rz.dot(Ry.dot(Rx)) # Rz*Ry*Rx
    
    return R.astype(np.float32)

def render_colors_core(np.ndarray[float, ndim=3, mode = "c"] image not None, 
                np.ndarray[float, ndim=2, mode = "c"] vertices not None, 
                np.ndarray[int, ndim=2, mode="c"] triangles not None, 
                np.ndarray[float, ndim=2, mode = "c"] colors not None, 
                np.ndarray[float, ndim=2, mode = "c"] depth_buffer not None,
                int nver, int ntri,
                int h, int w, int c
                ):   
    _render_colors_core(
        <float*> np.PyArray_DATA(image), <float*> np.PyArray_DATA(vertices), <int*> np.PyArray_DATA(triangles),  
        <float*> np.PyArray_DATA(colors), 
        <float*> np.PyArray_DATA(depth_buffer),
        nver, ntri,
        h, w, c)

cpdef np.ndarray[float,ndim=3] render_colors(np.ndarray[float,ndim=2] vertices, np.ndarray[int,ndim=2] triangles, np.ndarray[float,ndim=2] colors, int h, int w, int c, np.ndarray[float, ndim=3] BG):
    cdef np.ndarray[float, ndim=3] image
    if BG is None:
        image = np.zeros((h,w,c), dtype=np.float32)
    else:
        assert (BG.shape[0]==h and BG.shape[1]==w and BG.shape[2]==c)
        image = BG
    depth_buffer = np.zeros([h,w], dtype=np.float32, order='C') - 999999.

    # Change orders --> C-contiguous order (column major)
    _render_colors_core(<float*> np.PyArray_DATA(image), <float*> np.PyArray_DATA(vertices), <int*> np.PyArray_DATA(triangles), <float*> np.PyArray_DATA(colors), <float*> np.PyArray_DATA(depth_buffer), vertices.shape[0], triangles.shape[0], h, w, c)

    return image

cpdef np.ndarray[float, ndim=2] load_uv_coords(str filename):
    uv_map = io.loadmat(filename)
    return uv_map['UV'].astype(np.float32)

cpdef np.ndarray[float, ndim=2] generate_uv_map(Mesh mesh, float pitch, float yaw, float roll, list translation, int uv_width=512, int uv_height=512, np.ndarray background=None):
    cdef np.ndarray[float, ndim=2] vertices = np.array(mesh.vertices, dtype=np.float32)
    cdef np.ndarray[float, ndim=2] colors = np.array(mesh.colors, dtype=np.float32)
    cdef np.ndarray[int, ndim=2] triangles = np.array(mesh.tvi, dtype=np.int32)
    cdef np.ndarray[float, ndim=2] uv_coords, R
    cdef np.ndarray[float, ndim=1] T
    cdef np.ndarray[float, ndim=3] uv_texture_map
    cdef float S
    
    print('ready')
    colors = colors/np.max(colors)
    print('color')
    S = 180/(np.max(vertices[:,1]) - np.min(vertices[:,1]))  # Scale factor
    print('Scale')
    R = get_rotation_matrix(pitch, yaw, roll)  # Rotation matrix
    print('Rotation')
    T = np.array(translation, dtype=np.float32)  # 3D translation vectors
    print('Translation')
    # Scale*(V*Rotation^T) + Translation
    transformed_vertices = S * vertices.dot(R.T) + T
    print('transformed_vertices')

    # Processing UV
    uv_coords = load_uv_coords("BFM_UV.mat")
    print('loaded uv map')
    uv_coords[:,0] = uv_coords[:,0]*(uv_width-1)
    uv_coords[:,1] = uv_coords[:,1]*(uv_height-1)
    uv_coords[:,1] = uv_height - uv_coords[:,1] - 1
    uv_coords = np.hstack((uv_coords, np.zeros((uv_coords.shape[0],1), dtype=np.float32)))
    print('ready uv_coords')
    print(len(triangles), len(triangles[0]))
    uv_texture_map = render_colors(uv_coords, triangles, colors, uv_height, uv_width, 3, background)

    return uv_texture_map
    

cpdef np.ndarray[float, ndim=2] rescale_pca_basis(np.ndarray[float, ndim=2] orthonormal_basis, np.ndarray[float, ndim=1] eigenvalues):
    cdef np.ndarray[float, ndim=2] rescaled_basis
    cdef np.ndarray[float, ndim=1] sqrt_of_eigenvalues
    cdef int basis
    rescaled_basis = np.zeros_like(orthonormal_basis)
    sqrt_of_eigenvalues = np.sqrt(eigenvalues)
    for basis in range(orthonormal_basis.shape[1]):
        rescaled_basis[:,basis] = orthonormal_basis[:,basis]*sqrt_of_eigenvalues[basis]
    
    return rescaled_basis

cpdef np.ndarray[float, ndim=2] normalise_pca_basis(np.ndarray[float, ndim=2] rescaled_basis, np.ndarray[float, ndim=1] eigenvalues):
    cdef np.ndarray[float, ndim=2] orthonormal_basis
    cdef np.ndarray[float, ndim=1] sqrt_of_eigenvalues
    cdef int basis
    orthonormal_basis = np.zeros_like(rescaled_basis)
    sqrt_of_eigenvalues = np.sqrt(eigenvalues)
    for basis in range(rescaled_basis.shape[1]):
        orthonormal_basis[:,basis] = rescaled_basis[:,basis] / sqrt_of_eigenvalues[basis]
    
    return orthonormal_basis


cpdef Mesh sample_to_mesh(np.ndarray[float, ndim=1] shape_instance, np.ndarray[float, ndim=1] color_instance, list tvi, list tci, list texture_coordinates=[], list texture_triangle_indices=[]):
    cdef int num_vertices = shape_instance.shape[0] // 3, i
    cdef Mesh mesh

    mesh = Mesh()

    assert (shape_instance.shape[0]==color_instance.shape[0] or color_instance==None)
    assert (len(texture_coordinates)==0 or len(texture_coordinates)==(shape_instance.shape[0]//3) or len(texture_triangle_indices)!=0)

    for i in range(num_vertices):
        mesh.vertices.append([shape_instance[i*3], shape_instance[i*3+1], shape_instance[i*3+2]])
    
    if color_instance.shape[0] > 0:
        for i in range(num_vertices):
            mesh.colors.append([clamp(color_instance[i*3],0.0,1.0), clamp(color_instance[i*3+1],0.0,1.0), clamp(color_instance[i*3+2],0.0,1.0)])
    
    # Assign the triangle lists
    mesh.tvi = tvi
    # tci will be empty in case of a shape-only model
    mesh.tci = tci

    if len(texture_coordinates)!=0:
        for i in range(len(texture_coordinates)):
            mesh.texcoords.append([texture_coordinates[i][0], texture_coordinates[i][1]])
        if len(texture_triangle_indices)!=0:
            mesh.tti = texture_triangle_indices
    
    return mesh
