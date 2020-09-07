import h5py
import numpy as np
import eos, torch
import mesh_utils as utils
import cv2
import time
from scipy import io
import json

def load_bfm2019(bfm_2019_file):
    with h5py.File(bfm_2019_file, 'r') as hf:
        # The PCA shape model
        shape_mean = np.array(hf['shape/model/mean'])
        print(shape_mean.shape[0]//3)
        shape_orthogonal_pca_basis = np.array(hf['shape/model/pcaBasis'])
        shape_pca_variance = np.array(hf['shape/model/pcaVariance'])

        triangle_list = np.array(hf['shape/representer/cells'])

        shape_model = utils.PcaModel(shape_mean, shape_orthogonal_pca_basis, shape_pca_variance, triangle_list.transpose().tolist())

        # The PCA colour model
        color_mean = np.array(hf['color/model/mean'])
        color_orthogonal_pca_basis = np.array(hf['color/model/pcaBasis'])
        color_pca_variance = np.array(hf['color/model/pcaVariance'])
        
        color_model = utils.PcaModel(color_mean, color_orthogonal_pca_basis, color_pca_variance, triangle_list.transpose().tolist())

        # The PCA expression model
        expression_mean = np.array(hf['expression/model/mean'])
        expression_pca_basis = np.array(hf['expression/model/pcaBasis'])
        expression_pca_variance = np.array(hf['expression/model/pcaVariance'])

        expression_model = utils.PcaModel(expression_mean, expression_pca_basis, expression_pca_variance, triangle_list.transpose().tolist())
    
    return shape_model, color_model, expression_model

if __name__=='__main__':

    mat = io.loadmat('../Deep3DFaceReconstruction/output/015384.mat')
    print(mat['face_color'].shape)
    print(mat['coeff'].shape[1]-199)
    shape_model, color_model, expression_model = load_bfm2019('./bfm2019/model2019_face12.h5')
    # with open('model2019_textureMapping.json') as json_file:
    #     json_data = json.load(json_file)
    mm = utils.MorphableModel(shape_model, color_model, expression_model)
    mesh = mm.get_sample(np.random.randn(199).astype(np.float32), np.random.randn(199).astype(np.float32))
    utils.write_obj(mesh, 'doit.obj')

    # mesh = mm.get_mean()
    # start = time.time()
    # utils.write_obj(mesh, 'mean.obj')
    # #img = utils.generate_uv_map(mesh, 0.0, 0.0, 0.0, [0.0,1.0,1.0])
    # #print(time.time() - start)
    # #cv2.imwrite('uv_map.png', img*255)
    # utils.save_color_map(mesh, "mean.npy")
    # #utils.write_textured_obj(mesh, 'mean.obj')
    # # blendshapes = eos.morphablemodel.load_blendshapes('expression_blendshapes_3448.bin')
    # # print(len(blendshapes))

    # camera_distance = 2.732
    # elevation = 30
    # azimuth = 120
    # texture_size = 2

    