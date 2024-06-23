import numpy as np
from scipy.spatial.transform import Rotation as R
from obj_loader import OBJLoader
from obj_exporter import OBJExporter

# SLERP関数
def slerp(v0, v1, t):
    v0 = np.array(v0)
    v1 = np.array(v1)

    # ベクトルを正規化
    u0 = v0 / np.linalg.norm(v0)
    u1 = v1 / np.linalg.norm(v1)

    dot_product = np.dot(u0, u1)
    if dot_product < 0.0:
        v1 = -v1
        dot_product = -dot_product
    dot_product = np.clip(dot_product, -1.0, 1.0)
    omega = np.arccos(dot_product)
    if np.abs(omega) < 1e-10:
        return (1.0 - t) * u0 + t * u1
    sin_omega = np.sin(omega)
    u0_component = np.sin((1.0 - t) * omega) / sin_omega
    u1_component = np.sin(t * omega) / sin_omega
    return u0_component * v0 + u1_component * v1

# 中間フレームの生成
def generate_interpolated_frames(file1, file2, num_frames, output_dir):
    vertices1, faces1, normals1, texture_coords1, objects1, smoothing_groups1, lines1 = OBJLoader(file1)
    vertices2, faces2, normals2, texture_coords2, objects2, smoothing_groups2, lines2 = OBJLoader(file2)
    
    assert len(vertices1) == len(vertices2), "OBJ files must have the same number of vertices"
    
    for i in range(num_frames):
        t = i / (num_frames - 1)  # tの範囲を0から1に正規化

        interpolated_vertices = []
        for v1, v2 in zip(vertices1, vertices2):
           
            interpolated_vertex = slerp(v1, v2, t)
        
            interpolated_vertices.append(interpolated_vertex)
        output_file = f"{output_dir}/_{i + 251}.obj"
        OBJExporter(output_file, interpolated_vertices, faces1, normals1, texture_coords1, objects1, smoothing_groups1, lines1)

# 使用例
file1 = '/Users/yu/Desktop/岩本さん/data_for_skinning/LBS1/_250.obj'
file2 = '/Users/yu/Desktop/岩本さん/data_for_skinning/LBS1/_0.obj'
num_frames = 100  # 生成したいフレーム数
output_dir = '/Users/yu/Desktop/岩本さん/Meeting/6-24 SLERP/SLERP'

generate_interpolated_frames(file1, file2, num_frames, output_dir)
