import numpy as np
from scipy.spatial.transform import Rotation as R
import sys
import os

# 一つ上の階層のパスを取得
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# 親ディレクトリをsys.pathに追加
sys.path.append(parent_dir)

# 他のファイルにある関数のimport
from bvh_io.bvh_loader import BVHLoader, Node
from obj_io.obj_loader import OBJLoader
from obj_io.obj_exporter import OBJExporter
from lbs import get_all_joints, calculate_weights, set_rest_pose, compute_global_transform_matrix_1, compute_global_transform_matrix_2, linear_blend_skinning


bvh_file_path = os.path.abspath(os.path.join('./data/1_wayne_0_1_1.bvh'))
obj_file_path = os.path.abspath(os.path.join('./data/Male.obj'))
output_file_path = os.path.abspath(os.path.join('./data/lbs'))

# bvhファイルの読み込み
bvh_loader = BVHLoader(bvh_file_path)
bvh_loader.load()

# objファイルの読み込み
vertices, faces, normals, texture_coords, objects, smoothing_groups, lines = OBJLoader(obj_file_path)

# ジョイントのリストを取得
joints = get_all_joints(bvh_loader.root)

# 重みの計算
weights, indices = calculate_weights(vertices, joints, bvh_loader.root)

# フレームデータの取得
frames = bvh_loader.frames

# レストポーズフレームの取得
rest_pose_frames = set_rest_pose(bvh_loader)

# レストポーズフレームのグローバル変換行列の計算
_, _, rest_pose_transforms = compute_global_transform_matrix_1(bvh_loader.root, rest_pose_frames[0], 0, global_transforms={})
rest_pose_inverse_transforms = {k: np.linalg.inv(v) for k, v in rest_pose_transforms.items()}

# 行列Bをjoint_indexと関連付けて定義
B_matrices = rest_pose_inverse_transforms

# 各フレームごとに変換を行い、OBJファイルとして出力
for frame_idx, frame in enumerate(frames):
    # 通常フレームのグローバル変換行列の計算
    _, _, global_transforms = compute_global_transform_matrix_2(bvh_loader.root, frame, 0, global_transforms={})
    M_matrices = global_transforms

    # BとMの行列の積の計算 (M @ B)
    combined_transforms = {index: M_matrices[index] @ B_matrices[index] for index in M_matrices}

    # リニアブレンドスキニングを適用して頂点を変換
    transformed_vertices = linear_blend_skinning(vertices, combined_transforms, weights, indices)

    # OBJファイルの出力
    OBJExporter(f"{output_file_path}/{frame_idx}.obj", transformed_vertices, faces, normals, texture_coords, objects, smoothing_groups, lines)