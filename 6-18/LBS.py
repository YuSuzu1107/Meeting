import numpy as np
from bvh_loader import BVHLoader
from scipy.spatial.transform import Rotation as R
from obj_loader import OBJLoader
from obj_exporter import OBJExporter

# オイラー角を用いた回転行列の計算
def euler_to_rotation_matrix(euler_angles, order='XYZ'):
    r = R.from_euler(order[::-1], euler_angles[::-1], degrees=True)
    return r

# グローバルな回転行列を計算
def compute_joint_to_model_matrix(joint, parent_rotation=R.identity()):
    parent_rotation = R.from_matrix(parent_rotation.as_matrix()[:3, :3])
    offset_transformed = parent_rotation.apply(joint.offset)
    
    # 並行移動行列
    translation = np.eye(4)
    translation[:3, 3] = offset_transformed
    
    # ローカルな回転行列
    rotation_matrix = np.eye(4)
    if 'rotation' in joint.channels: 
        rotation_matrix[:3, :3] = euler_to_rotation_matrix(joint.rotation, order='XYZ').as_matrix()
    
    transform = translation @ rotation_matrix
    return transform, parent_rotation


def compute_bone_matrices_recursive(joint, frame, parent_transform=np.eye(4), parent_rotation=R.identity()):
    M, current_rotation = compute_joint_to_model_matrix(joint, parent_rotation)
    M_inv = np.linalg.inv(M)  

    euler_angles = []
    for channel in ['Xrotation', 'Yrotation', 'Zrotation']:
        if channel in joint.channels:
            idx = joint.channels.index(channel)
            euler_angles.append(frame[idx])
        else:
            euler_angles.append(0)
    
    B = np.eye(4)
    B[:3, :3] = euler_to_rotation_matrix(euler_angles, order='XYZ').as_matrix()
    
    bone_matrices = [parent_transform @ B @ M_inv]  
    
    for child in joint.children:
        child_bone_matrices = compute_bone_matrices_recursive(child, frame, M, current_rotation)
        bone_matrices.extend(child_bone_matrices)
    
    return bone_matrices

def find_parent(joint, root):
    if joint == root:
        return None
    for child in root.children:
        if joint == child:
            return root
        result = find_parent(joint, child)
        if result:
            return result
    return None

# 重みの計算
def calculate_weights(vertices, joints, root, c=16):
    num_vertices = len(vertices)
    num_joints = len(joints)
    weights = np.zeros((num_vertices, num_joints))

    epsilon = 1e-5  # ゼロ除算を防ぐための微小値
    
    for i, vertex in enumerate(vertices):
        vertex = np.array(vertex)
        total_weight = 0
        for j, joint in enumerate(joints):
            parent_joint = find_parent(joint, root)
            P0 = parent_joint.global_transform[:3, 3] if parent_joint else joint.global_transform[:3, 3]
            P1 = joint.global_transform[:3, 3]
            V1 = P1 - P0
            V2 = vertex - P0
            norm_V1 = np.dot(V1, V1)
            if norm_V1 < epsilon:
                norm_V1 = epsilon
            t = np.dot(V1, V2) / norm_V1
            t = np.clip(t, 0, 1)
            P = P0 + t * V1
            d = np.linalg.norm(vertex - P)
            weight = (d + epsilon) ** -c
            weights[i, j] = weight
            total_weight += weight
        
        if total_weight > epsilon:
            weights[i, :] /= total_weight
    
    return weights

def linear_blend_skinning(vertices, bone_matrices, weights):
    transformed_vertices = np.zeros_like(vertices)
    for i, vertex in enumerate(vertices):
        blended_vertex = np.zeros(3)
        for bone_matrix, weight in zip(bone_matrices, weights[i]):
            vertex_homogeneous = np.append(vertex, 1)
            transformed_vertex = (bone_matrix @ vertex_homogeneous)[:3]
            blended_vertex += weight * transformed_vertex
        transformed_vertices[i] = blended_vertex
    return transformed_vertices

def get_all_joints(joint):
    joints = [joint]
    for child in joint.children:
        joints.extend(get_all_joints(child))
    return joints

def calculate_rest_pose(joint, parent_transform=np.eye(4)):
    joint_transform, _ = compute_joint_to_model_matrix(joint)
    joint.global_transform = parent_transform @ joint_transform

    for child in joint.children:
        calculate_rest_pose(child, joint.global_transform)

def set_rest_pose(bvh_loader):
    for frame in bvh_loader.frames:
        joints = get_all_joints(bvh_loader.root)
        for joint in joints:
            for channel in ['Xrotation', 'Yrotation', 'Zrotation']:
                if channel in joint.channels:
                    idx = joint.channels.index(channel)
                    frame[idx] = 0
    return bvh_loader

def main():
    bvh_file_path = '/Users/yu/Desktop/岩本さん/data_for_skinning/1_wayne_0_1_1.bvh'
    obj_file_path = '/Users/yu/Desktop/岩本さん/data_for_skinning/Male.obj'
    output_file_path = '/Users/yu/Desktop/岩本さん/data_for_skinning/LBS/'

    bvh_loader = BVHLoader(bvh_file_path)
    bvh_loader.load()
    bvh_loader = set_rest_pose(bvh_loader)  # レストポーズの設定
    vertices, faces, normals, texture_coords, objects, smoothing_groups, lines = OBJLoader(obj_file_path)

    joints = get_all_joints(bvh_loader.root)
    calculate_rest_pose(bvh_loader.root)  # レストポーズの計算

    weights = calculate_weights(vertices, joints, bvh_loader.root)

    rest_pose_bone_matrices = compute_bone_matrices_recursive(bvh_loader.root, [0]*len(bvh_loader.frames[0]))
    rest_pose_transformed_vertices = linear_blend_skinning(vertices, rest_pose_bone_matrices, weights)

    for frame_idx, frame in enumerate(bvh_loader.frames):
        current_bone_matrices = compute_bone_matrices_recursive(bvh_loader.root, frame)
        transformed_vertices = linear_blend_skinning(rest_pose_transformed_vertices, current_bone_matrices, weights)
        OBJExporter(f"{output_file_path}_{frame_idx}.obj", transformed_vertices, faces, normals, texture_coords, objects, smoothing_groups, lines)

if __name__ == "__main__":
    main()
