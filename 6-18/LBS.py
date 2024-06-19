import numpy as np
from scipy.spatial.transform import Rotation as R
from bvh_loader import BVHLoader, Node
from obj_loader import OBJLoader
from obj_exporter import OBJExporter

# ジョイントの取得
def get_all_joints(joint):
    joints = [joint]
    for child in joint.children:
        joints.extend(get_all_joints(child))
    return joints

# レストポーズの計算
def set_rest_pose(bvh_loader):
    joints = get_all_joints(bvh_loader.root)
    rest_pose_frames = []
    for frame in bvh_loader.frames:
        rest_frame = frame.copy()  # フレームのコピーを作成
        idx = 0
        for joint in joints:
            for channel in ['Xposition', 'Yposition', 'Zposition', 'Xrotation', 'Yrotation', 'Zrotation']:
                if channel in joint.channels:
                    if idx >= len(rest_frame):
                        break  # モーションフレームの列数を超えたらループを終了
                    rest_frame[idx] = 0
                    idx += 1
            else:
                continue
            break  # 内部のループがブレイクされた場合、外部のループも終了
        rest_pose_frames.append(rest_frame)
    return rest_pose_frames

# オイラー角を用いた回転行列の計算
def euler_to_rotation_matrix(euler_angles, order='XYZ'):
    r = R.from_euler(order[::-1], euler_angles[::-1], degrees=True)
    return r.as_matrix()

# グローバル変換行列を計算
def compute_global_transform_matrix(joint, frame_data, index, parent_transform=np.eye(4), joint_index=0, global_transforms={}):

    # ローカルな並行移動行列
    translation = np.eye(4)
    translation[:3, 3] = joint.offset
    
    # ローカルな回転行列
    rotation_matrix = np.eye(4)
    if 'Xrotation' in joint.channels or 'Yrotation' in joint.channels or 'Zrotation' in joint.channels:
        rot_order = []
        axis_order = ''
        for axis in joint.channels:
            if axis == "Xrotation" and index < len(frame_data):
                rot_order.append(frame_data[index])
                index += 1
                axis_order += 'x'
            if axis == "Yrotation" and index < len(frame_data):
                rot_order.append(frame_data[index])
                index += 1
                axis_order += 'y'
            if axis == "Zrotation" and index < len(frame_data):
                rot_order.append(frame_data[index])
                index += 4
                axis_order += 'z'
        if rot_order and axis_order:
            rotation_matrix[:3, :3] = euler_to_rotation_matrix(rot_order, axis_order.upper())
    
    # ローカル変換行列
    local_transform = translation @ rotation_matrix

    # グローバル変換行列の計算
    global_transform = parent_transform @ local_transform

    # グローバル変換行列とそのインデックスを保存
    global_transforms[joint_index] = global_transform
 
    # 子ジョイントに対して再帰的に計算
    for child in joint.children:
        joint_index += 1
        index, joint_index, global_transforms = compute_global_transform_matrix(child, frame_data, index, global_transform, joint_index, global_transforms)
        
    return index, joint_index, global_transforms

# 親子関係の把握
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

# レストポーズにおけるグローバル座標の計算
def compute_rest_pose_global_positions(joint, parent_position=np.zeros(3)):
    joint.global_rest_position = parent_position + joint.offset
    for child in joint.children:
        compute_rest_pose_global_positions(child, joint.global_rest_position)
        
# 重みの計算
def calculate_weights(vertices, joints, root, c=16):
    num_vertices = len(vertices)
    weights = np.zeros((num_vertices, 4)) 
    indices = np.zeros((num_vertices, 4), dtype=int)

    epsilon = 1e-5  # ゼロ除算を防ぐための微小値

    # レストポーズのグローバル座標を計算
    compute_rest_pose_global_positions(root)
    
    for i, vertex in enumerate(vertices):
        vertex = np.array(vertex)
        joint_distances = []

        for j, joint in enumerate(joints):
           # print(j)
           # print(joint.name)
            parent_joint = find_parent(joint, root)
            P0 = parent_joint.global_rest_position if parent_joint else joint.global_rest_position
            P1 = joint.global_rest_position
            #print(P0)
            #print(P1)
            V1 = P1 - P0
            V2 = vertex - P0
            norm_V1 = np.dot(V1, V1)
            if norm_V1 < epsilon:
                norm_V1 = epsilon
            t = np.dot(V1, V2) / norm_V1 if norm_V1 > epsilon else 0  # tの計算
            t = np.clip(t, 0, 1)
            P = P0 + t * V1
            d = np.linalg.norm(vertex - P)
            if not np.isnan(d):
                joint_distances.append((d, j))

        # 距離が小さい順に並び替え
        joint_distances.sort()

        # 上位4つのジョイントを選択
        top_joint_distances = joint_distances[:4]
    
        # 合計が1となるように正規化された重みを計算
        inverse_distances = [(d + 1) ** -c for d, j in top_joint_distances]
        total_inverse_distance = sum(inverse_distances)
        for k, (d, j) in enumerate(top_joint_distances):
            weights[i, k] = inverse_distances[k] / total_inverse_distance
            indices[i, k] = j  # ジョイントのインデックスを保存
            #print(weights)
            #print(indices)

    return weights, indices

# リニアブレンドスキニング関数
def linear_blend_skinning(vertices, combined_transforms, weights, indices):
    transformed_vertices = np.zeros_like(vertices)
    for i, vertex in enumerate(vertices):
        blended_vertex = np.zeros(3)
        for j in range(4):
            bone_matrix = combined_transforms[indices[i, j]]
            weight = weights[i, j]
            vertex_homogeneous = np.append(vertex, 1)
            transformed_vertex = (bone_matrix @ vertex_homogeneous)[:3]
            blended_vertex += weight * transformed_vertex
        transformed_vertices[i] = blended_vertex
    return transformed_vertices

# メイン関数
def main():
    bvh_file_path = '/Users/yu/Desktop/岩本さん/data_for_skinning/1_wayne_0_1_1.bvh'
    obj_file_path = '/Users/yu/Desktop/岩本さん/data_for_skinning/Male.obj'
    output_file_path = '/Users/yu/Desktop/岩本さん/data_for_skinning/LBS/'

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
    _, _, rest_pose_transforms = compute_global_transform_matrix(bvh_loader.root, rest_pose_frames[0], 3, global_transforms={})
    rest_pose_inverse_transforms = {k: np.linalg.inv(v) for k, v in rest_pose_transforms.items()}
    #print(rest_pose_transforms)

    # 行列Bをjoint_indexと関連付けて定義
    B_matrices = rest_pose_inverse_transforms
    #M_matrices = global_transforms

    # 各フレームごとに変換を行い、OBJファイルとして出力
    for frame_idx, frame in enumerate(frames):
        # 通常フレームのグローバル変換行列の計算
        _, _, global_transforms = compute_global_transform_matrix(bvh_loader.root, frame, 0, global_transforms={})
        M_matrices = global_transforms

        # BとMの行列の積の計算 (M @ B)
        combined_transforms = {index: M_matrices[index] @ B_matrices[index] for index in M_matrices}

        # リニアブレンドスキニングを適用して頂点を変換
        transformed_vertices = linear_blend_skinning(vertices, combined_transforms, weights, indices)

        # OBJファイルの出力
        OBJExporter(f"{output_file_path}_{frame_idx}.obj", transformed_vertices, faces, normals, texture_coords, objects, smoothing_groups, lines)


if __name__ == "__main__":
    main()
