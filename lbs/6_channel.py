import numpy as np
from scipy.spatial.transform import Rotation as R
import sys
import os

# 一つ上の階層のパスを取得
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# 親ディレクトリをsys.pathに追加
sys.path.append(parent_dir)

from bvh_io.bvh_loader import BVHLoader, Node
from obj_io.obj_loader import OBJLoader
from obj_io.obj_exporter import OBJExporter

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

# グローバル変換行列を計算(レストポーズ用)
def compute_global_transform_matrix_1(joint, frame_data, index, parent_position=np.array([0,0,0]), parent_rotation=R.from_euler('xyz', [0,0,0], degrees=True), is_root=False, joint_index=0, global_transforms={}):
    if joint.channels:
        pos_order = []
        rot_order = []
        axis_order = ''
        for axis in joint.channels:
            if axis == "Xposition" and index < len(frame_data):
                pos_order.append(frame_data[index])
                index += 1
            if axis == "Yposition" and index < len(frame_data):
                pos_order.append(frame_data[index])
                index += 1
            if axis == "Zposition" and index < len(frame_data):
                pos_order.append(frame_data[index])
                index += 1
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
                index += 1
                axis_order += 'z'
        
        # ルートノードの場合の処理
        if is_root:
            x_pos, y_pos, z_pos = pos_order

            # 初期位置の設定
            joint.position = np.array([x_pos, y_pos, z_pos])
            
            # 初期回転の計算
            global_rotation = R.from_euler(axis_order[::-1], rot_order[::-1], degrees=True)
        else:
            # 回転の計算
            local_rotation = R.from_euler(axis_order[::-1], rot_order[::-1], degrees=True)
            global_rotation = parent_rotation * local_rotation

            # 位置の計算
            joint.position = parent_position + parent_rotation.apply(np.array(joint.offset))
    else:
        global_rotation = parent_rotation
        joint.position = parent_position + parent_rotation.apply(np.array(joint.offset))

    # グローバル変換行列の計算
    translation = np.eye(4)
    translation[:3, 3] = joint.position
    
    rotation_matrix = np.eye(4)
    rotation_matrix[:3, :3] = global_rotation.as_matrix()
    
    global_transform = translation @ rotation_matrix

    # グローバル変換行列とそのインデックスを保存
    global_transforms[joint_index] = global_transform
    
    for child in joint.children:
        joint_index += 1
        index, joint_index, global_transforms = compute_global_transform_matrix_1(child, frame_data, index, joint.position, global_rotation, joint_index=joint_index, global_transforms=global_transforms)
        
    return index, joint_index, global_transforms

# グローバル変換行列を計算
def compute_global_transform_matrix_2(joint, frame_data, index, parent_position=np.array([0,0,0]), parent_rotation=R.from_euler('xyz', [0,0,0], degrees=True), is_root=False, joint_index=0, global_transforms={}):
    if joint.channels:
        pos_order = []
        rot_order = []
        axis_order = ''
        for axis in joint.channels:
            if axis == "Xposition" and index < len(frame_data):
                pos_order.append(frame_data[index])
                index += 1
            if axis == "Yposition" and index < len(frame_data):
                pos_order.append(frame_data[index])
                index += 1
            if axis == "Zposition" and index < len(frame_data):
                pos_order.append(frame_data[index])
                index += 1
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
                index += 1
                axis_order += 'z'
        
        # ルートノードの場合の処理
        if is_root:
            x_pos, y_pos, z_pos = pos_order

            # 初期位置の設定
            joint.position = np.array([x_pos, y_pos, z_pos])
            
            # 初期回転の計算
            global_rotation = R.from_euler(axis_order[::-1], rot_order[::-1], degrees=True)
        else:
            # ジョイントが6チャンネルの場合
            if len(pos_order) == 3:
                local_position = np.array(pos_order)
            else:
                local_position = np.array(joint.offset)
                
            # 回転の計算
            local_rotation = R.from_euler(axis_order[::-1], rot_order[::-1], degrees=True)
            global_rotation = parent_rotation * local_rotation

            # 位置の計算
            joint.position = parent_position + parent_rotation.apply(local_position)

    # End Site の処理
    else:
        global_rotation = parent_rotation
        joint.position = parent_position + parent_rotation.apply(np.array(joint.offset))

    # グローバル変換行列の計算
    translation = np.eye(4)
    translation[:3, 3] = joint.position
    
    rotation_matrix = np.eye(4)
    rotation_matrix[:3, :3] = global_rotation.as_matrix()
    
    global_transform = translation @ rotation_matrix

    # グローバル変換行列とそのインデックスを保存
    global_transforms[joint_index] = global_transform
    
    for child in joint.children:
        joint_index += 1
        index, joint_index, global_transforms = compute_global_transform_matrix_2(child, frame_data, index, joint.position, global_rotation, joint_index=joint_index, global_transforms=global_transforms)
        
    return index, joint_index, global_transforms

# 親子関係の把握
def find_child(joint, root):
    for child in root.children:
        if joint == child:
            return child
        result = find_child(joint, child)
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
            child_joint = None
            if joint.children:
                child_joint = joint.children[0]
            
            P0 = joint.global_rest_position
            P1 = child_joint.global_rest_position if child_joint else joint.global_rest_position
           
            # P0 と P1 が等しい場合は次のイテレーションに進む
            if np.array_equal(P0, P1):
                continue
            
            V1 = P1 - P0
            V2 = vertex - P0
            norm_V1 = np.dot(V1, V1)
            if norm_V1 < epsilon:
                norm_V1 = epsilon
            t = np.dot(V1, V2) / norm_V1 if norm_V1 > epsilon else 0  # tの計算
            t = np.clip(t, 0, 1)
            P = P0 + t * V1
            d = np.linalg.norm(vertex - P)
            if not np.isnan(d) and j != 0:
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
    bvh_file_path = '../data/1_wayne_0_1_1.bvh'
    obj_file_path = '../data/Male.obj'
    output_file_path = '../data/LBS_6_channel/'

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
    _, _, rest_pose_transforms = compute_global_transform_matrix_1(bvh_loader.root, rest_pose_frames[0], 3, global_transforms={})
    rest_pose_inverse_transforms = {k: np.linalg.inv(v) for k, v in rest_pose_transforms.items()}

    # 行列Bをjoint_indexと関連付けて定義
    B_matrices = rest_pose_inverse_transforms

    # 各フレームごとに変換を行い、OBJファイルとして出力
    for frame_idx, frame in enumerate(frames):
        # 通常フレームのグローバル変換行列の計算
        _, _, global_transforms = compute_global_transform_matrix_2(bvh_loader.root, frame, 3, global_transforms={})
        M_matrices = global_transforms

        # BとMの行列の積の計算 (M @ B)
        combined_transforms = {index: M_matrices[index] @ B_matrices[index] for index in M_matrices}

        # リニアブレンドスキニングを適用して頂点を変換
        transformed_vertices = linear_blend_skinning(vertices, combined_transforms, weights, indices)

        # OBJファイルの出力
        OBJExporter(f"{output_file_path}_{frame_idx}.obj", transformed_vertices, faces, normals, texture_coords, objects, smoothing_groups, lines)


if __name__ == "__main__":
    main()
