import numpy as np
from scipy.spatial.transform import Rotation as R
from bvh_loader import BVHLoader 
from bvh_exporter import BVHExporter

# 線形補間関数（位置用）
def lerp(v0, v1, t):
    return (1 - t) * np.array(v0) + t * np.array(v1)

# SLERP関数
def slerp(p0, p1, t):
    p0 = np.array(p0)
    p1 = np.array(p1)
    
    # ベクトルを正規化
    norm_p0 = np.linalg.norm(p0)
    norm_p1 = np.linalg.norm(p1)
    
    if norm_p0 == 0:
        return p1 * t
    if norm_p1 == 0:
        return p0 * (1 - t)
    
    q0 = p0 / norm_p0
    q1 = p1 / norm_p1

    # クォータニオンのドット積を計算
    dot_product = np.dot(q0, q1)
    
    # ドット積が負の場合、q1を反転させて最短経路を取る
    if dot_product < 0.0:
        q1 = -q1
        dot_product = -dot_product
    
    # ドット積を[-1, 1]の範囲にクリップ
    dot_product = np.clip(dot_product, -1.0, 1.0)
    
    # 角度を計算
    omega = np.arccos(dot_product)
    
    # 角度が小さい場合、線形補間にフォールバック
    if np.abs(omega) < 1e-10:
        return (1.0 - t) * p0 + t * p1
    
    # 補間の計算
    sin_omega = np.sin(omega)
    q0_component = np.sin((1.0 - t) * omega) / sin_omega
    q1_component = np.sin(t * omega) / sin_omega
    
    return q0_component * p0 + q1_component * p1

# ジョイントのローカル位置と回転を取得する関数
def get_local_positions_and_rotations(bvh, frame):
    positions = []
    rotations = []
    index = 0

    def collect_positions(node, frame_data):
        nonlocal index
        if node.name == "End Site":
            return
        
        pos_order = []
        rot_order = []
        axis_order = ''
        
        for axis in node.channels:
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
        
        positions.append(np.array(pos_order))
        rotations.append(R.from_euler(axis_order[::-1], rot_order[::-1], degrees=True))
        
        for child in node.children:
            collect_positions(child, frame_data)

    collect_positions(bvh.root, frame)
    return positions, rotations

# 中間フレームの生成
def generate_interpolated_frames_bvh(file, start_frame_count, end_frame_count, num_interpolated_frames, output_file):
    bvh = BVHLoader(file)
    bvh.load()
    
    start_frames = bvh.frames[:start_frame_count]
    end_frames = bvh.frames[-end_frame_count:]
    frame1_positions, frame1_rotations = get_local_positions_and_rotations(bvh, start_frames[-1])
    frame2_positions, frame2_rotations = get_local_positions_and_rotations(bvh, end_frames[0])
    print(frame1_positions)
    all_interpolated_frames = []

    # 最初の100フレームを追加
    all_interpolated_frames.extend(start_frames)

    # 中間フレームを生成して追加
    for i in range(num_interpolated_frames):
        t = i / (num_interpolated_frames - 1)  # tの範囲を0から1に正規化
        interpolated_frame_data = []
        for pos1, pos2, rot1, rot2 in zip(frame1_positions, frame2_positions, frame1_rotations, frame2_rotations):
            # 位置情報は線形補間を使用
            interpolated_position = lerp(pos1, pos2, t)
            interpolated_frame_data.extend(interpolated_position)
            
            if rot1 and rot2:
                # 自分で定義したSLERP関数を使ったクォータニオンの補間
                interpolated_rotation = slerp(rot1.as_quat(), rot2.as_quat(), t)
                interpolated_rotation = R.from_quat(interpolated_rotation).as_euler('xyz', degrees=True)
                interpolated_frame_data.extend(interpolated_rotation)
            
        all_interpolated_frames.append(interpolated_frame_data)

    # 最後の100フレームを追加
    all_interpolated_frames.extend(end_frames)

    exporter = BVHExporter(bvh.root, all_interpolated_frames, bvh.frame_time)
    exporter.export(output_file)

# 使用例
file = '/Users/yu/Desktop/Animation_Lecture/Meeting/6-24 SLERP/1_wayne_0_1_1.bvh'
start_frame_count = 100  # 最初の100フレーム
end_frame_count = 100  # 最後の100フレーム
num_interpolated_frames = 20  # 生成したい中間フレーム数
output_file = '/Users/yu/Desktop/Animation_Lecture/Meeting/6-24 SLERP/interpolated.bvh'

generate_interpolated_frames_bvh(file, start_frame_count, end_frame_count, num_interpolated_frames, output_file)
