from __future__ import annotations
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
from bvh_io.bvh_exporter import BVHExporter

class PoseVectorCalculator:
    def __init__(self, bvh_loader, frame_time):
        """
        初期化

        :param bvh_loader: BVHLoaderオブジェクト
        :param frame_time: フレーム間の時間差
        """
        self.bvh_loader = bvh_loader
        self.frame_time = frame_time
        self.frames = len(bvh_loader.frames)
        self.joints = self.count_joints(bvh_loader.root)  # 正しいジョイント数をカウント
        self.joint_positions, self.joint_rotations = self.get_joint_positions_and_rotations()
        self.joint_velocities = self.calculate_joint_velocities()
        self.joint_rot_velocities = self.calculate_joint_rot_velocities()
        self.root_translation_velocities = self.calculate_root_translation_velocities()
        self.root_rotation_velocities = self.calculate_root_rotation_velocities()

    def count_joints(self, node):
        """
        再帰的にジョイントの数をカウントする関数

        :param node: 現在のジョイントノード
        :return: ジョイントの総数
        """
        count = 1  # 自身のジョイントをカウント
        for child in node.children:
            count += self.count_joints(child)
        return count

    def get_joint_positions_and_rotations(self):
        """
        各フレームのジョイントの位置と回転を計算する。

        :return: ジョイントの位置と回転 (frames, joints * 3) と (frames, joints * 4)
        """
        joint_positions = np.zeros((self.frames, self.joints * 3))
        joint_rotations = np.zeros((self.frames, self.joints * 4))
        
        for i in range(self.frames):
            frame_data = self.bvh_loader.frames[i]
            positions, rotations = self.get_global_positions_and_rotations(frame_data)
            
            if len(positions) != self.joints or len(rotations) != self.joints:
                raise ValueError(f"Frame {i}: Expected {self.joints} joints, but got {len(positions)} positions and {len(rotations)} rotations")
            
            for j, pos in enumerate(positions):
                joint_positions[i, j*3:(j+1)*3] = pos
            for j, rot in enumerate(rotations):
                joint_rotations[i, j*4:(j+1)*4] = rot.as_quat()
        
        return joint_positions, joint_rotations

    def get_global_positions_and_rotations(self, frame):
        positions = []
        rotations = []
        index = 0

        def collect_global_positions_and_rotations(node, frame_data, parent_position=np.zeros(3), parent_rotation=R.from_quat([0, 0, 0, 1])):
            nonlocal index

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

            # 位置の計算
            if pos_order:
                local_position = np.array(pos_order)
            else:
                local_position = np.array(node.offset)
            global_position = parent_position + parent_rotation.apply(local_position)
            
            # 回転の計算
            if rot_order:
                local_rotation = R.from_euler(axis_order[::-1], rot_order[::-1], degrees=True)
                global_rotation = parent_rotation * local_rotation
            else:
                global_rotation = parent_rotation

            # 結果をリストに追加
            positions.append(global_position)
            rotations.append(global_rotation)

            for child in node.children:
                collect_global_positions_and_rotations(child, frame_data, global_position, global_rotation)

        collect_global_positions_and_rotations(self.bvh_loader.root, frame)
        return positions, rotations

    def calculate_joint_velocities(self):
        """
        各フレームのジョイントの位置の速度を計算する。

        :return: ジョイントの位置の速度 (frames, joints * 3)
        """
        joint_velocities = np.zeros((self.frames, self.joints * 3))
        
        for i in range(1, self.frames):
            joint_velocities[i] = (self.joint_positions[i] - self.joint_positions[i - 1]) / self.frame_time
        
        # すべてのフレームに対して補正を適用
        for i in range(1, self.frames - 3):  # 端のフレームを除外して補正
            joint_velocities[i] = joint_velocities[i + 1] - (joint_velocities[i + 3] - joint_velocities[i + 2])

        return joint_velocities

    def calculate_joint_rot_velocities(self):
        """
        各フレームのジョイントの回転の速度を計算する。

        :return: ジョイントの回転の速度 (frames, joints * 3)
        """
        joint_rot_velocities = np.zeros((self.frames, self.joints * 3))
        
        for i in range(1, self.frames):
            for j in range(self.joints):
                current_rotation = R.from_quat(self.joint_rotations[i, j*4:(j+1)*4])
                previous_rotation = R.from_quat(self.joint_rotations[i - 1, j*4:(j+1)*4])
                relative_rotation = current_rotation * previous_rotation.inv()
                angle = relative_rotation.magnitude()
                axis = relative_rotation.as_rotvec() / angle if angle != 0 else np.zeros(3)
                joint_rot_velocities[i, j*3:(j+1)*3] = (axis * angle) / self.frame_time

        # すべてのフレームに対して補正を適用
        for i in range(1, self.frames - 3):  # 端のフレームを除外して補正
            joint_rot_velocities[i] = joint_rot_velocities[i + 1] - (joint_rot_velocities[i + 3] - joint_rot_velocities[i + 2])

        return joint_rot_velocities

    def calculate_root_translation_velocities(self):
        """
        各フレームのルートの並進速度を計算する。

        :return: ルートの並進速度 (frames, 3)
        """
        root_translation_velocities = np.zeros((self.frames, 3))
        
        for i in range(1, self.frames):
            current_position = self.joint_positions[i, :3]  # ルートの位置
            previous_position = self.joint_positions[i - 1, :3]
            delta_position = current_position - previous_position
            root_translation_velocities[i] = delta_position / self.frame_time

        # すべてのフレームに対して補正を適用
        for i in range(1, self.frames - 3):  # 端のフレームを除外して補正
            root_translation_velocities[i] = root_translation_velocities[i + 1] - (root_translation_velocities[i + 3] - root_translation_velocities[i + 2])
            
        return root_translation_velocities

    def calculate_root_rotation_velocities(self):
        """
        各フレームのルートの回転速度を計算する。

        :return: ルートの回転速度 (frames, 3)
        """
        root_rotation_velocities = np.zeros((self.frames, 3))
        
        for i in range(1, self.frames):
            current_rotation = R.from_quat(self.joint_rotations[i, :4])
            previous_rotation = R.from_quat(self.joint_rotations[i - 1, :4])
            delta_rotation = current_rotation * previous_rotation.inv()

            angle = delta_rotation.magnitude()
            axis = delta_rotation.as_rotvec() / angle if angle != 0 else np.zeros(3)
            root_rotation_velocities[i] = axis * (angle / self.frame_time)
        
        # すべてのフレームに対して補正を適用
        for i in range(1, self.frames - 3):  # 端のフレームを除外して補正
            root_rotation_velocities[i] = root_rotation_velocities[i + 1] - (root_rotation_velocities[i + 3] - root_rotation_velocities[i + 2])

        return root_rotation_velocities
    
    def calculate_pose_vector(self, frame):
        """
        指定されたフレームからポーズベクトルを計算する。

        :param frame: 現在のフレーム番号
        :return: ポーズベクトル
        """
        pose_vector = []

        # ジョイントのグローバル位置 (yt)
        pose_vector.extend(self.joint_positions[frame])

        # ジョイントのグローバル回転 (yr)
        pose_vector.extend(self.joint_rotations[frame])

        # ジョイントのローカル変換速度 (Ûyt)
        pose_vector.extend(self.joint_velocities[frame])

        # ジョイントのローカル回転速度 (Ûyr)
        pose_vector.extend(self.joint_rot_velocities[frame])

         # キャラクターのルートの変換速度 (Ûrt)
        root_velocity = self.root_translation_velocities[frame]  # ルートジョイントの速度
        pose_vector.extend(root_velocity)

        # キャラクターのルートの回転速度 (Ûrr)
        root_rot_velocity = self.root_rotation_velocities[frame]  # ルートジョイントの回転速度
        pose_vector.extend(root_rot_velocity)

        return pose_vector

    def extract_pose_vectors(self):
        """
        BVHファイルの各フレームからポーズベクトルを計算し、行列に保持する。

        :return: ポーズベクトルの行列
        """
        pose_vectors = []
        for i in range(self.frames):
            pose_vector = self.calculate_pose_vector(i)
            pose_vectors.append(pose_vector)
        
        return np.array(pose_vectors)
    
class FeatureVectorCalculator(PoseVectorCalculator):
    def __init__(self, bvh_loader, frame_time):
        super().__init__(bvh_loader, frame_time)
        self.foot_joints = self.find_foot_joints(bvh_loader.root, ['RightFoot', 'LeftFoot'])
        self.feature_std = None  # 標準偏差を保存するための変数

    def find_foot_joints(self, node, target_names):
        """
        指定された名前のジョイントのインデックスを見つける。

        :param node: 現在のジョイントノード
        :param target_names: 検索するジョイント名のリスト
        :return: インデックスのリスト
        """
        index_list = []
        current_index = 0

        def traverse(node):
            nonlocal current_index
            current_index += 1
            if node.name in target_names:
                index_list.append(current_index)
            for child in node.children:
                traverse(child)

        traverse(node)
        return index_list
    
    def calculate_feature_vector(self, frame):
        """
        指定されたフレームから特徴ベクトルを計算する。

        :param frame: 現在のフレーム番号
        :return: 特徴ベクトル
        """
        feature_vector = []

        # 現在の向きをクエリベクトルに追加
        current_orientation = R.from_quat(self.joint_rotations[frame, :4])
        current_direction = current_orientation.apply([1, 0, 0])[:2]  # XY平面の向き
        feature_vector.extend([float(val) for val in current_direction])

        # 将来の軌道位置 tt (20, 40, 60フレーム後)
        future_frames = [20, 40, 60]
        future_positions = []
        for f in future_frames:
            future_frame = min(frame + f, self.frames - 1)
            future_position = self.joint_positions[future_frame, :2] - self.joint_positions[frame, :2] # 2D (X, Y) のみ使用
            future_positions.extend([float(val) for val in future_position])
        feature_vector.extend(future_positions)

        # 将来の軌道方向 td (20, 40, 60フレーム後)
        future_directions = []
        for f in future_frames:
            future_frame = min(frame + f, self.frames - 1)
            future_orientation = R.from_quat(self.joint_rotations[future_frame, :4]) 
            delta_rotation = future_orientation * R.from_quat(self.joint_rotations[frame, :4]).inv()
            
            # XY平面における将来の軌道向きの2Dベクトル
            direction = delta_rotation.apply([1, 0, 0])[:2] # X軸方向のベクトルのXY成分
            
            future_directions.extend([float(val) for val in direction])

        feature_vector.extend(future_directions)

        # 足の位置 ft
        for j in self.foot_joints:
            foot_position = self.joint_positions[frame, j*3:(j+1)*3] 
            feature_vector.extend([float(val) for val in foot_position])
        
        # 足の速度 Ûft
        for j in self.foot_joints:
            foot_velocity = self.joint_velocities[frame, j*3:(j+1)*3] 
            feature_vector.extend([float(val) for val in foot_velocity])

        # 腰の速度 Ûht
        hip_velocity = self.root_translation_velocities[frame] # 3Dで計算
        feature_vector.extend([float(val) for val in hip_velocity])

        return feature_vector

    def extract_feature_vectors(self):
        """
        BVHファイルの各フレームから特徴ベクトルを計算し、行列に保持する。

        :return: 特徴ベクトルの行列
        """
        feature_vectors = []
        for i in range(self.frames):
            feature_vector = self.calculate_feature_vector(i)
            feature_vectors.append(feature_vector)
        
        feature_vectors = np.array(feature_vectors)

        return feature_vectors

# データベースの構築とファイルのロード
class MotionDataBase:
    # 初期化
    def __init__(self, bvh_directory):
        self.bvh_directory = bvh_directory
        self.pose_vectors = []
        self.feature_vectors = []
        self.frames_data = []
        self.file_indices = []  # 各フレームがどのファイルに属しているかの情報
        self.root = None
        self.frame_time = None

    # bvhファイルをまとめてロード
    def load_bvh_files(self):
        bvh_files = [f for f in os.listdir(self.bvh_directory) if f.endswith('.bvh')]
        for file_index, bvh_file in enumerate(bvh_files):
            # file_indexを渡す
            self.process_bvh(os.path.join(self.bvh_directory, bvh_file), file_index)

    # bvhファイルを処理
    def process_bvh(self, bvh_file_path, file_index):
        loader = BVHLoader(bvh_file_path)
        loader.load()
        frame_time = loader.frame_time
        self.frame_time = frame_time
        self.root = loader.root if self.root is None else self.root 

        pose_calculator = PoseVectorCalculator(loader, frame_time)
        feature_calculator = FeatureVectorCalculator(loader, frame_time)

        pose_vectors_matrix = pose_calculator.extract_pose_vectors()
        feature_vectors_matrix_1 = feature_calculator.extract_feature_vectors()

        self.pose_vectors.append(pose_vectors_matrix)
        self.feature_vectors.append(feature_vectors_matrix_1)
        
        # フレームデータとファイルインデックスを追加
        self.frames_data.extend(loader.frames)
        self.file_indices.extend([file_index] * len(loader.frames))

    # データベースとして保存
    def save_database(self, pose_vectors_path, feature_vectors_path):
        np.save(pose_vectors_path, np.vstack(self.pose_vectors))
        np.save(feature_vectors_path, np.vstack(self.feature_vectors))
    
    # データベースから全探索
    def find_closest_frame_data(self, query_vector):
        # クエリベクトルをNumPy配列に変換
        query_vector = np.array(query_vector)

        # 特徴ベクトルを行列にする
        feature_vectors_matrix = np.vstack(self.feature_vectors)
        # 最小となる行を見つけ、その行のindexを返す
        squared_distances = np.sum((feature_vectors_matrix - query_vector) ** 2, axis=1)
        closest_index = np.argmin(squared_distances)

        return self.frames_data[closest_index]

    # bvhファイルに出力
    def export_bvh(self, output_path, closest_pose_vectors):
        bvh_exporter = BVHExporter(self.root, closest_pose_vectors, self.frame_time)
        bvh_exporter.export(output_path)