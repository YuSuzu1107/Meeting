from __future__ import annotations
import sys
import os

# 一つ上の階層のパスを取得
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# 親ディレクトリをsys.pathに追加
sys.path.append(parent_dir)

# 他のファイルにある関数のimport
from bvh_io.bvh_loader import BVHLoader, Node
from motion_matching import FeatureVectorCalculator, MotionDataBase

# BVHファイルのディレクトリ設定とデータベースの構築
bvh_directory = os.path.abspath(os.path.join('./data/walk_run'))
database = MotionDataBase(bvh_directory)
database.load_bvh_files()
database.save_database('pose_vectors.npy', 'feature_vectors.npy')

# クエリとして使用するBVHファイルのロード
loader = BVHLoader(os.path.join('./data/lafan1/walk1_subject1.bvh'))  # 使用したいBVHファイルをロード
loader.load()

# FeatureVectorCalculatorの初期化
feature_calculator = FeatureVectorCalculator(loader, loader.frame_time)

# 結果のBVHデータを保存するリスト
result_pose_vectors = []

# クエリとして使用するBVHファイルの全フレームに対して処理を繰り返す
for frame_idx in range(len(loader.frames)):
    # 各フレームのクエリベクトルを作成
    query_vector = feature_calculator.calculate_feature_vector(frame_idx)

    # 最も近いフレームデータをデータベースから検索
    closest_frame_data = database.find_closest_frame_data(query_vector)

    # 検索結果を結果のBVHデータリストに追加
    result_pose_vectors.append(closest_frame_data)

# 結果のBVHデータをファイルとしてエクスポート
output_bvh_path = os.path.abspath(os.path.join('./data/motion_matching_data_3.bvh'))
database.export_bvh(output_bvh_path, result_pose_vectors)