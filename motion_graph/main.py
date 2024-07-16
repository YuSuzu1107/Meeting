import sys
import os

# 一つ上の階層のパスを取得
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# 親ディレクトリをsys.pathに追加
sys.path.append(parent_dir)

# 他のファイルにある関数のimport
from bvh_io.bvh_loader import BVHLoader
from slerp.slerp import generate_interpolated_frames_bvh
from motion_graph import compute_similarity_matrix, display_similarity_matrix, build_motion_graph, display_motion_graph


bvh_file = os.path.abspath(os.path.join('./data/1_wayne_0_1_1.bvh'))
loader = BVHLoader(bvh_file)
loader.load()

frames1 = loader.frames[:100]
frames2 = loader.frames[-100:]

root = loader.root

window_size = 20  # 連続フレームの数
frames1_start = 0
frames1_end = 99 - window_size
frames2_start = 150 
frames2_end = 249 - window_size
threshold = 3.0   # 類似度の閾値

similarity_matrix = compute_similarity_matrix(frames1, frames2, root, window_size)
display_similarity_matrix(similarity_matrix, frames1_start, frames2_start)

motion_graph, transition_start, transition_end = build_motion_graph(similarity_matrix, threshold, frames1_start, frames1_end, frames2_start, frames2_end)
print(transition_start, transition_end)
display_motion_graph(motion_graph)

num_interpolated_frames = 20  # 生成したい中間フレーム数
output_file = os.path.abspath(os.path.join('./data/motion_graph_data.bvh'))
generate_interpolated_frames_bvh(bvh_file, transition_start + 1, len(loader.frames) - transition_end, num_interpolated_frames, output_file)