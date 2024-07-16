import sys
import os

# 一つ上の階層のパスを取得
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# 親ディレクトリをsys.pathに追加
sys.path.append(parent_dir)

# 他のファイルにある関数のimport
from bvh_io.bvh_loader import BVHLoader, Node
from bvh_io.bvh_exporter import BVHExporter
from slerp import generate_interpolated_frames_bvh

bvh_file_path = os.path.abspath(os.path.join('./data/1_wayne_0_1_1.bvh'))
start_frame_count = 100  # 最初の100フレーム
end_frame_count = 100  # 最後の100フレーム
num_interpolated_frames = 20  # 生成したい中間フレーム数
output_file = os.path.abspath(os.path.join('./data/slerp_data.bvh'))

generate_interpolated_frames_bvh(bvh_file_path, start_frame_count, end_frame_count, num_interpolated_frames, output_file)
