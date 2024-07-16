import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import sys
import os

# 一つ上の階層のパスを取得
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# 親ディレクトリをsys.pathに追加
sys.path.append(parent_dir)

# 他のファイルにある関数のimport
from bvh_io.bvh_loader import BVHLoader, Node
from skeleton_animation import update_skeleton

# BVHファイルのロード
bvh_file_path = os.path.abspath(os.path.join('./data/1_wayne_0_1_1.bvh'))
loader = BVHLoader(bvh_file_path)
loader.load()

root = loader.root
frames = loader.frames

# 3Dプロットと軸の設定
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-1, 1)
ax.set_ylim(0, 2)
ax.set_zlim(-1, 1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# アニメーションの設定
ani = animation.FuncAnimation(fig, update_skeleton, frames=len(frames), fargs=(frames, ax, root), interval=loader.frame_time * 50)
plt.show()
