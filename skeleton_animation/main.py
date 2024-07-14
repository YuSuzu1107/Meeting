import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import matplotlib.animation as animation
import sys
import os

# 一つ上の階層のパスを取得
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# 親ディレクトリをsys.pathに追加
sys.path.append(parent_dir)

# 他のファイルにある関数のimport
from bvh_io.bvh_loader import BVHLoader, Node

# スケルトンの描画
def draw_skeleton(ax, node, parent_position=None):
    if parent_position is not None and not np.all(parent_position == 0):
        # 親ノードと子ノードを青い線で結ぶ
        ax.plot([parent_position[0], node.position[0]], 
                [parent_position[1], node.position[1]], 
                [parent_position[2], node.position[2]], 'b-')

    for child in node.children:
        draw_skeleton(ax, child, node.position)

# ノードの位置と回転を更新する関数
def update_node_position(node, frame_data, index, parent_position=[0,0,0], parent_rotation=[0,0,0], is_root=False):

    if node.channels:
        pos_order = list()
        rot_order = list() 
        axis_order = ''
        for axis in node.channels:
            if  axis == "Xposition" and index < len(frame_data):
                pos_order.append(frame_data[index])
                index += 1
            if  axis == "Yposition" and index < len(frame_data):
                pos_order.append(frame_data[index])
                index += 1
            if  axis == "Zposition" and index < len(frame_data):
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
            node.position = np.array([x_pos, y_pos, z_pos])
            
            # 初期回転の計算
            global_rotation = R.from_euler(axis_order[::-1], rot_order[::-1], degrees=True)

        # Joint の場合の処理
        else:
            # ジョイントが6チャンネルの場合
            if len(pos_order) == 3:
                local_position = np.array(pos_order)
            else:
                local_position = np.array(node.offset)
    
            # 回転の計算
            local_rotation = R.from_euler(axis_order[::-1], rot_order[::-1], degrees=True)
            global_rotation = parent_rotation * local_rotation

            # 位置の計算
            node.position = parent_position + parent_rotation.apply(local_position)
    
    
    # End Site の処理
    else:
        global_rotation = parent_rotation
        node.position = parent_position + parent_rotation.apply(np.array(node.offset))

    for child in node.children:
        index = update_node_position(child, frame_data, index, node.position, global_rotation)
    return index

# アニメーションのフレームを更新する関数
def update_skeleton(num, frames, ax, root):
    ax.clear()
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 2)
    ax.set_zlim(-1, 1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    frame_data = frames[num]
    index = 0
    index = update_node_position(root, frame_data, index, is_root=True)
    draw_skeleton(ax, root, root.position)

# BVHファイルのロード
path = os.path.abspath(os.path.join('./data/1_wayne_0_1_1.bvh'))
loader = BVHLoader(path)
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
