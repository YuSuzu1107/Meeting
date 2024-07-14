import numpy as np
from scipy.spatial.transform import Rotation as R

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