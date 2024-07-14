import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import matplotlib.animation as animation
import os

# bvhファイルのロード
# ノードのクラス
class Node:

    # 各ノードの初期化
    def __init__(self, name):
        self.name = name
        self.offset = None
        self.channels = []
        self.children = []
        self.position = np.zeros(3)

# ローダークラスの定義
class BVHLoader:

    # 初期化
    def __init__(self, file_path):
        self.file_path = file_path
        self.root = None
        self.frames = []
        self.frame_time = 0.0

    # ロード関数の定義
    def load(self):
        print(f"Loading BVH file from: {self.file_path}")
        with open(self.file_path, 'r') as file:
            lines = file.readlines()

        # 階層構造の解析
        self.parse_hierarchy_section(lines)

        # モーションデータの解析
        self.parse_motion_section(lines)

        # ロード結果の表示
        if self.root is not None:
            print("BVH file loaded successfully")
        else:
            print("Failed to load BVH file")

    # 階層構造の解析の概要
    def parse_hierarchy_section(self, lines):
        hierarchy_parsed = False
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if "HIERARCHY" in line:
                i += 1
                continue
            if not hierarchy_parsed:
                self.root, i = self.parse_hierarchy(lines, i)
                if self.root is None:
                    raise ValueError("Failed to parse hierarchy")
                print(f"Hierarchy parsed successfully with root: {self.root.name}")
                hierarchy_parsed = True
            i +=  1
    
    # 階層構造の解析の実行
    def parse_hierarchy(self, lines, i, level=0):
        line = lines[i].strip()
        parts = line.split()
        node = None

        if parts[0] == "ROOT" or parts[0] == "JOINT":
            node = Node(parts[1])
            i += 1  # スキップして { の行を読み取る
            if lines[i].strip() != "{":
                raise ValueError(f"Expected '{{' at line {i}, but found: {lines[i].strip()}")
            i += 1  # { の行をスキップ
            node.offset = list(map(float, lines[i].strip().split()[1:]))
            i += 1
            node.channels = lines[i].strip().split()[2:]
            i += 1

            # 深さ優先のノード解析のループ
            while i < len(lines):
                line = lines[i].strip()
                if line == '}':
                    i += 1  # '}' をスキップして次の行に進む
                    break
                elif "JOINT" in line or "End" in line:
                    # 関数を再帰的に呼び出し、子ノードを解析
                    child_node, i = self.parse_hierarchy(lines, i, level + 1)
                    node.children.append(child_node)
                else:
                    i += 1

        # 終端ノードの解析
        elif parts[0] == "End":
            node = Node("End Site")
            i += 1  # スキップして { の行を読み取る
            if lines[i].strip() != "{":
                raise ValueError(f"Expected '{{' at line {i}, but found: {lines[i].strip()}")
            i += 1  # { の行をスキップ
            node.offset = list(map(float, lines[i].strip().split()[1:]))
            i += 1
            if lines[i].strip() != "}":
                raise ValueError(f"Expected '}}' at line {i}, but found: {lines[i].strip()}")
            i += 1  # } の行をスキップ

        return node, i

    # モーションデータの解析の概要
    def parse_motion_section(self, lines):
        motion_section_found = False
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if "MOTION" in line:
                motion_section_found = True
                self.parse_motion(lines[i+1:])
                print("Motion data parsed successfully")
                break
            i += 1
        if not motion_section_found:
            raise ValueError("MOTION section not found in BVH file")
        
    def parse_frame(self, line):
        return list(map(float, line.strip().split()))
    
    # モーションデータの解析の実行
    def parse_motion(self, lines):
        self.frames = []
        frame_lines = []

        for line in lines:
            if "Frames:" in line:
                continue
            if "Frame Time:" in line:
                self.frame_time = float(line.split()[2])
                continue
            frame_lines.append(line.strip())

        self.frames = [self.parse_frame(line) for line in frame_lines]

# スケルトンの描画
def draw_skeleton(ax, node, parent_position=None):
    if parent_position is not None:
        # 親ノードと子ノードを青い線で結ぶ
        ax.plot([parent_position[0], node.position[0]], 
                [parent_position[1], node.position[1]], 
                [parent_position[2], node.position[2]], 'b-')

    for child in node.children:
        draw_skeleton(ax, child, node.position)

# ノードの位置と回転を更新する関数
def update_node_position(node, frame_data, index, parent_position=[0,0,0], parent_rotation=[0,0,0], is_root=False):

    if node.channels:
        rot_order = list() 
        axis_order = ''
        for axis in node.channels:
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
            x_pos, y_pos, z_pos = frame_data[:3]

            # 初期位置の設定
            node.position = np.array([x_pos, y_pos, z_pos])
            
            # 初期回転の計算
            global_rotation = R.from_euler(axis_order[::-1], rot_order[::-1], degrees=True)
        # Joint
        else:
            # 回転の計算
            local_rotation = R.from_euler(axis_order[::-1], rot_order[::-1], degrees=True)
            global_rotation = parent_rotation * local_rotation

            # 位置の計算
            node.position = parent_position + parent_rotation.apply(np.array(node.offset))

    # Site
    else:
        global_rotation = parent_rotation
        node.position = parent_position + parent_rotation.apply(np.array(node.offset))

    for child in node.children:
        index = update_node_position(child, frame_data, index, node.position, global_rotation)
    return index

# アニメーションのフレームを更新する関数
def update_skeleton(num, frames, ax, root):
    ax.clear()
    ax.set_xlim(-100, 100)
    ax.set_ylim(0, 200)
    ax.set_zlim(-100, 100)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    frame_data = frames[num]
    index = 3
    update_node_position(root, frame_data, index, is_root=True)
    draw_skeleton(ax, root, root.position)

# BVHファイルのロード
path = os.path.abspath(os.path.join('../data/perfume.bvh'))
loader = BVHLoader(path)
loader.load()

root = loader.root
frames = loader.frames
# 3Dプロットと軸の設定
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-100, 100)
ax.set_ylim(0, 200)
ax.set_zlim(-100, 100)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# アニメーションの設定
ani = animation.FuncAnimation(fig, update_skeleton, frames=len(frames), fargs=(frames, ax, root), interval=loader.frame_time * 50)
plt.show()
