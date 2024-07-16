import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.ndimage import minimum_filter
import sys
import os

# 一つ上の階層のパスを取得
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# 親ディレクトリをsys.pathに追加
sys.path.append(parent_dir)

# 他のファイルにある関数のimport
from skeleton_animation.skeleton_animation import update_node_position


# ジョイント位置をリストに追加
def compute_frame_positions(root, frame_data):
    positions = []
    update_node_position(root, frame_data, 0, is_root=True)
    def collect_positions(node):
        positions.append(node.position)
        for child in node.children:
            collect_positions(child)
    collect_positions(root)
    return positions

# 類似度行列の定義
def compute_similarity_matrix(frames1, frames2, root, window_size):
    num_frames1 = len(frames1)
    num_frames2 = len(frames2)
    similarity_matrix = np.zeros((num_frames1 - window_size, num_frames2 - window_size))

    frame_positions1 = [compute_frame_positions(root, frame) for frame in frames1]
    frame_positions2 = [compute_frame_positions(root, frame) for frame in frames2]

    for i in range(num_frames1 - window_size):
        for j in range(num_frames2 - window_size):
            similarity_matrix[i, j] = calculate_window_distance(frame_positions1, frame_positions2, i, j, window_size)

    return similarity_matrix

# 遷移コストの計算
def calculate_window_distance(frame_positions1, frame_positions2, start_i, start_j, window_size):
    distance_sum = 0
    for k in range(window_size):
        positions1 = frame_positions1[start_i + k]
        positions2 = frame_positions2[start_j + k]
        distance_sum += frame_distance(positions1, positions2)
    return distance_sum

# 距離の計算
def frame_distance(positions1, positions2):
    points1 = np.array(positions1)
    points2 = np.array(positions2)

    R_optimal, t_optimal = compute_optimal_transform(points1, points2)

    transformed_positions1 = np.dot(points1, R_optimal.T) + t_optimal
    pos_diff_squared = np.sum(np.linalg.norm(transformed_positions1 - points2, axis=1)**2)

    return pos_diff_squared

# 最適な剛体変換のための平行移動と回転を計算
def compute_optimal_transform(points1, points2):
    # 重心の計算
    centroid1 = np.mean(points1, axis=0)
    centroid2 = np.mean(points2, axis=0)

    # ポイントの中心化
    centered1 = points1 - centroid1
    centered2 = points2 - centroid2

    # クロス共分散行列の計算
    H = np.dot(centered1.T, centered2)

    # 特異値分解の実行
    U, S, Vt = np.linalg.svd(H)
    R_optimal = np.dot(Vt.T, U.T)

    # 回転行列の調整
    if np.linalg.det(R_optimal) < 0:
        Vt[-1, :] *= -1
        R_optimal = np.dot(Vt.T, U.T)

    # 並行移動ベクトルの計算
    t_optimal = centroid2.T - np.dot(R_optimal, centroid1.T)

    return R_optimal, t_optimal

# 類似度行列のヒートマップを用いた可視化
def display_similarity_matrix(similarity_matrix, frames1_start, frames2_start):
    similarity_matrix_T = similarity_matrix.T
    plt.imshow(similarity_matrix_T, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Similarity Matrix')
    plt.xlabel('Frame Index')
    plt.ylabel('Frame Index')

    # x軸とy軸の目盛りの設定
    x_tick_interval = 10
    y_tick_interval = 10

    x_ticks = np.arange(frames1_start, frames1_start + similarity_matrix_T.shape[1], x_tick_interval)  # x軸の目盛り位置
    y_ticks = np.arange(frames2_start, frames2_start + similarity_matrix_T.shape[0], y_tick_interval)  # y軸の目盛り位置
    
    plt.xticks(ticks=np.arange(0, similarity_matrix_T.shape[1], x_tick_interval), labels=x_ticks)
    plt.yticks(ticks=np.arange(0, similarity_matrix_T.shape[0], y_tick_interval), labels=y_ticks)

    plt.gca().invert_yaxis() 
    plt.show()

# 極小値の検出
def find_local_minima(similarity_matrix):
    local_minima = (similarity_matrix == minimum_filter(similarity_matrix, footprint=np.ones((3, 3))))

    # 局所最小値の位置を示すブール型行列を返す
    return local_minima

# モーショングラフの構築
def build_motion_graph(similarity_matrix, threshold, frames1_start, frames1_end, frames2_start, frames2_end):
    G = nx.DiGraph()

    # 始まりと終わりのノードを追加
    G.add_node(frames1_start, bipartite=0)
    G.add_node(frames1_end, bipartite=0)
    G.add_node(frames2_start, bipartite=1)
    G.add_node(frames2_end, bipartite=1)

    num_windows1 = similarity_matrix.shape[0]
    num_windows2 = similarity_matrix.shape[1]
    local_minima = find_local_minima(similarity_matrix)

    for i in range(num_windows1):
        for j in range(num_windows2):
            if local_minima[i, j] and similarity_matrix[i, j] < threshold:
                G.add_node(i + frames1_start, bipartite=0)
                G.add_node(j + frames2_start, bipartite=1)
                G.add_edge(i + frames1_start, j + frames2_start, thickness=1)
                a = i + frames1_start
                b = j + frames2_start
                
    # 中間ノードを取得
    intermediate_nodes = sorted(G.nodes - {frames1_start, frames1_end, frames2_start, frames2_end})

    # ノードを frames2_start より小さいグループと大きいグループに分ける
    group1 = [node for node in intermediate_nodes if node < frames2_start]
    group2 = [node for node in intermediate_nodes if node >= frames2_start]

    # グループ1間のエッジを追加
    if group1:
        G.add_edge(frames1_start, group1[0], thickness=3)
        for i in range(len(group1) - 1):
            G.add_edge(group1[i], group1[i + 1], thickness=3)
        G.add_edge(group1[-1], frames1_end, thickness=3)

    else:
        # 中間ノードがない場合、直接始まりから終わりへのエッジを追加
        G.add_edge(frames1_start, frames1_end, thickness=3)

    # グループ2間のエッジを追加
    if group2:
        G.add_edge(frames2_start, group2[0], thickness=3)
        for i in range(len(group2) - 1):
            G.add_edge(group2[i], group2[i + 1], thickness=3)
        G.add_edge(group2[-1], frames2_end, thickness=3)

    else:
        # 中間ノードがない場合、直接始まりから終わりへのエッジを追加
        G.add_edge(frames2_start, frames2_end, thickness=3)
    
    return G, a, b

# モーショングラフの可視化
def display_motion_graph(graph):
    # ノードを2つのグループに分ける
    l, r = nx.bipartite.sets(graph)
    pos = nx.bipartite_layout(graph, l)

    nx.draw(graph, pos, with_labels=True, node_color='lightblue')

    # エッジの描画（太いエッジ）
    thick_edges = [(u, v) for u, v, d in graph.edges(data=True) if d['thickness'] == 3]
    nx.draw_networkx_edges(graph, pos, edgelist=thick_edges, edge_color='black', width=3)

    # エッジの描画（通常の太さ）
    normal_edges = [(u, v) for u, v, d in graph.edges(data=True) if d['thickness'] == 1]
    nx.draw_networkx_edges(graph, pos, edgelist=normal_edges, edge_color='red', width = 1)
 
    plt.title("Motion Graph Visualization")
    plt.show()
