import numpy as np

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

# 使用例
loader = BVHLoader('../data/1_wayne_1_11_12.bvh')
loader.load()
