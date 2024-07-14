class BVHExporter:

    # 初期化
    def __init__(self, root, frames, frame_time):
        self.root = root
        self.frames = frames
        self.frame_time = frame_time

    # エキスポート関数の定義
    def export(self, file_path):
        if self.root is None:
            raise ValueError("Root node is None. Cannot export BVH file.")
        
        with open(file_path, 'w') as file:
            # HIERARCHYセクションの書き出し
            file.write("HIERARCHY\n")
            self.write_hierarchy(file, self.root, 0)
            
            # MOTIONセクションの書き出し
            file.write("MOTION\n")
            file.write(f"Frames: {len(self.frames)}\n")
            file.write(f"Frame Time: {self.frame_time}\n")
            
            for frame in self.frames:
                frame_str = " ".join(map(str, frame))
                file.write(frame_str + "\n")

    # HIERARCHYの具体的な記述法
    def write_hierarchy(self, file, node, level):
        if node is None:
            return
        
        indent = '  ' * level
        if node.name == "End Site":
            file.write(f"{indent}End Site\n")
            file.write(f"{indent}{{\n")
            file.write(f"{indent}  OFFSET {' '.join(map(str, node.offset))}\n")
            file.write(f"{indent}}}\n")
        else:
            node_type = "ROOT" if level == 0 else "JOINT"
            file.write(f"{indent}{node_type} {node.name}\n")
            file.write(f"{indent}{{\n")
            file.write(f"{indent}  OFFSET {' '.join(map(str, node.offset))}\n")
            file.write(f"{indent}  CHANNELS {len(node.channels)} {' '.join(node.channels)}\n")
            
            for child in node.children:
                self.write_hierarchy(file, child, level + 1)
            
            file.write(f"{indent}}}\n")
