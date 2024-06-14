def OBJLoader(file_path):
    vertices = []
    faces = []
    normals = []
    texture_coords = []
    objects = []
    smoothing_groups = []
    lines = []

    with open(file_path, 'r') as file:
        for line in file:
            lines.append(line.strip())
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == 'v':
                # 頂点情報
                vertices.append(list(map(float, parts[1:4])))
            elif parts[0] == 'vn':
                # 法線情報
                normals.append(list(map(float, parts[1:4])))
            elif parts[0] == 'vt':
                # テクスチャ座標
                texture_coords.append(list(map(float, parts[1:3])))
            elif parts[0] == 'f':
                # 面情報
                face = []
                for vert in parts[1:]:
                    vertex_data = vert.split('/')
                    # 頂点インデックス、テクスチャインデックス、法線インデックスを取得
                    vertex_index = int(vertex_data[0]) - 1 if vertex_data[0] else None
                    texture_index = int(vertex_data[1]) - 1 if len(vertex_data) > 1 and vertex_data[1] else None
                    normal_index = int(vertex_data[2]) - 1 if len(vertex_data) > 2 and vertex_data[2] else None
                    face.append((vertex_index, texture_index, normal_index))
                faces.append(face)
            elif parts[0] == 'o':
                # オブジェクト名
                objects.append(parts[1])
            elif parts[0] == 's':
                # スムージンググループ
                smoothing_groups.append(parts[1])

    return vertices, faces, normals, texture_coords, objects, smoothing_groups, lines

# 使用例
file_path = '/Users/yu/Desktop/岩本さん/data_for_skinning/Male.obj'
vertices, faces, normals, texture_coords, objects, smoothing_groups, lines = OBJLoader(file_path)
