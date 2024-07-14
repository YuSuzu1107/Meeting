def OBJExporter(file_path, vertices, faces, normals, texture_coords, objects, smoothing_groups, lines):
    with open(file_path, 'w') as file:
        v_index = 0
        vn_index = 0
        vt_index = 0
        f_index = 0
        o_index = 0
        s_index = 0

        for line in lines:
            parts = line.split()
            if parts[0] == 'v' and v_index < len(vertices):
                v = vertices[v_index]
                file.write(f'v {" ".join(f"{coord:.6f}" for coord in v)}\n')
                v_index += 1
            elif parts[0] == 'vn' and vn_index < len(normals):
                vn = normals[vn_index]
                file.write(f'vn {" ".join(f"{coord:.4f}" for coord in vn)}\n')
                vn_index += 1
            elif parts[0] == 'vt' and vt_index < len(texture_coords):
                vt = texture_coords[vt_index]
                file.write(f'vt {" ".join(f"{coord:.6f}" for coord in vt)}\n')
                vt_index += 1
            elif parts[0] == 'f' and f_index < len(faces):
                face_elements = []
                for vert in faces[f_index]:
                    vertex_index = vert[0] + 1 if vert[0] is not None else ''
                    texture_index = vert[1] + 1 if vert[1] is not None else ''
                    normal_index = vert[2] + 1 if vert[2] is not None else ''
                    face_elements.append(f'{vertex_index}/{texture_index}/{normal_index}')
                file.write(f'f {" ".join(face_elements)}\n')
                f_index += 1
            elif parts[0] == 'o' and o_index < len(objects):
                file.write(f'o {objects[o_index]}\n')
                o_index += 1
            elif parts[0] == 's' and s_index < len(smoothing_groups):
                file.write(f's {smoothing_groups[s_index]}\n')
                s_index += 1
            else:
                file.write(line + '\n')
