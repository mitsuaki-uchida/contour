import numpy as np
import matplotlib.pyplot as plt

def azimuth(line):
    return normalize(np.arctan2((line[0] - line[1])[1], (line[0] - line[1])[0]))

def inverse(azimuth):
    return (azimuth + np.pi) % (np.pi * 2.0)

def same_direction(line1, line2):
    return (azimuth(line1) == azimuth(line2))

def is_on_line(line, point):
    d1 = np.linalg.norm(line[0] - point)
    d2 = np.linalg.norm(line[1] - point)
    line_distance = np.linalg.norm(line[1] - line[0])

    return (line_distance == (d1 + d2))

def normalize(azimuth):
    return (azimuth if azimuth >= 0 else azimuth + (np.pi * 2.0))

def intersect(line1, line2):
    a1 = line1[0]
    b1 = line1[1]
    a2 = line2[0]
    b2 = line2[1]

    deno = np.cross(b1 - a1, b2 - a2)

    if deno == 0.0:
        return None

    s = np.cross(a2 - a1, b2 - a2) / deno
    t = np.cross(b1 - a1, a1 - a2) / deno

    if (s < 0.0) or (1.0 < s) or (t < 0.0) or (1.0 < t):
        # 交差無し
        return None

    # 交差あり
    return np.array([a1[0] + s * (b1 - a1)[0], a1[1] + s * (b1 - a1)[1]])

def is_in_bound(polygon, point):
    for index in range(len(polygon) - 1):
        cross = np.cross(polygon[index + 1] - polygon[index], point - polygon[index])

        if cross > 0.0:
            # 外側
            return False

    return True

def calc_contour(rects):
    polygons = rects[:]
    contour = list()
    current = polygons.pop(0)

    while polygons:
        contour.clear()
        mergee = polygons.pop(0)

        for mergee_index in range(len(mergee) - 1):
            vectors = [np.array(point) for point in current]
            if not is_in_bound(vectors, np.array(mergee[mergee_index])):
                break
        else:
            # mergeeのすべての点がcurrentの内部
            contour = current[:]
            continue

        current_index = 0

        while current_index < (len(current) - 1):
            vectors = [np.array(point) for point in mergee]
            if not is_in_bound(vectors, np.array(current[current_index])):
                break

            current_index += 1
        else:
            # currentのすべての点がmergeeの内部
            current = mergee
            contour = mergee[:]
            continue

        contour.append(current[current_index])

        while (len(contour) < 2) or (contour[0] != contour[-1]):
            # 次に進むラインを決定
            before_current_index = current_index - 1 if current_index != 0 else len(current) - 2
            next_current_index = current_index + 1 if current_index != len(current) - 2 else 0

            a1 = np.array(current[before_current_index])
            b1 = np.array(current[current_index])
            c1 = np.array(current[next_current_index])

            current_line = (a1, b1)
            current_line_azimuth = azimuth(current_line)

            next_line = (b1, c1)
            next_line_azimuth = azimuth(next_line)

            max_diff_angle = normalize(next_line_azimuth - inverse(current_line_azimuth))

            current_mergee_index = None
            current_point = b1

            for mergee_index in range(len(mergee) - 1):
                a2 = np.array(mergee[mergee_index])
                b2 = np.array(mergee[mergee_index + 1])

                line = (a2, b2)

                # 交点
                intersect_point = intersect(current_line, line)
                
                if intersect_point is None:
                    # 交差しない
                    if same_direction(current_line, line) and is_on_line(line, current_line[1]):
                        intersect_point = current_line[1]
                    else:
                        continue

                if not np.array_equal(current_line[1], intersect_point):
                    # 交点がb1ではない
                    continue

                if np.array_equal(line[1], current_line[1]):
                    # b1とb2が同位置
                    # TODO: b2がb1に届いていないケース？ -> 方位角で判断
                    continue

                diff_angle = normalize(azimuth((a2, b2)) - inverse(current_line_azimuth))

                if diff_angle == max_diff_angle:
                    # 方位角が同一である場合は短い方を優先
                    distance1 = np.linalg.norm((b1 - np.array(mergee[current_mergee_index]))
                                               if current_mergee_index is not None
                                               else (next_line[0] - next_line[1]))
                    distance2 = np.linalg.norm(b1 - np.array(mergee[mergee_index]))
                    if distance2 < distance1:
                        current_mergee_index = mergee_index
                        current_point = intersect_point
                elif diff_angle > max_diff_angle:
                    # 方位角が大きい方を優先
                    current_mergee_index = mergee_index
                    current_point = intersect_point
                    # 最大角度を更新
                    max_diff_angle = diff_angle

            if current_mergee_index is not None:
                # mergee側に移動
                mergee, current = current, mergee
                current_index = current_mergee_index

            min_distance = np.finfo(np.float32).max
            next_point = None
            mergee_index_of_next_point = 0

            for mergee_index in range(len(mergee) - 1):
                a1 = np.array(current_point)
                b1 = np.array(current[current_index + 1])
                a2 = np.array(mergee[mergee_index])
                b2 = np.array(mergee[mergee_index + 1])

                current_line = (a1, b1)
                mergee_line = (a2, b2)
                intersect_point = intersect(current_line, mergee_line)

                if intersect_point is None:
                    # 交差無し
                    continue

                if np.array_equal(intersect_point, current_line[0]):
                    # 始点で交差 -> 解決済み
                    continue

                if np.array_equal(intersect_point, current_line[1]):
                    # 終点で交差 -> 終点が始点になった際に解決
                    continue

                current_line_azimuth = azimuth((current_line[0], intersect_point))
                intersect_line_azimuth = azimuth((intersect_point, mergee_line[1]))

                diff_angle = normalize(current_line_azimuth - intersect_line_azimuth)
                diff_angle = diff_angle if diff_angle < np.pi else diff_angle - (np.pi * 2.0)

                if diff_angle >= 0:
                    # 同方位、または内側は無視
                    continue

                distance = np.linalg.norm(intersect_point - current_line[0])

                if distance < min_distance:
                    min_distance = distance
                    intersect_pos = intersect_point
                    next_point = intersect_point
                    mergee_index_of_next_point = mergee_index
            
            if next_point is not None:
                contour.append(next_point.tolist())
                if (len(contour) >= 2) and (contour[0] == contour[-1]):
                    # この時点で一周
                    break

                mergee, current = current, mergee
                current_index = mergee_index_of_next_point

            current_index = current_index + 1 if current_index != len(current) - 2 else 0
            contour.append(current[current_index])

            #print(contour)

        if current == contour:
            # 接触無し
            # mergeeがcurrentの内側、または接していない
            polygons.append(mergee)

        current = contour[:]
        #print(contour)

    #print(contour)

    adjusted = []
    current_point = contour[0]
    adjusted.append(current_point)

    for index in range(1, len(contour) - 1):
        next_index = index + 1 if index != len(contour) - 1 else 0
        next_point = contour[index]
        next_next_point = contour[next_index]

        if not same_direction((np.array(current_point), np.array(next_point)),
                              (np.array(current_point), np.array(next_next_point))):
            adjusted.append(next_point)
            current_point = next_point

    current_point = contour[-1]
    adjusted.append(current_point)

    contour = adjusted

    #print(contour)
    return contour

if __name__ == '__main__':
    rect1 = [[1.0, 1.0], [1.0, 3.0], [3.0, 3.0], [3.0, 1.0], [1.0, 1.0]]
    rect2 = [[2.0, 2.0], [2.0, 4.0], [4.0, 4.0], [4.0, 2.0], [2.0, 2.0]]
    rect3 = [[3.0, 3.0], [3.0, 5.0], [5.0, 5.0], [5.0, 3.0], [3.0, 3.0]]
    rect4 = [[1.0, 1.0], [1.0, 2.0], [2.0, 2.0], [2.0, 1.0], [1.0, 1.0]]
    rect5 = [[1.0, 1.0], [1.0, 10.0], [10.0, 10.0], [10.0, 1.0], [1.0, 1.0]]
    rect6 = [[3.0, 1.0], [3.0, 3.0], [6.0, 3.0], [6.0, 1.0], [3.0, 1.0]]
    rect7 = [[2.0, 1.0], [2.0, 4.0], [3.0, 4.0], [3.0, 1.0], [2.0, 1.0]]
    rect8 = [[1.0, 2.0], [1.0, 3.0], [4.0, 3.0], [4.0, 2.0], [1.0, 2.0]]

    rect_a = [[1.0, 1.0], [1.0, 4.0], [4.0, 4.0], [4.0, 1.0], [1.0, 1.0]]
    rect_b = [[5.0, 6.0], [5.0, 3.0], [2.0, 3.0], [2.0, 6.0], [5.0, 6.0]]
    rect_c = [[3.0, 5.0], [3.0, 8.0], [6.0, 8.0], [6.0, 5.0], [3.0, 5.0]]

    ax = plt.subplot()

    rects = [rect6, rect2, rect4, rect7]
    #rects = [rect_a, rect_b, rect_c]

    for rect in rects:
        xs = [point[0] for point in rect]
        ys = [point[1] for point in rect]
        ax.plot(xs, ys, ':o', color='blue')
        ax.scatter(xs, ys, s=100, color='blue')

    if True:
        contour = calc_contour(rects)

        xs = [point[0] for point in contour]
        ys = [point[1] for point in contour]
        ax.plot(xs, ys, '-o', color='red')

    plt.show(block=True)
