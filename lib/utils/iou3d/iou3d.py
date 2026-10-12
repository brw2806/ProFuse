import torch

EPS = 1e-8
class Point():
    def __init__(self, x=0, y=0): 
        self.x = x
        self.y = y
    def set(self, x, y):
        self.x = x
        self.y = y
    def __add__(self, b):
        self.x += b.x
        self.y += b.y
        return self
    def __sub__(self, b):
        self.x -= b.x
        self.y -= b.y
        return self    

def rotate_around_center(center, angle_cos, angle_sin, p):
    new_x = (p.x - center.x) * angle_cos + (p.y - center.y) * angle_sin + center.x;
    new_y = -(p.x - center.x) * angle_sin + (p.y - center.y) * angle_cos + center.y;
    p.set(new_x, new_y)
    return p

def check_in_box2d(box, p):
    # params: box (5) [x1, y1, x2, y2, angle]
    MARGIN = 1e-5;

    center_x = (box[0] + box[2]) / 2
    center_y = (box[1] + box[3]) / 2
    angle_cos = torch.cos(-box[4])
    angle_sin = torch.sin(-box[4])  # rotate the point in the opposite direction of box
    rot_x = (p.x - center_x) * angle_cos + (p.y - center_y) * angle_sin + center_x
    rot_y = -(p.x - center_x) * angle_sin + (p.y - center_y) * angle_cos + center_y
    return (rot_x > box[0] - MARGIN and rot_x < box[2] + MARGIN and rot_y > box[1] - MARGIN and rot_y < box[3] + MARGIN)

def point_cmp(a, b, center):
    return torch.atan2(a.y - center.y, a.x - center.x) > torch.atan2(b.y - center.y, b.x - center.x);

def cross2(a, b):
    return a.x * b.y - a.y * b.x;

def cross(p1, p2, p0):
    return (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y)

def check_rect_cross(p1, p2, q1, q2):
    ret = min(p1.x,p2.x) <= max(q1.x,q2.x)  and \
              min(q1.x,q2.x) <= max(p1.x,p2.x) and \
              min(p1.y,p2.y) <= max(q1.y,q2.y) and \
              min(q1.y,q2.y) <= max(p1.y,p2.y)
    return ret

def intersection(p1, p0, q1, q0):
    ans = Point(0, 0)
    # import pdb; pdb.set_trace()
    #  fast exclusion
    if (check_rect_cross(p0, p1, q0, q1) == 0):
        return ans, 0

    #  check cross standing
    s1 = cross(q0, p1, p0)
    s2 = cross(p1, q1, p0)
    s3 = cross(p0, q1, q0)
    s4 = cross(q1, p1, q0)

    if (not(s1 * s2 > 0 and s3 * s4 > 0)):
        return ans, 0

    #  calculate intersection of two lines
    s5 = cross(q1, p1, p0)
    if(abs(s5 - s1) > EPS):
        ans.x = (s5 * q0.x - s1 * q1.x) / (s5 - s1);
        ans.y = (s5 * q0.y - s1 * q1.y) / (s5 - s1);
    else:
        a0 = p0.y - p1.y
        b0 = p1.x - p0.x
        c0 = p0.x * p1.y - p1.x * p0.y
        a1 = q0.y - q1.y
        b1 = q1.x - q0.x
        c1 = q0.x * q1.y - q1.x * q0.y
        D = a0 * b1 - a1 * b0

        ans.x = (b0 * c1 - b1 * c0) / D;
        ans.y = (a1 * c0 - a0 * c1) / D;
    
    return ans, 1

def box_overlap(box_a, box_b):
    a_x1, a_y1, a_x2, a_y2, a_angle = box_a
    b_x1, b_y1, b_x2, b_y2, b_angle = box_b
    center_a = Point((a_x1 + a_x2) / 2, (a_y1 + a_y2) / 2)
    center_b = Point((b_x1 + b_x2) / 2, (b_y1 + b_y2) / 2)

    box_a_corners = [0] * 5
    box_a_corners[0] = Point(a_x1, a_y1)
    box_a_corners[1] = Point(a_x2, a_y1)
    box_a_corners[2] = Point(a_x2, a_y2)
    box_a_corners[3] = Point(a_x1, a_y2)

    box_b_corners = [0] * 5
    box_b_corners[0] = Point(b_x1, b_y1)
    box_b_corners[1] = Point(b_x2, b_y1)
    box_b_corners[2] = Point(b_x2, b_y2)
    box_b_corners[3] = Point(b_x1, b_y2)

    a_angle_cos = torch.cos(a_angle)
    a_angle_sin = torch.sin(a_angle)
    b_angle_cos = torch.cos(b_angle)
    b_angle_sin = torch.sin(b_angle)

    for k in range(4):
        box_a_corners[k] = rotate_around_center(center_a, a_angle_cos, a_angle_sin, box_a_corners[k])
        box_b_corners[k] = rotate_around_center(center_b, b_angle_cos, b_angle_sin, box_b_corners[k])

    box_a_corners[4] = box_a_corners[0]
    box_b_corners[4] = box_b_corners[0]

    cross_points = [0]*16
    poly_center = Point(0, 0)
    cnt, flag = 0, 0
    
    for i in range(4):
        for j in range(4):
            cross_points[cnt], flag = intersection(box_a_corners[i + 1], box_a_corners[i], box_b_corners[j + 1], box_b_corners[j])
            if (flag):
                # import pdb; pdb.set_trace()
                poly_center = poly_center + cross_points[cnt]
                cnt += 1

    # check corners
    for k in range(4):
        if (check_in_box2d(box_a, box_b_corners[k])):
            poly_center = poly_center + box_b_corners[k];
            cross_points[cnt] = box_b_corners[k];
            cnt += 1
        if (check_in_box2d(box_b, box_a_corners[k])):
            poly_center = poly_center + box_a_corners[k];
            cross_points[cnt] = box_a_corners[k];
            cnt += 1

    if cnt > 0:
        poly_center.x /= cnt;
        poly_center.y /= cnt;

    #  sort the points of polygon
    temp = Point()
    for j in range(cnt - 1):
        for i in range(cnt - j - 1):
            if (point_cmp(cross_points[i], cross_points[i + 1], poly_center)):
                temp = cross_points[i]
                cross_points[i] = cross_points[i + 1]
                cross_points[i + 1] = temp

    # get the overlap areas
    area = 0
    for k in range(cnt - 1):
        area += cross2(cross_points[k] - cross_points[0], cross_points[k + 1] - cross_points[0])

    return abs(area) / 2.0;

def iou_bev(box_a, box_b):
    sa = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    sb = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    s_overlap = box_overlap(box_a, box_b)
    return s_overlap / max(sa + sb - s_overlap, EPS)

def boxes_iou_bev(boxes_a, boxes_b):
    M, _ = boxes_a.shape
    N, _ = boxes_b.shape
    ans_iou = torch.zeros(M, N).cuda()
    for i in range(M):
        for j in range(N):
            ans_iou[i, j] = iou_bev(boxes_a[i], boxes_b[j])
    return ans_iou

def boxes_overlap_bev(boxes_a, boxes_b):
    M, _ = boxes_a.shape
    N, _ = boxes_b.shape
    ans_overlap = torch.zeros(M, N).cuda()
    for i in range(M):
        for j in range(N):
            ans_overlap[i, j] = box_overlap(boxes_a[i], boxes_b[j])
    return ans_overlap

def nms(bboxes, scores, threshold=0.5):
        _, order = scores.squeeze().sort(0, descending=True)
        keep = []
        while order.numel() > 0:       # torch.numel()返回张量元素个数
            if order.numel() == 1:     # 保留框只剩一个
                i = order.item()
                keep.append(i)
                break
            else:
                # import pdb; pdb.set_trace()
                i = order[0].item()    # 保留scores最大的那个框box[i]
                keep.append(i)

            # 计算box[i]与其余各框的IOU
            iou = boxes_iou_bev(bboxes[i].unsqueeze(0), bboxes[order[1:]]).squeeze()
            idx = (iou <= threshold).nonzero().squeeze() # 注意此时idx为[N-1,] 而order为[N,]
            if idx.numel() == 0:
                break
            order = order[idx+1]  # 修补索引之间的差值
        return torch.LongTensor(keep)