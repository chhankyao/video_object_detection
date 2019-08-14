import numpy as np
import xml.etree.ElementTree as ET


def parse_bbox(label_file, cls_map, valid_classes=None):
    boxes = []
    root = ET.parse(label_file).getroot()
    for obj in root.findall('object'):
        cls = cls_map[obj.find('name').text]
        if valid_classes is None or cls in valid_classes:
            x1 = int(obj.find('bndbox/xmin').text)
            y1 = int(obj.find('bndbox/ymin').text)
            x2 = int(obj.find('bndbox/xmax').text)
            y2 = int(obj.find('bndbox/ymax').text)
            boxes.append([x1, y1, x2, y2, cls])
    return boxes


def rescale_bbox(box, old_size, new_size):
    x1 = int(box[0] * new_size[1] / old_size[1])
    y1 = int(box[1] * new_size[0] / old_size[0])
    x2 = int(box[2] * new_size[1] / old_size[1])
    y2 = int(box[3] * new_size[0] / old_size[0])
    return x1, y1, x2, y2


def from_tracking_box(box):
    x1, y1, w, h = box
    return int(x1), int(y1), int(x1+w), int(y1+h)
    
    
def to_tracking_box(box):
    x1, y1, x2, y2 = box
    return int(x1), int(y1), int(x2-x1), int(y2-y1)


def iou(box1, box2):
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    x_int = min(box1[2], box2[2]) - max(box1[0], box2[0])
    y_int = min(box1[3], box2[3]) - max(box1[1], box2[1])
    intersection = x_int * y_int if x_int > 0 and y_int > 0 else 0
    union = a1 + a2 - intersection
    return intersection*1. / (union + 1e-9)


def get_tp_iou(boxes_pred, boxes_gt, iou_thres):
    tp = 0
    tp_iou = 0
    matched_gt = []
    for box1 in boxes_pred:
        for i, box2 in enumerate(boxes_gt):
            if i not in matched_gt:
                if box1[-1] == box2[-1] and iou(box1, box2) >= iou_thres:
                    tp += 1
                    tp_iou += iou(box1, box2)
                    matched_gt.append(i)
                    break
    return tp, tp_iou
    
    
def get_tp(boxes_pred, boxes_gt, iou_thres):
    tp = []
    matched_gt = []
    for box1 in boxes_pred:
        matched = False
        for i, box2 in enumerate(boxes_gt):
            if i not in matched_gt:
                if box1[-1] == box2[-1] and iou(box1, box2) >= iou_thres:
                    matched = True
                    matched_gt.append(i)
                    break
        if matched:
            tp.append(1)
        else:
            tp.append(0)
    return tp
               
        
def compute_ap(recall, precision):
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def compute_mAP(tp, conf, cls_pred, cls_gt):
    
    tp = np.array(tp)
    conf = np.array(conf)
    cls_pred = np.array(cls_pred)
    cls_gt = np.array(cls_gt)
    
    unique_classes = np.unique(cls_gt)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in unique_classes:
        i = cls_pred == c
        n_gt = (cls_gt == c).sum()  # Number of ground truth objects
        n_p = i.sum()               # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))
            
    return ap
