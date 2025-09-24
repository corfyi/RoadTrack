import json, os
import  re
from tracker.baseline.tracker import BaselineTracker
from detect.detect import detect_by_iou

def compute_iou(tlwh1, tlwh2):
    r1 = tlwh1[0] + tlwh1[2]
    b1 = tlwh1[1] + tlwh1[3]
    r2 = tlwh2[0] + tlwh2[2]
    b2 = tlwh2[1] + tlwh2[3]
    left = max(tlwh1[0], tlwh2[0])
    top = max(tlwh1[1], tlwh2[1])
    right = min(r1, r2)
    bottom = min(b1, b2)
    if left < right and top < bottom:
        inter_area = (right - left) * (bottom - top)
        union_area = tlwh1[2] * tlwh1[3] + tlwh2[2] * tlwh2[3] - inter_area
        return inter_area / union_area
    return 0

def extract_sequence_name(filename):
    match = re.search(r'K\d{3}', filename)
    return match.group(0) if match else None

def load_dets(det_file, filter_regions):
    dets_by_frame = {}
    max_frame_id = 0
    with open(det_file, 'r') as f:
        for line in f:
            line = line.strip().split(',')
            frame_id = int(line[0])
            max_frame_id = max(max_frame_id, frame_id)
            x1, y1 = float(line[2]), float(line[3])
            w, h = float(line[4]), float(line[5])
            x2, y2 = x1 + w, y1 + h
            conf, cls_id = float(line[6]), int(line[7])
            bbox = [x1, y1, x2, y2]

            u, v = (x1 + x2) / 2, (y1 + y2) / 2
            if any(x1_f < u < x2_f and y1_f < v < y2_f for x1_f, y1_f, x2_f, y2_f in filter_regions):
                continue

            dets_by_frame.setdefault(frame_id, []).append((bbox, conf, cls_id))
    return dets_by_frame, max_frame_id

def load_gt(gt_file):
    gt_tracks, gt_frames = {}, {}
    with open(gt_file, 'r') as f:
        for line in f:
            frame_id, track_id, x1, y1, w, h, conf = map(float, line.strip().split(',')[:7])
            frame_id, track_id = int(frame_id), int(track_id)
            box = (int(x1), int(y1), int(w), int(h))
            gt_tracks.setdefault(track_id, []).append((frame_id, box, conf))
            gt_frames.setdefault(frame_id, []).append(box)
    return gt_tracks, gt_frames

def eval(sample_name, filter_data, root_dir, eval_positive=False):
    sample_dir = os.path.join(root_dir, sample_name)
    seq_name = extract_sequence_name(sample_name)
    det_file = os.path.join(sample_dir, 'data', 'det', 'yolo11m_nms05.txt')
    gt_file = os.path.join(sample_dir, 'data', 'gt', 'gt.txt')

    gt_tracks, gt_frames = load_gt(gt_file) if eval_positive else ({}, {})
    filter_regions = filter_data.get(seq_name, [])
    dets_by_frame, max_frame_id = load_dets(det_file, filter_regions)

    tracker = BaselineTracker(conf_thresh=0.3,match_thresh=0.9,cdt=30)
    anormal_tracks, anormal_frames = detect_by_iou(tracker, max_frame_id, dets_by_frame)

    TP, TN, FP, latency_sum = 0, 0, 0, 0

    for track_id, frames in gt_tracks.items():
        num_frames = len(frames)
        count, is_matched, latency_idx = 0, False, 0
        for frame_id, bbox1, _ in frames:
            if frame_id in anormal_frames:
                for bbox2 in anormal_frames[frame_id]:
                    if compute_iou(bbox1, bbox2) >= 0.5:
                        if not is_matched:
                            is_matched = True
                            latency_sum += latency_idx
                        count += 1
                        break
            latency_idx += 1
        if count / num_frames >= 0.2:
            TP += 1
        else:
            TN += 1

    for track_id, frames in anormal_tracks.items():
        count = 0
        for frame_id, bbox1, _ in frames:
            if frame_id in gt_frames:
                for bbox2 in gt_frames[frame_id]:
                    if compute_iou(bbox1, bbox2) >= 0.5:
                        count += 1
                        break
        if count / len(frames) < 0.9:
            print(f"\n{sample_name}")
            print(f"False Positive: track_id={track_id}, start_frame={frames[0][0]}, end_frame={frames[-1][0]}, bbox={frames[0][1]}, conf={frames[0][2]:.2f}")
            FP += 1

    return TP, TN, FP, latency_sum

def main():
    with open('filter_files/filter.json', 'r') as f:
        filter_data = json.load(f)

    TP, TN, FP, latency_sum = 0, 0, 0, 0

    for root_dir, eval_positive in [('datasets/P/', True), ('datasets/N/', False)]:
        for sample_name in sorted(os.listdir(root_dir)):
            tp1, tn1, fp1, latency = eval(sample_name, filter_data, root_dir, eval_positive)
            TP += tp1
            TN += tn1
            FP += fp1
            latency_sum += latency if eval_positive else 0

    print(f"TP = {TP}, TN = {TN}, FP = {FP}")
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + TN) if (TP + TN) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1_score:.2f}")
    if TP > 0:
        print(f"latency = {latency_sum / TP:.0f}")

if __name__ == "__main__":
    main()
