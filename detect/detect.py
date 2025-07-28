
def compute_iou(xyxy1,xyxy2):
    tlwh1 = [xyxy1[0], xyxy1[1], xyxy1[2] - xyxy1[0], xyxy1[3] - xyxy1[1]]
    tlwh2 = [xyxy2[0], xyxy2[1], xyxy2[2] - xyxy2[0], xyxy2[3] - xyxy2[1]]

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
        iou = inter_area / union_area
    else:
        iou = 0
    return iou


def detect_by_iou(tracker, max_frame_id, dets_by_frame):

    anormal_frames = {}
    anormal_tracks = {}
    tmp_tracks = {}
    anormal_ids = {}

    for frame_id in range(1, max_frame_id + 1):
        tracker.predict()
        dets = dets_by_frame.get(frame_id, [])
        tracker.update(dets)

        for track_id in tracker.deleted_ids:
            if track_id in tmp_tracks:
                del tmp_tracks[track_id]

        for t in tracker.tracks:
            if t.is_confirmed():
                bbox1 = t.get_bbox()
                if t.id not in tmp_tracks:
                    tmp_tracks[t.id] =  {'start_frame': frame_id, 'bbox':bbox1,  'count': 1}
                else:
                    tmp_tracks[t.id]['count'] += 1
                    if frame_id - tmp_tracks[t.id]['start_frame'] > 30:
                        tmp_tracks[t.id]['start_frame'] = frame_id
                        bbox2 = tmp_tracks[t.id]['bbox']
                        iou = compute_iou(bbox1, bbox2)
                        tmp_tracks[t.id]['bbox'] = bbox1
                        if iou > 0.85:
                            anormal_ids[t.id] = True
                        else:
                            del tmp_tracks[t.id]
                    
                    if t.id in anormal_ids:
                        if frame_id not in anormal_frames:
                            anormal_frames[frame_id] = []
                        x1, y1, x2, y2 = bbox1
                        anormal_frames[frame_id].append((x1, y1, x2 - x1, y2 - y1))
                        if t.id not in anormal_tracks:
                            anormal_tracks[t.id] = []
                        anormal_tracks[t.id].append((frame_id, (x1, y1, x2 - x1, y2 - y1), t.conf))

                
                

    return anormal_tracks,anormal_frames
