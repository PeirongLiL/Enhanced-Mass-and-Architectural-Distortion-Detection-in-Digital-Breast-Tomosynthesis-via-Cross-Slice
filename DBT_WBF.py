import pandas as pd
import numpy as np


def iou(box1, box2):
    """
    计算两个框的交并比 (IoU)
    box: [x, y, width, height]
    """
    x1_min = box1[0]
    y1_min = box1[1]
    x1_max = box1[0] + box1[2]
    y1_max = box1[1] + box1[3]

    x2_min = box2[0]
    y2_min = box2[1]
    x2_max = box2[0] + box2[2]
    y2_max = box2[1] + box2[3]

    inter_min_x = max(x1_min, x2_min)
    inter_min_y = max(y1_min, y2_min)
    inter_max_x = min(x1_max, x2_max)
    inter_max_y = min(y1_max, y2_max)

    inter_area = max(0, inter_max_x - inter_min_x) * max(0, inter_max_y - inter_min_y)

    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)

    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area != 0 else 0


def center_distance(box1, box2):
    """
    计算两个框的中心点之间的距离
    box: [x, y, width, height]
    """
    center_x1 = box1[0] + box1[2] / 2
    center_y1 = box1[1] + box1[3] / 2
    center_x2 = box2[0] + box2[2] / 2
    center_y2 = box2[1] + box2[3] / 2

    distance = np.sqrt((center_x1 - center_x2) ** 2 + (center_y1 - center_y2) ** 2)
    return distance


def dbt_wbf(df, iou_threshold=0.1, distance_threshold=0, score_threshold=0.001):

    # # 限制框的长宽比
    df = df[(df['Width'] / df['Height'] > 1 / 3) & (df['Width']/ df['Height'] < 3) ]
    df = df.groupby(['PatientID', 'StudyUID', 'View', 'Z'], group_keys=True).apply(
        lambda x: x[x['Score'] > score_threshold].nlargest(6, 'Score')
    ).reset_index(drop=True)
    df = df[(df['Width']  > 10) & (df['Height']  > 10)]
    # df = df[(df['Width'] < 1300) & (df['Height'] < 1300)]

    boxes = df[['X', 'Y', 'Width', 'Height']].values
    scores = df['Score'].values
    z = df[['Z', ]].values
    # print(max(z)[0]/4)
    z_t=max(z)[0]/4
    indices = np.argsort(scores)[::-1]

    keep = []
    score_weights = []
    processed = set()

    while len(indices) > 0:  # 对所有框判断
        current = indices[0]  # 当前分数最大的框


        overlapping_indices = [current]
        remaining_indices = indices[1:]

        for i in remaining_indices:  # 检查所有和当前框iou大于阈值且z轴距离小于阈值的框
            if iou(boxes[current], boxes[i]) > iou_threshold  and abs(z[current] - z[i]) <= z_t:
            # if iou(boxes[current], boxes[i]) > iou_threshold  and abs(z[current] - z[i]) <= z_t:

                overlapping_indices.append(i)

        # 判断是否有重叠框
        if len(overlapping_indices) > z_t/2:
            # 使用基于z轴距离的加权平均
            total_weight_s = 0.0
            weighted_score_sum = 0.0
            for i in overlapping_indices:
                z_diff = abs(z[current] - z[i])

                # 计算权重，z_diff 越小权重越大
                weight_z = max(0,1 - (z_diff / z_t)**3+ 0.3)  # 避免除以负数
                weighted_score_sum += scores[i] * weight_z
                total_weight_s += weight_z



            mean_score = weighted_score_sum / total_weight_s

            # weight_n = min(0.7 + len(overlapping_indices) * 0.02, 1.3)
            weight_n=len(overlapping_indices)/z_t
            combined_score = mean_score * weight_n
            keep.append(current)
            score_weights.append(combined_score)

                # 标记重叠框为已处理
            for idx in overlapping_indices:
                processed.add(idx)

        # 更新 indices 列表，移除已处理的框
        indices = [i for i in remaining_indices if i not in processed]

    # 对 score_weights 进行归一化
    if score_weights:  # 避免空列表导致除零错误
        min_score = min(score_weights)
        max_score = max(score_weights)
        score_weights = [(s - min_score) / (max_score - min_score) for s in score_weights]

    # 更新df中的Score列
    df.loc[keep, 'Score'] = score_weights

    return df.iloc[keep]


def compute_tp_fp_fn(gts, preds):
    """
    计算真阳性 (TP), 假阳性 (FP) 和 假阴性 (FN)，并返回 TP 平均的 IoU
    Args:
        gts (DataFrame): Ground truth annotations (标注框).
        preds (DataFrame): Predictions (预测框).
    Returns:
        Tuple[int, int, int, float]: Counts of true positives, false positives, false negatives, and average IoU for TPs.
    """
    tp = 0
    fp = 0
    fn = 0
    iou_sum = 0
    z_sum = 0

    gt_boxes = gts[['X', 'Y', 'Width', 'Height', 'Slice', 'VolumeSlices']].values
    pred_boxes = preds[['X', 'Y', 'Width', 'Height', 'Z']].values

    gt_centers = [((bbox[0] + bbox[2] / 2), (bbox[1] + bbox[3] / 2)) for bbox in gt_boxes]
    pred_centers = [((bbox[0] + bbox[2] / 2), (bbox[1] + bbox[3] / 2)) for bbox in pred_boxes]

    gt_thresholds = [max(100, np.linalg.norm([(bbox[2]), (bbox[3])]) / 2) for bbox in gt_boxes]

    matched = [False] * len(gt_centers)
    for pred_idx, (pc, pz) in enumerate(zip(pred_centers, preds['Z'])):
        matched_any = False
        for i, (gc, threshold) in enumerate(zip(gt_centers, gt_thresholds)):
            gt_z = gt_boxes[i][4]
            gt_depth = gt_boxes[i][5]
            z_range = 0.25 * gt_depth
            if (not matched[i] and
                    np.linalg.norm(np.array(pc) - np.array(gc)) <= threshold and
                    (gt_z - z_range <= pz <= gt_z + z_range)):
                tp += 1
                matched[i] = True
                matched_any = True
                # 计算并累加IoU
                iou_sum += iou(pred_boxes[pred_idx][:4], gt_boxes[i][:4])
                z_sum+=abs(pz - gt_z)
                break

        if not matched_any:
            fp += 1

    fn += len(gt_centers) - sum(matched)

    # 计算平均IoU，避免除以0
    avg_iou = iou_sum

    return tp, fp, fn, avg_iou,z_sum


# 读取CSV文件
file_path = r'D:\pycharm project\mmyolo-main\demo\new3.csv'
df = pd.read_csv(file_path)

# 按PatientID, StudyUID, View分组，并在每个组内执行v
result_df = df.groupby(['PatientID', 'StudyUID', 'View'], group_keys=False).apply(dbt_wbf)

# 保存结果到新的CSV文件
result_df.to_csv('faster_result.csv', index=False)

print("NMS处理完成，结果已保存到nms_result.csv")

# 读取CSV文件
pred_file_path = 'faster_result.csv'
gt_file_path = r'D:\pycharm project\dataset\dbt duke\manifest-1617905855234/boxes-test.csv'

pred_df = pd.read_csv(pred_file_path)
gt_df = pd.read_csv(gt_file_path)

# 按PatientID, StudyUID, View分组，并在每个组内执行NMS和计算TP, FP, FN
results = []

for group_key, group in pred_df.groupby(['PatientID', 'StudyUID', 'View']):
    # 提取相同组的标注框
    gt_group = gt_df[(gt_df['PatientID'] == group_key[0]) &
                     (gt_df['StudyUID'] == group_key[1]) &
                     (gt_df['View'] == group_key[2])]

    # 计算真阳性、假阳性和假阴性
    tp, fp, fn ,avg_iou,z_sum= compute_tp_fp_fn(gt_group, group)
    results.append((group_key[0], group_key[1], group_key[2], tp, fp, fn,avg_iou,z_sum))

# 计算总体Recall
total_tp = sum([x[3] for x in results])
total_fp = sum([x[4] for x in results])
total_fn = sum([x[5] for x in results])
recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
avg_iou =sum([x[6] for x in results])/total_tp
avg_z =sum([x[7] for x in results])/total_tp
print(f"avg_iou: {avg_iou:.4f}")
print(f"avg_z: {avg_z:.4f}")

# 输出Recall
print(f"Overall Recall: {recall:.4f}")
print(f"Overall TP: {total_tp}")
print(f"Overall FP: {total_fp}")
print(f"Overall FN: {total_fn}")

# 保存结果到CSV文件
result_df = pd.DataFrame(results, columns=['PatientID', 'StudyUID', 'View', 'TP', 'FP', 'FN','Avg_iou','Avg_z'])
result_df.to_csv('recall_results.csv', index=False)

print("Recall计算完成，结果已保存到recall_results.csv")
