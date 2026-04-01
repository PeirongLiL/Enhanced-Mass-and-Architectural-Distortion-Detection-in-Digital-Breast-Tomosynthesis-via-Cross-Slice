# https://github.com/MaciejMazurowski/duke-dbt-data/blob/master/duke_dbt_data.py
from typing import AnyStr, BinaryIO, Dict, List, NamedTuple, Optional, Union
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import pydicom as dicom
from skimage.exposure import rescale_intensity
from sklearn.metrics import roc_auc_score
def evaluate(
    labels_fp: pd._typing.BaseBuffer,
    boxes_fp: pd._typing.BaseBuffer,
    predictions_fp: pd._typing.BaseBuffer,
) -> Dict[str, float]:
    """Evaluate predictions"""
    df_labels = pd.read_csv(labels_fp)
    df_boxes = pd.read_csv(boxes_fp, dtype={"VolumeSlices": float})
    df_pred = pd.read_csv(predictions_fp, dtype={"Score": float})
    df_boxes = df_boxes[df_boxes['AD'] == 1]
    df_labels = df_labels.reset_index().set_index(["StudyUID", "View"]).sort_index()#使用studyuid view重新索引
    # df_labels = df_labels[df_labels['Cancer'] == 1]
    df_boxes = df_boxes.reset_index().set_index(["StudyUID", "View"]).sort_index()
    df_pred = df_pred.reset_index().set_index(["StudyUID", "View"]).sort_index()
    df_pred["TP"] = 0
    df_pred["GTID"] = -1 #GT的id

    thresholds = [df_pred["Score"].max() + 1.0]

    # find true positive predictions and assign detected ground truth box ID
    #寻找TP的检测和它对应的GTID
    for box_pred in df_pred.itertuples():
        if box_pred.Index not in df_boxes.index:
            continue#查看是不是有病灶的检测

        df_boxes_view = df_boxes.loc[[box_pred.Index]]#对应标注
        view_slice_offset = df_boxes.loc[[box_pred.Index], "VolumeSlices"].iloc[0] / 4#深度范围
        tp_boxes = [
            b
            for b in df_boxes_view.itertuples()
            if _is_tp(box_pred, b, slice_offset=view_slice_offset)#确定哪些检测框与当前预测框匹配（真阳性）
        ]
        if len(tp_boxes) > 1:#如果一个预测与多个真实框匹配，使用 _distance 函数计算距离，并选择最近的一个
            # find the nearest GT box
            tp_distances = [_distance(box_pred, b) for b in tp_boxes]
            tp_boxes = [tp_boxes[np.argmin(tp_distances)]]
        if len(tp_boxes) > 0:#TP
            tp_i = tp_boxes[0].index#真值index,每个标注框唯一
            df_pred.loc[df_pred["index"] == box_pred.index, ("TP", "GTID")] = (1, tp_i)
            thresholds.append(box_pred.Score)


    thresholds.append(df_pred["Score"].min() - 1.0)
    result = {}
    df_pred = df_pred[df_pred.index.isin(df_boxes.index)]
    df_pred.to_csv('froc.csv', index=True)
    df_labels = df_labels[df_labels.index.isin(df_boxes.index)]
    evaluation_fps_positive = (1.0, 2.0, 3.0, 4.0)
    tpr_positive,ci = _froc(
        df_pred=df_pred,#预测结果
        thresholds=thresholds,#刚刚得到的阈值，每一个值关系一个预测是不是TP
        n_volumes=len(df_labels),#多少个体积
        df_annotations=df_boxes,#多少个标注框
        evaluation_fps=evaluation_fps_positive,
    )
    tpr_values, fpr_values = _roc(
        df_pred=df_pred,  # 预测结果数据框
        thresholds=thresholds,  # 阈值列表
        df_annotations=df_boxes,  # 标注框
        evaluation_fps=evaluation_fps_positive,  # 用于评价的FP值（如果需要的话）
        n_volumes=len(df_labels),  # 总体积数
    )

    result.update(
        dict(
            (f"sensitivity_at_{int(x)}_fps_positive", y)
            for x, y in zip(evaluation_fps_positive, tpr_positive)
        )
    )
    result.update(
        dict(
            (f"ci_at_{int(x)}_fps_positive", y)
            for x, y in zip(evaluation_fps_positive, ci)
        )
    )
    result.update({"mean_sensitivity_positive": np.mean(tpr_positive)})
    return result

from typing import List, Tuple
def _froc(
    df_pred: pd.DataFrame,
    thresholds: List[float],
    n_volumes: int,
    df_annotations: pd.DataFrame,
    evaluation_fps: Tuple[float],
    n_bootstrap: int = 1,  # 新增参数，控制 bootstrap 的次数
    ci: Tuple[float] = (2.5, 97.5)  # 新增参数，控制置信区间的界限
) -> Tuple[List[float], List[Tuple[float, float]]]:
    tpr = []
    fps = []
    # 存储每个 FPs 下的所有 bootstrap 计算结果
    bootstrap_tprs = {fp: [] for fp in evaluation_fps}
    # 主循环：按阈值分组处理
    for th in sorted(thresholds, reverse=True):
        df_th = df_pred.loc[df_pred["Score"] >= th]
        df_th_unique_tp = df_th.reset_index().drop_duplicates(
            subset=["StudyUID", "View", "TP", "GTID"]
        )  # 确保每个阳性只被计算一次
        n_tps_th = float(sum(df_th_unique_tp["TP"]))  # TP 数量
        tpr_th = n_tps_th / df_annotations.shape[0]  # 使用 df_annotations 总行数代替 n_boxes
        n_fps_th = float(len(df_th[df_th["TP"] == 0]))  # FP 数量
        fps_th = n_fps_th / n_volumes  # FP 数量除以体积数  FPs
        tpr.append(tpr_th)
        fps.append(fps_th)

    # 计算在给定 FPs 下的敏感度
    evaluated_tprs = [np.interp(x, fps, tpr) for x in evaluation_fps]
    # 绘制FROC曲线，限制X轴范围在0到4
    plt.figure(figsize=(10, 6))
    plt.figure(dpi=300)
    plt.plot(fps, tpr, linestyle='-', color='b')
    plt.xticks([0, 1, 2, 3, 4])  # 设置X轴刻度
    plt.xlim(0, 4.1)  # 限制X轴范围
    plt.xlabel('False Positives per Image (FPPI)')
    plt.ylabel('Recall (Sensitivity)')
    plt.title('FROC Curve (0-4 FPPI)')
    plt.grid(True)
    plt.show()
    # 计算在给定 FPs 下的敏感度

    # 执行 bootstrap
    unique_images = df_pred.groupby(['StudyUID', 'View']).size().reset_index()[['StudyUID', 'View']]    # 执行 bootstrap
    for i in range(n_bootstrap):
        # 随机选择图片，允许重复，并加入SAMPLEID
        sampled_images = unique_images.sample(n=len(unique_images), replace=True)
        sampled_images['SAMPLEID'] = range(len(sampled_images))  # 每次采样分配唯一的SAMPLEID
        sampled_df = pd.merge(sampled_images, df_pred, on=['StudyUID', 'View'], how='left')
        # 计算此次采样中所有选中图片的标注框总数
        sampled_annotations = pd.merge(sampled_images, df_annotations, on=['StudyUID', 'View'], how='left')
        n_boxes_sample = sampled_annotations.shape[0]

        sample_tpr = []
        sample_fps = []

        for th in sorted(thresholds, reverse=True):
            df_th = sampled_df[sampled_df["Score"] >= th]
            # 注意这里我们使用了SAMPLEID来确保即使同一张图被多次采样，每次也独立计算TP
            df_th_unique_tp = df_th.drop_duplicates(subset=["SAMPLEID", "TP", "GTID"])
            n_tps_th = float(sum(df_th_unique_tp["TP"]))
            tpr_th = n_tps_th / n_boxes_sample
            n_fps_th = float(len(df_th[df_th["TP"] == 0]))
            fps_th = n_fps_th / n_volumes
            sample_tpr.append(tpr_th)
            sample_fps.append(fps_th)

        for fp, interp_tpr in zip(evaluation_fps, [np.interp(x, sample_fps, sample_tpr) for x in evaluation_fps]):
            bootstrap_tprs[fp].append(interp_tpr)

    # 计算置信区间
    ci_tprs = []
    L4=0
    U4=0
    for fp in evaluation_fps:
        lower = np.percentile(bootstrap_tprs[fp], ci[0])
        upper = np.percentile(bootstrap_tprs[fp], ci[1])
        L4+=lower/4
        U4+=upper/4
        ci_tprs.append((lower, upper))
    print('CI:',L4,U4)


    return evaluated_tprs, ci_tprs


def _roc(
        df_pred: pd.DataFrame,
        thresholds: List[float],
        df_annotations: pd.DataFrame,
        evaluation_fps: Tuple[float],
        n_volumes: int,
) -> Tuple[List[float], List[float]]:
    tpr = []
    fpr = []

    # 主循环：按阈值分组处理
    for th in sorted(thresholds, reverse=True):
        df_th = df_pred.loc[df_pred["Score"] >= th]

        # 计算 TP 数量
        df_th_unique_tp = df_th.reset_index().drop_duplicates(
            subset=["StudyUID", "View", "TP", "GTID"]
        )  # 确保每个阳性只被计算一次
        n_tps_th = float(sum(df_th_unique_tp["TP"]))  # TP 数量
        tpr_th = n_tps_th / df_annotations.shape[0]  # 使用 df_annotations 总行数代替 n_boxes
        tpr.append(tpr_th)

        # 计算 FP 数量
        n_fps_th = float(len(df_th[df_th["TP"] == 0]))  # FP 数量
        fpr_th = n_fps_th / len(df_pred[df_pred['GTID'] == -1])  # 假正例数除以负样本数
        fpr.append(fpr_th)

    # 计算AUC
    auc_value = np.trapz(tpr, fpr)

    # 绘制ROC曲线
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(fpr, tpr, linestyle='-', color='b', label=f'AUC = {auc_value:.4f}')
    plt.xticks(np.linspace(0, 1, 5))  # [0, 0.1, 0.2, ..., 1.0]
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve')
    plt.grid(True)
    plt.legend(loc='lower right')  # 在右下角显示图例
    plt.show()

    return tpr, fpr
def _is_tp(
    box_pred: NamedTuple, box_true: NamedTuple, slice_offset: int, min_dist: int = 100
) -> bool:
    pred_y = box_pred.Y + box_pred.Height / 2
    pred_x = box_pred.X + box_pred.Width / 2
    pred_z = box_pred.Z + box_pred.Depth / 2
    true_y = box_true.Y + box_true.Height / 2
    true_x = box_true.X + box_true.Width / 2
    true_z = box_true.Slice
    # 2D distance between true and predicted center points
    dist = np.linalg.norm((pred_x - true_x, pred_y - true_y))
    # compute radius based on true box size
    dist_threshold = np.sqrt(box_true.Width ** 2 + box_true.Height ** 2) / 2.0
    dist_threshold = max(dist_threshold, min_dist)
    slice_diff = np.abs(pred_z - true_z)
    # TP if predicted center within radius and slice within slice offset
    return dist <= dist_threshold and slice_diff <= slice_offset


def _distance(box_pred: NamedTuple, box_true: NamedTuple) -> float:
    pred_y = box_pred.Y + box_pred.Height / 2
    pred_x = box_pred.X + box_pred.Width / 2
    pred_z = box_pred.Z+ box_pred.Depth / 2
    true_y = box_true.Y + box_true.Height / 2
    true_x = box_true.X + box_true.Width / 2
    true_z = box_true.Slice

    return np.linalg.norm((pred_x - true_x, pred_y - true_y, pred_z - true_z))

path = r'D:\pycharm project\dataset\dbt duke\manifest-1617905855234'
result=evaluate(path+'\BCS-DBT-labels-test.csv',path+'\BCS-DBT-boxes-test.csv',r'predict_3d/faster_result.csv')
print(result)