import os

import scipy.ndimage

from duke_dbt_data import dcmread_image, read_boxes, draw_box
import pandas as pd

import omidb
import numpy as np
import cv2
import csv
import gc
import json

df = read_boxes(boxes_fp="data/boxes-train.csv", filepaths_fp="data/file-paths-train.csv")#读取了box和地址参数，并一起返回


# print(df)
output_path = r'B:\sample3/'
output_folder = 'train'
# create folders
os.makedirs(output_path + '/bbox', exist_ok=True)
try:
    os.mkdir(output_path)
except OSError as e:
    if (e.errno != 17):
        print("copy_case: creation of %s failed" % output_path)
        print(e)
try:
    os.mkdir(output_path + output_folder)
#    os.mkdir(output_path+output_folder) todo: save lesion patches.

except OSError as e:
    if (e.errno != 17):
        print("copy_case: creation of %s failed" % output_path)
        print(e)


def get_masks_and_sizes_of_connected_components(img_mask):#做掩码，返回mask：每个像素属于哪个连通域，mask_pixels_dict：掩码字典，每个连通域像素的数量
    """
    Finds the connected components from the mask of the image
    """
    mask, num_labels = scipy.ndimage.label(img_mask)

    mask_pixels_dict = {}
    for i in range(num_labels + 1):
        this_mask = (mask == i)
        if img_mask[this_mask][0] != 0:
            # Exclude the 0-valued mask
            mask_pixels_dict[i] = np.sum(this_mask)

    return mask, mask_pixels_dict
def get_mask_of_largest_connected_component(img_mask):  #提取最大连通区域的掩膜。
    """
    Finds the largest connected component from the mask of the image
    """
    mask, mask_pixels_dict = get_masks_and_sizes_of_connected_components(img_mask)
    largest_mask_index = pd.Series(mask_pixels_dict).idxmax()
    largest_mask = mask == largest_mask_index
    return largest_mask
def get_edge_values(img, largest_mask, axis):#获取最大连通区域在X或Y方向上的边界（起始、终止位置）。
    """
    Finds the bounding box for the largest connected component
    """
    assert axis in ["x", "y"]
    has_value = np.any(largest_mask, axis=int(axis == "y"))#
    edge_start = np.arange(img.shape[int(axis == "x")])[has_value][0]
    edge_end = np.arange(img.shape[int(axis == "x")])[has_value][-1] + 1
    return edge_start, edge_end
def get_bottommost_pixels(img, largest_mask, y_edge_bottom):#获取掩膜底部非零像素在X方向上的位置。
    """
    Gets the bottommost nonzero pixels of dilated mask before cropping.
    """
    bottommost_nonzero_y = y_edge_bottom - 1
    bottommost_nonzero_x = np.arange(img.shape[1])[largest_mask[bottommost_nonzero_y, :] > 0]
    return bottommost_nonzero_y, bottommost_nonzero_x
def include_buffer_y_axis(img, y_edge_top, y_edge_bottom, buffer_size):#给上下或左右边界增加缓冲区，避免切除重要内容。
    """
    Includes buffer in all sides of the image in y-direction
    """
    if y_edge_top > 0:
        y_edge_top -= min(y_edge_top, buffer_size)
    if y_edge_bottom < img.shape[0]:
        y_edge_bottom += min(img.shape[0] - y_edge_bottom, buffer_size)
    return y_edge_top, y_edge_bottom



def crop_img_from_largest_connected(img, mode='left', erode_dialate=True, iterations=50,
                                    buffer_size=10):
    """
    Performs erosion on the mask of the image, selects largest connected component,
    dialates the largest connected component, and draws a bounding box for the result
    with buffers

    input:
        - img:   2D numpy array
        - mode:  breast pointing left or right

    output: a tuple of (window_location, rightmost_points,
                        bottommost_points, distance_from_starting_side)
        - window_location: location of cropping window w.r.t. original dicom image so that segmentation
           map can be cropped in the same way for training.
        - rightmost_points: rightmost nonzero pixels after correctly being flipped in the format of
                            ((y_start, y_end), x)
        - bottommost_points: bottommost nonzero pixels after correctly being flipped in the format of
                             (y, (x_start, x_end))
        - distance_from_starting_side: number of zero columns between the start of the image and start of
           the largest connected component w.r.t. original dicom image.
    """
    assert mode in ("left", "right")

    img_mask = img > 0

    # Erosion in order to remove thin lines in the background
    if erode_dialate:
        img_mask = scipy.ndimage.binary_erosion(img_mask, iterations=iterations) #做腐蚀

    # Select mask for largest connected component
    largest_mask = get_mask_of_largest_connected_component(img_mask) #提取最大连通区域的掩膜。

    # Dilation to recover the original mask, excluding the thin lines
    if erode_dialate:
        largest_mask = scipy.ndimage.binary_dilation(largest_mask, iterations=iterations)#做扩张

    # figure out where to crop
    y_edge_top, y_edge_bottom = get_edge_values(img, largest_mask, "y")
    x_edge_left, x_edge_right = get_edge_values(img, largest_mask, "x")#获取裁剪边界



    # include maximum 'buffer_size' more pixels on both sides just to make sure we don't miss anything
    y_edge_top, y_edge_bottom = include_buffer_y_axis(img, y_edge_top, y_edge_bottom, buffer_size)#增加边界缓冲

    # 计算新格式的裁剪框坐标
    crop_x = x_edge_left
    crop_y = y_edge_top
    crop_x2 = x_edge_right  # = x + w
    crop_y2 = y_edge_bottom  # = y + h

    out_bbox = omidb.mark.BoundingBox(crop_x, crop_y, crop_x2, crop_y2) #创建外接矩形的对象，包含矩形的左上角坐标和右下角坐标。
    return out_bbox  # returns bounding box and mask image. 返回外接矩形对象 out_bbox 和掩码图像 img2。


def copy_case_rgb_multi(view, client, episode, image_rgb, side, bbox_list, slice_index):
    filename = client + "_" + episode + "_" + view + "_" + str(slice_index) + "_rgb.png"
    print(filename)

    if image_rgb.ndim > 2:
        comp = image_rgb.shape[0]
    else:
        comp = 1

    for nc in range(comp):
        image = image_rgb if comp == 1 else image_rgb[nc]
        dims = image.shape
        image_2d_scaled = (np.maximum(image, 0) / image.max()) * 255.0
        image_2d_scaled = np.uint8(image_2d_scaled)

        if side.lower() == 'r':
            image_2d_scaled = cv2.flip(image_2d_scaled, 1)
            for bbox_roi in bbox_list:
                aux = bbox_roi.x2
                bbox_roi.x2 = dims[1] - bbox_roi.x1
                bbox_roi.x1 = dims[1] - aux

        if nc == 0:
            bbox = crop_img_from_largest_connected(image_2d_scaled)  # 裁剪区域
        image_crop = image_2d_scaled[bbox.y1:bbox.y2, bbox.x1:bbox.x2]

        if nc == 0 and comp > 1:
            out_rgb = np.zeros((image_crop.shape[0], image_crop.shape[1], comp))
        out_rgb[:, :, nc] = image_crop

    H = out_rgb.shape[0]
    W = out_rgb.shape[1]
    # 调整所有框的坐标（适应裁剪后的图像）
    adjusted_bboxes = []
    for bbox_roi in bbox_list:
        new_box = omidb.mark.BoundingBox(
            x1=bbox_roi.x1 - bbox.x1,
            y1=bbox_roi.y1 - bbox.y1,
            x2=min(W,bbox_roi.x2 - bbox.x1),
            y2=min(H,bbox_roi.y2 - bbox.y1)
        )
        adjusted_bboxes.append(new_box)

    # 保存图像
    aux_folder = output_path + output_folder + "/" + filename
    cv2.imwrite(aux_folder, out_rgb, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    # 绘制所有框
    for adj_box in adjusted_bboxes:
        cv2.rectangle(out_rgb, (adj_box.x1, adj_box.y1), (adj_box.x2, adj_box.y2), (255, 0, 0), 3)


    # # 保存带框图像
    # bbox_filename = client + "_" + episode + "_" + view + "_" + str(slice_index) + "_rgb_bbox.png"
    # bbox_aux_folder = output_path + '/bbox' + "/" + bbox_filename
    # cv2.imwrite(bbox_aux_folder, out_rgb, [cv2.IMWRITE_PNG_COMPRESSION, 20])
    # 保存为每行一个图像，多个框放入一列，形式为：[[x1,y1,x2,y2], [x1,y1,x2,y2], ...]
    with open(output_path + '/train_bboxes.csv', 'a+', newline='') as file:
        writer = csv.writer(file)

        # 转成纯列表结构
        box_list = [[b.x1, b.y1, b.x2, b.y2] for b in adjusted_bboxes]

        writer.writerow([client, episode, filename, side, bbox, slice_index, box_list])
    # 返回所有框的裁剪后坐标 + 图像路径和大小
    box_coords = [[b.x1, b.y1, min(b.x2, W), min(b.y2, H)] for b in adjusted_bboxes]
    del out_rgb
    gc.collect()
    return box_coords, aux_folder, H, W

grouped = df.groupby(["PatientID", "StudyUID", "View", "Slice"])
dataset_dicts = []
image_counter = 0  # 处理顺序编号
for group_key, group_df in grouped:
    client, episode, view, slice_index = group_key
    print("处理图片：", client, view, slice_index)

    # 读取图像
    image_path = os.path.join(r'D:\pycharm project\dataset\dbt duke\manifest-1617905855234',
                              group_df.iloc[0]["descriptive_path"])
    image3D = dcmread_image(fp=image_path, view=view)
    image_rgb = np.array([
        image3D[slice_index - 1],
        image3D[slice_index],
        image3D[slice_index + 1]
    ])
    del image3D
    gc.collect()

    # 所有bbox打包为 omidb.mark.BoundingBox 实例列表
    bbox_list = []
    for _, row in group_df.iterrows():
        x, y, w, h = row[["X", "Y", "Width", "Height"]]
        bbox_list.append(omidb.mark.BoundingBox(x, y, x + w, y + h))

    # 调用新函数
    bbox_rois, aux_folder, gao, kuan = copy_case_rgb_multi(
        view, client, episode, image_rgb, view[0], bbox_list, slice_index
    )
    del image_rgb
    gc.collect()

    # 构造 record
    ann = []
    for bbox_roi in bbox_rois:
        obj = {
            'bbox': [int(x) for x in bbox_roi],
            "bbox_mode": 0,
            "segmentation": [],
            "category_id": 0,
        }
        ann.append(obj)

    record = {
        "file_name": aux_folder,
        "image_id": image_counter,
        "height": gao,
        "width": kuan,
        "annotations": ann
    }
    image_counter += 1

    dataset_dicts.append(record)




save_dir = 'data/'
new_dict = {i["file_name"]: i for i in dataset_dicts}
with open(save_dir + "sample.json", "w") as outfile:
    json.dump(new_dict, outfile)

