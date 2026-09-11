# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved. 
#   
# Licensed under the Apache License, Version 2.0 (the "License");   
# you may not use this file except in compliance with the License.  
# You may obtain a copy of the License at   
#   
#     http://www.apache.org/licenses/LICENSE-2.0    
#   
# Unless required by applicable law or agreed to in writing, software   
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
# See the License for the specific language governing permissions and   
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import itertools

from ppdet.metrics.json_results import get_det_res, get_det_poly_res, get_seg_res, get_solov2_segm_res, get_keypoint_res, get_pose3d_res
from ppdet.metrics.map_utils import draw_pr_curve

from ppdet.utils.logger import setup_logger
logger = setup_logger(__name__)


def _plain_ascii_table(rows):
    """Format a small table without the optional terminaltables package."""
    normalized_rows = [["" if value is None else str(value) for value in row]
                       for row in rows]
    if not normalized_rows:
        return ""
    column_count = max(len(row) for row in normalized_rows)
    for row in normalized_rows:
        row.extend([""] * (column_count - len(row)))
    widths = [
        max(len(row[column]) for row in normalized_rows)
        for column in range(column_count)
    ]
    separator = '+' + '+'.join('-' * (width + 2)
                               for width in widths) + '+'
    output = [separator]
    for row_index, row in enumerate(normalized_rows):
        output.append('| ' + ' | '.join(
            value.ljust(widths[column])
            for column, value in enumerate(row)) + ' |')
        if row_index == 0:
            output.append(separator)
    output.append(separator)
    return '\n'.join(output)


def get_infer_results(outs, catid, bias=0):
    """
    Get result at the stage of inference.
    The output format is dictionary containing bbox or mask result.

    For example, bbox result is a list and each element contains
    image_id, category_id, bbox and score.
    """
    if outs is None or len(outs) == 0:
        raise ValueError(
            'The number of valid detection result if zero. Please use reasonable model and check input data.'
        )

    im_id = outs['im_id']

    infer_res = {}
    if 'bbox' in outs:
        if len(outs['bbox']) > 0 and len(outs['bbox'][0]) > 6:
            infer_res['bbox'] = get_det_poly_res(
                outs['bbox'], outs['bbox_num'], im_id, catid, bias=bias)
        else:
            infer_res['bbox'] = get_det_res(
                outs['bbox'], outs['bbox_num'], im_id, catid, bias=bias)

    if 'mask' in outs:
        # mask post process
        infer_res['mask'] = get_seg_res(outs['mask'], outs['bbox'],
                                        outs['bbox_num'], im_id, catid)

    if 'segm' in outs:
        infer_res['segm'] = get_solov2_segm_res(outs, im_id, catid)

    if 'keypoint' in outs:
        infer_res['keypoint'] = get_keypoint_res(outs, im_id)
        outs['bbox_num'] = [len(infer_res['keypoint'])]

    if 'pose3d' in outs:
        infer_res['pose3d'] = get_pose3d_res(outs, im_id)
        outs['bbox_num'] = [len(infer_res['pose3d'])]

    return infer_res

def calculate_ap50(precision, recall):
    """
    计算给定精确度（precision）和召回率（recall）数组下的 mAP50。

    Args:
        precision (numpy.ndarray): 精确度数组。
        recall (numpy.ndarray): 召回率数组。

    Returns:
        float: mAP50 的值。
    """
    num_points = precision.shape[0]
    recall_thresholds = np.arange(0.0, 1.01, 0.01)
    precisions_at_recalls = np.zeros_like(recall_thresholds)
    for i, t in enumerate(recall_thresholds):
        recalls_above_t = recall[recall >= t]
        if recalls_above_t.size > 0:
            precisions_above_t = precision[recall >= t]
            precisions_at_recalls[i] = np.max(precisions_above_t)
        else:
            precisions_at_recalls[i] = 0.0
    ap50 = np.mean(precisions_at_recalls)
    return ap50


def cocoapi_eval(jsonfile,
                 style,
                 coco_gt=None,
                 anno_file=None,
                 max_dets=(100, 300, 1000),
                 classwise=False,
                 sigmas=None,
                 use_area=True):
    """
    Args:
        jsonfile (str): Evaluation json file, eg: bbox.json, mask.json.
        style (str): COCOeval style, can be `bbox` , `segm` , `proposal`, `keypoints` and `keypoints_crowd`.
        coco_gt (str): Whether to load COCOAPI through anno_file,
                 eg: coco_gt = COCO(anno_file)
        anno_file (str): COCO annotations file.
        max_dets (tuple): COCO evaluation maxDets.
        classwise (bool): Whether per-category AP and draw P-R Curve or not.
        sigmas (nparray): keypoint labelling sigmas.
        use_area (bool): If gt annotations (eg. CrowdPose, AIC)
                         do not have 'area', please set use_area=False.
    """
    assert coco_gt != None or anno_file != None
    if style == 'keypoints_crowd':
        #please install xtcocotools==1.6
        from xtcocotools.coco import COCO
        from xtcocotools.cocoeval import COCOeval
    else:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

    if coco_gt == None:
        coco_gt = COCO(anno_file)
    logger.info("Start evaluate...")
    coco_dt = coco_gt.loadRes(jsonfile)
    if style == 'proposal':
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.params.useCats = 0
        coco_eval.params.maxDets = list(max_dets)
    elif style == 'keypoints_crowd':
        coco_eval = COCOeval(coco_gt, coco_dt, style, sigmas, use_area)
    else:
        coco_eval = COCOeval(coco_gt, coco_dt, style)
        if style == 'bbox':
            # The competition caps detections at 100 per image.  Make the
            # otherwise implicit pycocotools defaults explicit and verifiable.
            coco_eval.params.maxDets = [1, 10, 100]
            assert len(coco_eval.params.recThrs) == 101
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    if style == 'bbox':
        # Extra diagnostics only: they do not alter COCO matching or scores.
        image_ids = coco_gt.getImgIds()
        prediction_counts = {}
        for det in coco_dt.dataset.get('annotations', []):
            image_id = det['image_id']
            prediction_counts[image_id] = prediction_counts.get(image_id, 0) + 1
        image_count = max(len(image_ids), 1)
        raw_prediction_count = sum(prediction_counts.values())
        effective_prediction_count = sum(
            min(prediction_counts.get(image_id, 0), 100)
            for image_id in image_ids)

        # Report a threshold-independent P/R diagnostic from the IoU=0.50 PR
        # curves.  For each category choose its best-F1 point, then macro-average
        # the corresponding precision and recall.  AP/AR remain the official
        # metrics printed by COCOeval above.
        precision_iou50 = coco_eval.eval['precision'][0, :, :, 0, -1]
        recall_grid = coco_eval.params.recThrs
        best_precisions = []
        best_recalls = []
        for category_index in range(precision_iou50.shape[1]):
            category_precision = precision_iou50[:, category_index]
            valid = category_precision >= 0
            if not np.any(valid):
                continue
            precision_values = category_precision[valid]
            recall_values = recall_grid[valid]
            f1_values = (2.0 * precision_values * recall_values /
                         np.maximum(precision_values + recall_values, 1e-12))
            best_index = int(np.argmax(f1_values))
            best_precisions.append(float(precision_values[best_index]))
            best_recalls.append(float(recall_values[best_index]))

        macro_precision = (float(np.mean(best_precisions))
                           if best_precisions else float('nan'))
        macro_recall = (float(np.mean(best_recalls))
                        if best_recalls else float('nan'))
        logger.info(
            'AIC2026 diagnostics: raw predictions/image={:.3f}, '
            'evaluated predictions/image={:.3f} (maxDets=100), '
            'macro P/R at per-class best-F1 on IoU=0.50={:.4f}/{:.4f}.'.format(
                raw_prediction_count / image_count,
                effective_prediction_count / image_count,
                macro_precision,
                macro_recall))
    if classwise:
        # Compute per-category AP and PR curve
        try:
            from terminaltables import AsciiTable
        except ImportError:
            AsciiTable = None
            logger.warning(
                'terminaltables is not installed; using the built-in '
                'dependency-free table formatter.')
        precisions = coco_eval.eval['precision']
        cat_ids = coco_gt.getCatIds()
        # precision: (iou, recall, cls, area range, max dets)
        assert len(cat_ids) == precisions.shape[2]
        results_per_category = []
        ap50_per_category = []  # 存储每个类别的 mAP50 结果
        curve_output_dir = os.path.join(
            os.path.dirname(os.path.abspath(jsonfile)),
            style + '_pr_curve')
        for idx, catId in enumerate(cat_ids):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            nm = coco_gt.loadCats(catId)[0]
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]

            precision_50 = precisions[0, :, idx, 0, -1]
            precision_50 = precision_50[precision_50 > -1]
            if precision.size:
                ap = np.mean(precision)
                ap50 = np.mean(precision_50)
            else:
                ap = float('nan')
                ap50 = float('nan')
            # 计算每个类别的 mAP50
            recall_array = np.arange(0.0, 1.01, 0.01)
            # ap50 = calculate_ap50(precision, recall_array, iou_threshold=0.5)
            # ap50_per_category.append(ap50)
            results_per_category.append(
                (str(nm["name"]), '{:0.3f}'.format(float(ap)),'{:0.3f}'.format(float(ap50))))
            pr_array = precisions[0, :, idx, 0, 2]
            # recall_array = np.arange(0.0, 1.01, 0.01)
            draw_pr_curve(
                pr_array,
                recall_array,
                out_dir=curve_output_dir,
                file_name='{}_precision_recall_curve.jpg'.format(nm["name"]))

        # # 获取类别数和IOU阈值数
        # num_classes = precisions.shape[2]
        # print(num_classes)
        # num_iou_thresholds = precisions.shape[0]
        # print(num_iou_thresholds)
        # for class_idx in range(num_classes):
        #     class_precisions = precisions[:, :, class_idx, 0, -1]
        #
        #     # 计算每个IOU阈值下的最大精确度
        #     max_precisions = np.max(class_precisions, axis=1)
        #     # 计算每个类别的 AP
        #     ap = np.mean(max_precisions)
        #     print('ap :',ap)
        #
        #     # 计算每个类别的 mAP50
        #     ap50 = np.mean(max_precisions[:num_iou_thresholds // 2])
        #     ap50_per_category .append(ap50)
        #     results_per_category[class_idx] = results_per_category[class_idx] + ('{:0.3f}'.format(float(ap50)),)


        num_columns = min(9, len(results_per_category) * 3)
        results_flatten = list(itertools.chain(*results_per_category))
        headers = ['category', 'AP', 'mAP50'] * (num_columns // 3)
        results_2d = itertools.zip_longest(
            *[results_flatten[i::num_columns] for i in range(num_columns)])
        table_data = [headers]
        table_data += [result for result in results_2d]
        table_text = (AsciiTable(table_data).table if AsciiTable is not None
                      else _plain_ascii_table(table_data))
        logger.info('Per-category of {} AP and mAP50: \n{}'.format(
            style, table_text))
        #logger.info('lamr_iou50: ',float(coco_eval.eval['lamr_iou50']))
        #print('lamr_iou50: ',coco_eval.eval['lamr_iou50'][0])
        logger.info("per-category PR curve has output to {} folder.".format(
            curve_output_dir))
        # precisions = coco_eval.eval['precision']
        # cat_ids = coco_gt.getCatIds()
        # # precision: (iou, recall, cls, area range, max dets)
        # assert len(cat_ids) == precisions.shape[2]
        # results_per_category = []
        # for idx, catId in enumerate(cat_ids):
        #     # area range index 0: all area ranges
        #     # max dets index -1: typically 100 per image
        #     nm = coco_gt.loadCats(catId)[0]
        #     precision = precisions[:, :, idx, 0, -1]
        #     precision = precision[precision > -1]
        #     if precision.size:
        #         ap = np.mean(precision)
        #     else:
        #         ap = float('nan')
        #     results_per_category.append(
        #         (str(nm["name"]), '{:0.3f}'.format(float(ap))))
        #     pr_array = precisions[0, :, idx, 0, 2]
        #     recall_array = np.arange(0.0, 1.01, 0.01)
        #     draw_pr_curve(
        #         pr_array,
        #         recall_array,
        #         out_dir=style + '_pr_curve',
        #         file_name='{}_precision_recall_curve.jpg'.format(nm["name"]))
        #
        # num_columns = min(6, len(results_per_category) * 2)
        # results_flatten = list(itertools.chain(*results_per_category))
        # headers = ['category', 'AP'] * (num_columns // 2)
        # results_2d = itertools.zip_longest(
        #     * [results_flatten[i::num_columns] for i in range(num_columns)])
        # table_data = [headers]
        # table_data += [result for result in results_2d]
        # table = AsciiTable(table_data)
        # logger.info('Per-category of {} AP: \n{}'.format(style, table.table))
        # logger.info("per-category PR curve has output to {} folder.".format(
        #     style + '_pr_curve'))
    # flush coco evaluation result
    sys.stdout.flush()
    return coco_eval.stats


def json_eval_results(metric, json_directory, dataset):
    """
    cocoapi eval with already exists proposal.json, bbox.json or mask.json
    """
    assert metric == 'COCO'
    anno_file = dataset.get_anno()
    json_file_list = ['proposal.json', 'bbox.json', 'mask.json']
    if json_directory:
        assert os.path.exists(
            json_directory), "The json directory:{} does not exist".format(
                json_directory)
        for k, v in enumerate(json_file_list):
            json_file_list[k] = os.path.join(str(json_directory), v)

    coco_eval_style = ['proposal', 'bbox', 'segm']
    for i, v_json in enumerate(json_file_list):
        if os.path.exists(v_json):
            cocoapi_eval(v_json, coco_eval_style[i], anno_file=anno_file)
        else:
            logger.info("{} not exists!".format(v_json))
