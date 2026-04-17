import multiprocessing
from numpy.lib.function_base import trapz
import General_GPS_Correction
import datetime
import sys
import os
import json
import computer_vision_utils as cv_util
import argparse
import cv2
from PIL import Image, ImageFont, ImageDraw, ImageEnhance, ImageChops, ImageOps


#记录运行时间 输入开始时间和结束时间
def report_time(start, end):
    print("-----------------------------------------------------------")
    print(
        "Start date time: {0}\nEnd date time: {1}\nTotal running time: {2}.".format(
            start, end, end - start
        )
    )

#读取json文件中的内容
def get_anchors_from_json(path):
 
    with open(path, "r") as outfile:
        anchors_dict = json.load(outfile)

    return anchors_dict

#加载配置文件
def load_settings(settings_path):
    with open(settings_path, "r") as f:
        settings_dict = json.load(f)
    General_GPS_Correction.settings.nearest_number = settings_dict["nearest_number"]
    General_GPS_Correction.settings.discard_transformation_perc_inlier = settings_dict[
        "discard_transformation_perc_inlier"
    ]
    General_GPS_Correction.settings.transformation = getattr(
        cv_util.Transformation, settings_dict["transformation"]
    )
    General_GPS_Correction.settings.percentage_next_neighbor = settings_dict[
        "percentage_next_neighbor"
    ]
    General_GPS_Correction.settings.cores_to_use = settings_dict["cores_to_use"]
    General_GPS_Correction.settings.draw_GCPs = settings_dict["draw_GCPs"]

    General_GPS_Correction.settings.sub_set_choosing = settings_dict["sub_set_choosing"]
    General_GPS_Correction.settings.N_perc = settings_dict["N_perc"]
    General_GPS_Correction.settings.E_perc = settings_dict["E_perc"]


def get_args():
    parser = argparse.ArgumentParser(
        description="MegaStitch Drone Stitching and Geo-correction script.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-d",
        "--data",
        help="The path to the data directory.",
        metavar="data",
        default="/home/hipeson/data/zjl/Rice/Rice/2024-06-22_21-34-07/raw_data",
        # default="",
        required=False,
        type=str,
    )

    parser.add_argument(
        "-c",
        "--col",
        help="图像有几行.",
        metavar="col",
        default=8,
        # default="",
        required=False,
        type=str,
    )

    parser.add_argument(
        "-w",
        "--row",
        help="图像有几列",
        metavar="row",
        default=58,
        # default="",
        required=False,
        type=str,
    )

    parser.add_argument(
        "-r",
        "--result",
        help="The path to the directory where the results will be saved.",
        default="/home/hipeson/data/zjl/Rice/result/2024-06-22/",
        # default="/data/lio/Ourss-SIFT/2024-0628-3",

        metavar="result",
        required=False,
        type=str,
    )

    parser.add_argument(
        "-g",
        "--gcp",
        help="The path to Ground Control Points (GCPs) files. Refer to readme for formatting of the json/csv file.",
        metavar="gcp",
        required=False,
        type=str,
    )

    parser.add_argument(
        "-s",
        "--settings",
        help="The path to the json file that contains the configuration/settings information.",
        metavar="settings",
        default="/home/hipeson/zjl/FTP-Stitch-250531/sample_settings.json",
        required=False,
        type=str,
    )

    parser.add_argument(
        "-v",
        "--verbose",
        help="increase output verbosity",
        metavar="verbose",
        default=False,
        required=False,
        type=bool,
    )    

    return parser.parse_args()

def main():

    args = get_args()

    start_time = datetime.datetime.now()
    # subpath = "sum" + "/"
    result = args.result
    # print(result)
    if not os.path.exists(args.result):
        os.makedirs(args.result)

    transformation_path = os.path.join(args.result, "transformation.json")
    ortho_path = os.path.join(args.result, "ortho.png")
    ortho_ori_path = os.path.join(args.result, "ortho_ori.png")

    plot_path = os.path.join(args.result, "initial_GPS.png")
    corrected_coordinates_path = os.path.join(args.result, "corrected_coordinates.json")
    overlap_ratios_path = os.path.join(args.result, "overlap_ratios.txt")
    corrected_homography_path = os.path.join(args.result, "corrected_homography.json")
    log_path = os.path.join(args.result, "log.txt")
    sift_path = os.path.join(args.result, "SIFT")
    result_path = os.path.join(args.result, "Result")

    if not os.path.exists(sift_path):
        os.makedirs(sift_path)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(os.path.join(args.result, "Match")):
        os.makedirs(os.path.join(args.result, "Match"))

    General_GPS_Correction.init_setting(args.data, args.col, args.row)
    General_GPS_Correction.settings.Dataset = os.path.basename(
        os.path.normpath(args.data)
    )

    General_GPS_Correction.settings.AllGCPRMSE = True

    load_settings(args.settings)

    #修改输入流
    if args.verbose == True:
        original = sys.stdout
        log_file = open(log_path, "w")
        sys.stdout = log_file

    

    if hasattr(args, "gcp") and args.gcp is not None:
        anchors_dict = get_anchors_from_json(args.gcp)
    else:
        anchors_dict = None
    

    # 输入目前是第几行
    # stitch_method 0为全图拼接 1为单行拼接，2为单列拼接
    field = General_GPS_Correction.Field(sift_p=sift_path, tr_p=transformation_path,cc_p=corrected_coordinates_path, ch_p=corrected_homography_path,result_p=result_path,working=result,cnt=0,stitch_method=0, or_p = overlap_ratios_path)

    if (
        General_GPS_Correction.settings.transformation
        == cv_util.Transformation.similarity
    ):

        (
            coords_dict,
            H,
            H_inv,
            abs_tr,
            _,
            _,
            _,
        ) = field.geo_correct_MegaStitchSimilarity(anchors_dict)
    elif (
        General_GPS_Correction.settings.transformation == cv_util.Transformation.affine
    ):
        (
            coords_dict,
            H,
            H_inv,
            abs_tr,
            _,
            _,
        ) = field.geo_correct_MegaStitchAffine(anchors_dict, None)
    elif (
        General_GPS_Correction.settings.transformation
        == cv_util.Transformation.homography
    ):
        if (
            General_GPS_Correction.settings.preprocessing_transformation.lower()
            == "none"
        ):
            (
                coords_dict,
                H,
                H_inv,
                abs_tr,
                _,
                _,
            ) = field.geo_correct_BundleAdjustment_Homography(anchors_dict, None)
        elif (
            General_GPS_Correction.settings.preprocessing_transformation.lower()
            == "similarity"
        ):
            (
                coords_dict,
                H,
                H_inv,
                abs_tr,
                _,
                _,
            ) = field.geo_correct_MegaStitch_Similarity_Bundle_Adjustment_Homography(
                anchors_dict, None
            )
        elif (
            General_GPS_Correction.settings.preprocessing_transformation.lower()
            == "affine"
        ):
            (
                coords_dict,
                H,
                H_inv,
                abs_tr,
                _,
                _,
            ) = field.geo_correct_MegaStitch_Affine_Bundle_Adjustment_Homography(
                anchors_dict, None
            )

    if H is None:
        gcp_inf = None
    else:
        gcp_inf = (anchors_dict, H_inv, abs_tr)
    #生成变换地图，颜色越绿代表内点百分比越高
    field.generate_transformation_accuracy_histogram(
        coords_dict, plot_path.replace("initial_GPS", "transformation_plot")
    )
    print("coords_dict为{0}, plot_path为{1}".format(coords_dict, plot_path))
    #生成GPS地图
    field.visualize_field_centers(True,plot_path)
    
    # field.save_field_centers_visualization(plot_path)
    field.save_absolute_H(corrected_homography_path,abs_tr,coords_dict)
    field.save_field_coordinates(corrected_coordinates_path, coords_dict)
    
    # ortho = field.generate_field_ortho(coords_dict, gcp_info=gcp_inf,abs_tr=abs_tr)

    # field.save_field_ortho(ortho, ortho_path)

    # field.save_field_ortho(ortho_ori, ortho_ori_path)

    end_time = datetime.datetime.now()

    report_time(start_time, end_time)
    if args.verbose == True:
        sys.stdout = original
        log_file.close()

if __name__ == "__main__":
    main()
    

