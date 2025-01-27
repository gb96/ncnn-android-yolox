// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "yolox.h"

#include <iostream>
#include <string>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "cpu.h"
#include "cv_param.h"


static const cv::Scalar ccBlack = cv::Scalar(0, 0, 0);
static const cv::Scalar ccWhite = cv::Scalar(255, 255, 255);
static const cv::Scalar ccRed = cv::Scalar(255, 0, 0);
static const cv::Scalar ccOrange = cv::Scalar(255, 165, 0);
static const cv::Scalar ccGreen = cv::Scalar(0, 255, 0);
static const cv::Size zoom_px_size = cv::Size(ZOOM, ZOOM);


// YOLOX use the same focus in yolov5
class YoloV5Focus : public ncnn::Layer
{
public:
    YoloV5Focus()
    {
        one_blob_only = true;
    }

    virtual int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;

        int outw = w / 2;
        int outh = h / 2;
        int outc = channels * 4;

        top_blob.create(outw, outh, outc, 4u, 1, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < outc; p++)
        {
            const float* ptr = bottom_blob.channel(p % channels).row((p / channels) % 2) + ((p / channels) / 2);
            float* outptr = top_blob.channel(p);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    *outptr = *ptr;

                    outptr += 1;
                    ptr += 2;
                }

                ptr += w;
            }
        }

        return 0;
    }
};

DEFINE_LAYER_CREATOR(YoloV5Focus)


struct GridAndStride
{
    int grid0;
    int grid1;
    int stride;
};

static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& detectedobjs, int left, int right)
{
    int i = left;
    int j = right;
    float p = detectedobjs[(left + right) / 2].prob;

    while (i <= j)
    {
        while (detectedobjs[i].prob > p)
            i++;

        while (detectedobjs[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(detectedobjs[i], detectedobjs[j]);

            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(detectedobjs, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(detectedobjs, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& detectedobjs, std::vector<int>& picked, float nms_threshold)
{
    float min_nms = 1.0;
    float max_nms = 0.0;
    float sum_nms = 0.0;
    int nms_count = 0;
    int discard_count = 0;

    const int n = detectedobjs.size();

    picked.clear();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = detectedobjs[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = detectedobjs[i];

        int keep = 1;
        for (int j : picked)
        {
            const Object& b = detectedobjs[j];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[j] - inter_area;
            float IoU = inter_area / union_area;  //  range is 0 .. 1
            max_nms = std::max(max_nms, IoU);
            min_nms = std::min(min_nms, IoU);
            sum_nms += IoU;
            nms_count++;
            if (IoU > nms_threshold) {
                // discard additional detections that overlap existing higher-probability ones
                keep = 0;
                discard_count++;
            }
        }

        if (keep)
            picked.push_back(i);
    }
//    if (nms_count != 0) {
//        float avg_nms = sum_nms/nms_count;
//        __android_log_print(ANDROID_LOG_INFO, "yolox", "nms_sorted_bboxes picked=%d/%d discarded=%d max_nms=%.2f min_nms=%.2f avg_nms=%.2f", picked.size(), n, discard_count, max_nms, min_nms, avg_nms);
//    }

}

static int generate_grids_and_stride(const int target_size, std::vector<int>& strides, std::vector<GridAndStride>& grid_strides)
{
    for (auto stride : strides)
    {
        int num_grid = target_size / stride;
        for (int g1 = 0; g1 < num_grid; g1++)
        {
            for (int g0 = 0; g0 < num_grid; g0++)
            {
                grid_strides.push_back((GridAndStride){g0, g1, stride});
            }
        }
    }

    return 0;
}

static void generate_yolox_proposals(std::vector<GridAndStride> grid_strides, const ncnn::Mat& feat_blob, float prob_threshold, std::vector<Object>& objects)
{
//    const int num_grid = feat_blob.h;
    // fprintf(stderr, "output height: %d, width: %d, channels: %d, dims:%d\n", feat_blob.h, feat_blob.w, feat_blob.c, feat_blob.dims);

//    const int num_class = feat_blob.w - 5;

    const int num_anchors = grid_strides.size();

    const float* feat_ptr = feat_blob.channel(0);
    for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++)
    {
        const int grid0 = grid_strides[anchor_idx].grid0;
        const int grid1 = grid_strides[anchor_idx].grid1;
        const int stride = grid_strides[anchor_idx].stride;

        // yolox/models/yolo_head.py decode logic
        //  outputs[..., :2] = (outputs[..., :2] + grids) * strides
        //  outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        float x_center = (feat_ptr[0] + grid0) * stride;
        float y_center = (feat_ptr[1] + grid1) * stride;
        float w = exp(feat_ptr[2]) * stride;
        float h = exp(feat_ptr[3]) * stride;
        float x0 = x_center - w * 0.5f;
        float y0 = y_center - h * 0.5f;

        float box_objectness = feat_ptr[4];
        // preference detections that are more central horizontally
        float central_bias_x = 0.0f;
        float size_bias = 0.0f;

        // for (int class_idx = 0; class_idx < num_class; class_idx++)
        {
            // instead of checking detections for all classes,
            // only look for a single target class
            int class_idx = TARGET_OBJECT_CLASS_IDX;

            float box_cls_score = feat_ptr[5 + class_idx];
            float box_prob = box_objectness * box_cls_score;
            if (box_prob > prob_threshold)
            {
                central_bias_x = 1.2f - abs(x_center - CAMERA_CENTER_X) / CAMERA_RES_WIDTH;
                size_bias = 0.07f * w;
                Object obj;
                obj.rect.x = x0;
                obj.rect.y = y0;
                obj.rect.width = w;
                obj.rect.height = h;
                obj.label = class_idx;
                obj.prob = box_prob * central_bias_x * size_bias;

                objects.push_back(obj);
//                __android_log_print(ANDROID_LOG_INFO, "yolox", "detection %d/%d x: %.1f y: %.1f height: %.1f, width: %.1f, central_bias_x: %.1f, size_bias: %.1f, prob: %0.2f, channels: %d, dims:%d\n", anchor_idx, num_anchors, x_center, y_center, h, w, central_bias_x, size_bias, obj.prob, feat_blob.c, feat_blob.dims);
//                __android_log_print(ANDROID_LOG_INFO, "yolox", "detection %d/%d x: %.1f y: %.1f height: %.1f, width: %.1f, central_bias_x: %.1f, size_bias: %.1f, prob: %0.2f\n", anchor_idx, num_anchors, x_center, y_center, h, w, central_bias_x, size_bias, obj.prob);
//                __android_log_print(ANDROID_LOG_INFO, "yolox", "detection %d/%d grid0: %d, central_bias_x: %.1f, size_bias: %.1f, prob: %0.2f\n", anchor_idx, num_anchors, grid0, central_bias_x, size_bias, obj.prob);
            }

        } // class loop
        feat_ptr += feat_blob.w;

    } // point anchor loop
}
 
 
Yolox::Yolox()
{
    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);
}

int Yolox::load(const char* modeltype, int _target_size, const float* _mean_vals, const float* _norm_vals, bool use_gpu)
{
    yolox.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    yolox.opt = ncnn::Option();
    yolox.opt.num_threads = ncnn::get_big_cpu_count();
    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(yolox.opt.num_threads);

#if NCNN_VULKAN
    yolox.opt.use_vulkan_compute = use_gpu;
#endif
    yolox.register_custom_layer("YoloV5Focus", YoloV5Focus_layer_creator);
    yolox.opt.blob_allocator = &blob_pool_allocator;
    yolox.opt.workspace_allocator = &workspace_pool_allocator;

    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "%s.param", modeltype);
    sprintf(modelpath, "%s.bin", modeltype);

    yolox.load_param(parampath);
    yolox.load_model(modelpath);

    target_size = _target_size;
    mean_vals[0] = _mean_vals[0];
    mean_vals[1] = _mean_vals[1];
    mean_vals[2] = _mean_vals[2];
    norm_vals[0] = _norm_vals[0];
    norm_vals[1] = _norm_vals[1];
    norm_vals[2] = _norm_vals[2];

    frame_num = 0;

    return 0;
}

int Yolox::load(AAssetManager* mgr, const char* modeltype, int _target_size, const float* _mean_vals, const float* _norm_vals, bool use_gpu)
{
    yolox.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    yolox.opt = ncnn::Option();
    yolox.opt.num_threads = ncnn::get_big_cpu_count();
    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(yolox.opt.num_threads);

#if NCNN_VULKAN
    yolox.opt.use_vulkan_compute = use_gpu;
#endif
    yolox.register_custom_layer("YoloV5Focus", YoloV5Focus_layer_creator);
    yolox.opt.blob_allocator = &blob_pool_allocator;
    yolox.opt.workspace_allocator = &workspace_pool_allocator;

    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "%s.param", modeltype);
    sprintf(modelpath, "%s.bin", modeltype);

    yolox.load_param(mgr, parampath);
    yolox.load_model(mgr, modelpath);


    target_size = _target_size;
    mean_vals[0] = _mean_vals[0];
    mean_vals[1] = _mean_vals[1];
    mean_vals[2] = _mean_vals[2];
    norm_vals[0] = _norm_vals[0];
    norm_vals[1] = _norm_vals[1];
    norm_vals[2] = _norm_vals[2];

    frame_num = 0;

    return 0;
}


int Yolox::detect(const cv::Mat& rgb, std::vector<Object>& objects, float prob_threshold, float nms_threshold)
{
//    __android_log_print(ANDROID_LOG_INFO, "Yolox", "detect(rows=%d, cols=%d)", rgb.rows, rgb.cols);
    int img_w = rgb.cols;
    int img_h = rgb.rows;

    // letterbox pad to multiple of 32
    int w = img_w;
    int h = img_h;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }
     ncnn::Mat inRgb = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB, img_w, img_h, w, h);
//    ncnn::Mat inBgr = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB2BGR, img_w, img_h, w, h);

    int kernel_size = 3;
//    int ddepth = CV_16S;

//    cv::Mat laplacian;
    cv::Mat src_grey;
    // Reduce noise by blurring with a Gaussian filter ( kernel size = 5 )

//    cv::GaussianBlur(rgb, rgb, cv::Size(kernel_size, kernel_size), 0, 0 );
    cvtColor(rgb, src_grey, cv::COLOR_BGR2GRAY); // Convert the image to grayscale
    medianBlur(src_grey, src_grey, 5);
//    std::vector<cv::Vec3f> circles;
//    HoughCircles(src_grey, circles, cv::HOUGH_GRADIENT, 1,
//                 src_grey.rows/16,  // change this value to detect circles with different distances to each other
//                 100, 30, 2, 30 // change the last two parameters
//            // (min_radius & max_radius) to detect larger circles
//    );
//    for ( auto c : circles ) {
//        cv::Point center = cv::Point(c[0], c[1]);
//        // circle center
//        // circle( rgb, center, 1, ccBlack, 3, cv::LINE_AA);
//        // circle outline
//        int radius = c[2];
//        circle( rgb, center, radius, ccWhite, 2, cv::LINE_AA);
//    }

//    Laplacian(src_grey, laplacian, ddepth, kernel_size);

    // pad to target_size rectangle
    // yolov5/utils/datasets.py letterbox
    int wpad = target_size-w;//(w + 31) / 32 * 32 - w;
    int hpad = target_size-h;//(h + 31) / 32 * 32 - h;
//    __android_log_print(ANDROID_LOG_INFO, "Yolox", "target_size=%d wpad=%d hpad=%d", target_size, wpad, hpad);
    ncnn::Mat in_pad;
    ncnn::copy_make_border(inRgb, in_pad, 0, hpad, 0, wpad, ncnn::BORDER_CONSTANT, 114.f); // 114.f

    // so for 0-255 input image, rgb_mean should multiply 255 and norm should div by std.
    // new release of yolox has deleted this preprocess,if you are using new release please don't use this preprocess.
    in_pad.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = yolox.create_extractor();

    ex.input("images", in_pad);

    std::vector<Object> proposals;

    {
        ncnn::Mat out;
        ex.extract("output", out);

        std::vector<int> strides = {8}; // {8, 16, 32} might have stride=64 FIXME
        std::vector<GridAndStride> grid_strides;
        generate_grids_and_stride(target_size, strides, grid_strides);
        generate_yolox_proposals(grid_strides, out, prob_threshold, proposals);
    }

    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    size_t size = picked.size();

    objects.resize(size);
    int detCount = 0;
    for (size_t i = 0; i < size; i++)
    {
        objects[i] = proposals[picked[i]];
        if (objects[i].label != TARGET_OBJECT_CLASS_IDX) continue;

        detCount++;

        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x) / scale;
        float y0 = (objects[i].rect.y) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height) / scale;

        // clip
        x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
//        int _w = std::round(objects[i].rect.width);
//        int _h = std::round(objects[i].rect.height);
        objects[i].rect.width = ZOOM * (x1 - x0);
        objects[i].rect.height = ZOOM * (y1 - y0);

//        cv::Mat src_grey_aoi = src_grey.colRange(x0, x0 + _w).rowRange(y0, y0 + _h);
//        std::vector<cv::Vec3f> circles;
//        HoughCircles(src_grey_aoi, circles, cv::HOUGH_GRADIENT, 1,
//                     5,  // change this value to detect circles with different distances to each other
//                     200, 90, 3, 0 // change the last two parameters
//                // (min_radius & max_radius) to detect larger circles
//        );
//        for ( auto c : circles ) {
//            cv::Point center = cv::Point(c[0], c[1]);
//            // circle center
//            // circle( rgb, center, 1, ccBlack, 3, cv::LINE_AA);
//            // circle outline
//            int radius = c[2];
//            circle( rgb, center, radius, ccWhite, 3, cv::LINE_AA);
//        }
    }

//    if (detCount != 0) {
//        // save image
//        char filepath[128];
//        sprintf(filepath, "/sdcard/Pictures/rgb_%d.jpg", frame_num);
//        cv::Mat bgr;
//        cvtColor(rgb, bgr, cv::COLOR_RGB2BGR); // Convert the RGB image to BGR
//        bool result = cv::imwrite(filepath, bgr);
////        __android_log_print(ANDROID_LOG_INFO, "yolox", "imwrite frame_num=%d result=%d", frame_num, result);
//    }
    frame_num++;

    return 0;
}

int Yolox::draw(cv::Mat& rgb, const std::vector<Object>& objects)
{
//    static const char* class_names[] = {
//            "00 person", "01 bicycle", "02 car", "03 motorcycle", "04 airplane", "05 bus", "06 train", "07 truck", "08 boat", "09 traffic light",
//            "10 fire hydrant", "11 stop sign", "12 parking meter", "13 bench", "14 bird", "15 cat", "16 dog", "17 horse", "18 sheep", "19 cow",
//            "20 elephant", "21 bear", "22 zebra", "23 giraffe", "24 backpack", "25 umbrella", "26 handbag", "27 tie", "28 suitcase", "29 frisbee",
//            "30 skis", "31 snowboard", "32 sports ball", "33 kite", "34 baseball bat", "35 baseball glove", "36 skateboard", "37 surfboard",
//            "38 tennis racket", "39 bottle", "40 wine glass", "41 cup", "42 fork", "43 knife", "44 spoon", "45 bowl", "46 banana", "47 apple",
//            "48 sandwich", "49 orange", "50 broccoli", "51 carrot", "52 hot dog", "53 pizza", "54 donut", "55 cake", "56 chair", "57 couch",
//            "58 potted plant", "59 bed", "60 dining table", "61 toilet", "62 tv", "63 laptop", "64 mouse", "65 remote", "66 keyboard", "67 cell phone",
//            "68 microwave", "69 oven", "70 toaster", "71 sink", "72 refrigerator", "73 book", "74 clock", "75 vase", "76 scissors", "77 teddy bear",
//            "78 hair drier", "79 toothbrush"
//    };

    int detect_count = 0;
    float min_prob = 1.0;
    float max_prob = 0.0;
    float sum_prob = 0.0;

    const cv::Mat rgbClone = rgb.clone();
    char* detect_info[TARGET_OBJECT_MAX_DETECT_COUNT];

    for (const auto & obj : objects)
    {
        // display at most TARGET_OBJECT_MAX_DETECT_COUNT detected traffic lights,
        // the most probable detections
        if (detect_count >= TARGET_OBJECT_MAX_DETECT_COUNT) break;

        // only count and display traffic light detections
        if (obj.label != TARGET_OBJECT_CLASS_IDX) continue;

        // track maximum probability, will typically be the first object due to sort
        max_prob = std::max(obj.prob, max_prob);
        min_prob = std::min(obj.prob, min_prob);
        sum_prob += obj.prob;

//        const unsigned char* color = colors[detect_count % 19];
        detect_count++;

        int x = obj.rect.x;
        int y = obj.rect.y;

        int wZoom = obj.rect.width;
        int hZoom = obj.rect.height;
        int wOrig = wZoom/ZOOM;
        int hOrig = hZoom/ZOOM;
        int x2 = x + wOrig;
        int y2 = y + hOrig;
        int x2zoom = x + wZoom;
        int y2zoom = y + hZoom;

        // clip zoom output to image limits:
        if (x2zoom >= rgb.cols) {
            x2zoom = rgb.cols - 1;
            wZoom = x2zoom - x;
        }
        if (y2zoom >= rgb.rows) {
            y2zoom = rgb.rows - 1;
            hZoom = y2zoom - y;
        }

        int redSum = 0;
        int greenSum = 0;
        int blueSum = 0;

        int redHueCount = 0;
        int amberHueCount = 0;
        int greenHueCount = 0;
        int otherHueCount = 0;

        const int doubleWidth = 2 * wOrig;

        for (int row = y, rz = y; row < y2; row++, rz += 2) {
            uchar *p = rgbClone.data + (row * rgb.cols + x) * 3;
            for (int col = x, cz = x; col < x2; col++, cz += 2) {
                const uchar r = p[0];
                const uchar g = p[1];
                const uchar b = p[2];
                cv::Scalar pixelColor(r, g, b);

                const int testCol1 = 3 * (col - x);
                if (testCol1 > wOrig && testCol1 < 2 * doubleWidth) {
                    // Sample pixel colour in centre third of pixel columns
                    redSum += r;
                    greenSum += g;
                    blueSum += b;

                    // Hue calculation from RGB:
                    int h = 0;
                    if (r != g || b != g) {
                        int v = r;  // max(r, g, b)
                        if (g > v) v = g;
                        if (b > v) v = b;
                        int w = r;  // min(r, g, b)
                        if (g < w) w = g;
                        if (b < w) w = b;

                        if (v == r) {
                            int gi = g;
                            int bi = b;
                            h = 60 * (gi - bi) / (v - w);
                        } else if (v == g) {
                            int bi = b;
                            int ri = r;
                            h = 120 + 60 * (bi - ri) / (v - w);
                        } else {
                            // v == b
                            int ri = r;
                            int gi = g;
                            h = 240 + 60 * (ri - gi) / (v - w);
                        }
                        if (h < 0) h += 360;
                    }

                    if (h >= 359 || h <= 10) {
                        // Traffic Light Red
                        redHueCount += 2;
                    } else if (h >= 358 || h <= 26) {
                        // off-Traffic Light Red
                        redHueCount++;
                    } else if (h >= 55 && h <= 61) {
                        // Warm yellow
                        amberHueCount+=2;
                    } else if (h >= 176 && h <= 180) {
                        // Traffic light cyan green
                        greenHueCount+=2;
                    } else if (h >= 170 && h <= 182) {
                        // Cyan green
                        greenHueCount++;
                    } else {
                        // Other hue
                        otherHueCount++;
                    }
                }

                cv::rectangle(rgb, cv::Rect(cv::Point(cz, rz), zoom_px_size), pixelColor, -1);
                p += 3;
            }
        }
        if (redSum >= greenSum) {
            if (redHueCount > amberHueCount) {
                // Red light
                cv::rectangle(rgb, obj.rect, ccRed, TARGET_OBJECT_BORDER_THICKNESS);
//                __android_log_print(ANDROID_LOG_ERROR, "ncnn", "RED   redSum=%d, greenSum=%d, blueSum=%d, redHue=%d, amberHue=%d, greenHue=%d", redSum, greenSum, blueSum, redHueCount, amberHueCount, greenHueCount);
            } else {
                // Amber light
                cv::rectangle(rgb, obj.rect, ccOrange, TARGET_OBJECT_BORDER_THICKNESS);
//                __android_log_print(ANDROID_LOG_ERROR, "ncnn", "AMBER redSum=%d, greenSum=%d, blueSum=%d, redHue=%d, amberHue=%d, greenHue=%d", redSum, greenSum, blueSum, redHueCount, amberHueCount, greenHueCount);
            }
        } else {
            if (greenHueCount > amberHueCount || greenHueCount > redHueCount || greenSum > blueSum) {
                // Green light
                cv::rectangle(rgb, obj.rect, ccGreen, TARGET_OBJECT_BORDER_THICKNESS);
//                __android_log_print(ANDROID_LOG_ERROR, "ncnn", "GREEN redSum=%d, greenSum=%d, blueSum=%d, redHue=%d, amberHue=%d, greenHue=%d", redSum, greenSum, blueSum, redHueCount, amberHueCount, greenHueCount);
            } else {
                // No Red/Amber/Green?
                cv::rectangle(rgb, obj.rect, ccBlack, TARGET_OBJECT_BORDER_THICKNESS);
//                __android_log_print(ANDROID_LOG_ERROR, "ncnn", "BLACK redSum=%d, greenSum=%d, blueSum=%d, redHue=%d, amberHue=%d, greenHue=%d", redSum, greenSum, blueSum, redHueCount, amberHueCount, greenHueCount);
            }
        }
        detect_info[detect_count - 1] = new char[128];
        sprintf(detect_info[detect_count - 1], "prob: %f, redSum: %d, greenSum: %d, blueSum %d, redHue: %d, amberHue: %d, greenHue: %d",
                obj.prob, redSum, greenSum, blueSum, redHueCount, amberHueCount, greenHueCount);
    }

    if (detect_count != 0) {
//        float avg_prob = sum_prob/detect_count;
//        __android_log_print(ANDROID_LOG_INFO, "yolox", "draw n=%d max_prob=%.2f min_prob=%.2f avg_prob=%.2f", detect_count, max_prob, min_prob, avg_prob);

        // save image
        char filepath[128];
        sprintf(filepath, "/sdcard/Pictures/Lights/rgb_%d_det.jpg", frame_num);
        cv::Mat bgr;
        cvtColor(rgb, bgr, cv::COLOR_RGB2BGR); // Convert the RGB image to BGR
        bool result = cv::imwrite(filepath, bgr);
        // metadata = pyexiv2.ImageMetadata(filepath)
        // metadata.write()
        sprintf(filepath, "/sdcard/Pictures/Lights/rgb_%d_det.txt", frame_num);
        std::ofstream fw(filepath, std::ofstream::out);
        if (fw.is_open()) {
            fw << "frame: \t" << frame_num << "\n";
            fw << "detect_count: \t" << detect_count << "\n";
            fw << "max_prob: \t" << max_prob << "\n";
            fw << "min_prob: \t" << min_prob << "\n";
            fw << "avg_prob: \t" << (sum_prob / (float)detect_count) << "\n";
            for (int i = 0; i < detect_count; i++) {
                // Dump extra info about each detection
                fw << (i + 1) << ": " << detect_info[i] << "\n";
            }

        }
        fw.close();

//        __android_log_print(ANDROID_LOG_INFO, "yolox", "imwrite frame_num=%d result=%d", frame_num, result);

    }

    return 0;
}
