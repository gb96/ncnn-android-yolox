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

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "cpu.h"
#include "cv_param.h"



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
    picked.clear();

    const int n = detectedobjs.size();

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
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
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
    const int num_grid = feat_blob.h;
    // fprintf(stderr, "output height: %d, width: %d, channels: %d, dims:%d\n", feat_blob.h, feat_blob.w, feat_blob.c, feat_blob.dims);

    const int num_class = feat_blob.w - 5;

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
                central_bias_x = 1.0f - abs(x_center - CAMERA_CENTER_Y) / CAMERA_RES_HEIGHT;
                size_bias = 0.05f * w;
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

    return 0;
}


int Yolox::detect(const cv::Mat& rgb, std::vector<Object>& objects, float prob_threshold, float nms_threshold)
{
//    __android_log_print(ANDROID_LOG_INFO, "Yolox", "detect(rows=%d, cols=%d)", rgb.rows, rgb.cols);
//    __android_log_print(ANDROID_LOG_INFO, "Yolox", "CROP_FACTOR=%d CROP_W=%d CROP_H=%d", CROP_FACTOR, CROP_W, CROP_H);

//    cv::Mat cropped_rgb = rgb(cv::Range(CROP_Y0,CROP_Y1), cv::Range(CROP_X0,CROP_X1));

    int img_w = rgb.cols;
    int img_h = rgb.rows;
//    __android_log_print(ANDROID_LOG_INFO, "Yolox", "CROP_COLS=%d CROP_ROWS=%d", img_w, img_h);

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

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB, img_w, img_h, w, h);
    // ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB2BGR, img_w, img_h, w, h);

    // pad to target_size rectangle
    // yolov5/utils/datasets.py letterbox
    int wpad = target_size-w;//(w + 31) / 32 * 32 - w;
    int hpad = target_size-h;//(h + 31) / 32 * 32 - h;
//    __android_log_print(ANDROID_LOG_INFO, "Yolox", "target_size=%d wpad=%d hpad=%d", target_size, wpad, hpad);
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, 0, hpad, 0, wpad, ncnn::BORDER_CONSTANT, 114.f); // 114.f

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
    for (size_t i = 0; i < size; i++)
    {
        objects[i] = proposals[picked[i]];
        if (objects[i].label != TARGET_OBJECT_CLASS_IDX) continue;

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
        objects[i].rect.width = ZOOM * (x1 - x0);
        objects[i].rect.height = ZOOM * (y1 - y0);
    }

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
    static const unsigned char colors[19][3] = {
        { 54,  67, 244},
        { 99,  30, 233},
        {176,  39, 156},
        {183,  58, 103},
        {181,  81,  63},
        {243, 150,  33},
        {244, 169,   3},
        {212, 188,   0},
        {136, 150,   0},
        { 80, 175,  76},
        { 74, 195, 139},
        { 57, 220, 205},
        { 59, 235, 255},
        {  7, 193, 255},
        {  0, 152, 255},
        { 34,  87, 255},
        { 72,  85, 121},
        {158, 158, 158},
        {139, 125,  96}
    };

//    static const cv::Scalar cc_black = cv::Scalar(0, 0, 0);
//    static const cv::Scalar cc_white = cv::Scalar(255, 255, 255);
    static const cv::Scalar ccRed = cv::Scalar(255, 0, 0);
    static const cv::Scalar ccGreen = cv::Scalar(0, 255, 0);
    static const cv::Size zoom_px_size = cv::Size(ZOOM, ZOOM);

    int color_index = 0;
    float max_prob = 0.0;
    const cv::Mat rgbClone = rgb.clone();

    for (const auto & obj : objects)
    {
        // display at most TARGET_OBJECT_MAX_DETECT_COUNT detected traffic lights,
        // the most probable detections
        if (color_index >= TARGET_OBJECT_MAX_DETECT_COUNT) break;

        // only count and display traffic light detections
        if (obj.label != TARGET_OBJECT_CLASS_IDX) continue;

        // track maximum probability, should always be the first object due to sort
        // max_prob = fmax(max_prob, obj.prob);
        if (color_index == 0) {
            max_prob = obj.prob;
        }
//        const unsigned char* color = colors[color_index % 19];
        color_index++;

//        cv::Scalar cc(color[0], color[1], color[2]);

        // cv::rectangle(rgb, obj.rect, cc, 4*ZOOM); // 1

        // char text[256];
        // sprintf(text, "%s %.0f%%", class_names[obj.label], obj.prob * 100);

//        char text[5];
//        sprintf(text, "%.0f%%", obj.prob * 100);

//        int baseLine = 0;
        int fontScale = 2* ZOOM;
        int thickness = 2 * ZOOM;


//        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y; // - label_size.height - baseLine;
//        if (y < 0)
//            y = 0;
//        if (x + label_size.width > rgb.cols)
//            x = rgb.cols - label_size.width;

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

//        __android_log_print(ANDROID_LOG_ERROR, "ncnn", "count=%d, max prob=%.0f%%", color_index, max_prob * 100);
//        __android_log_print(ANDROID_LOG_ERROR, "ncnn", "x=%d, y=%d, x2=%d, y2=%d, x2z=%d, y2z=%d, cols=%d, rows=%d", x, y, x2, y2, x2zoom, y2zoom, rgb.cols, rgb.rows);
//        __android_log_print(ANDROID_LOG_ERROR, "ncnn", "orig w=%d, h=%d", wOrig, hOrig);

        int redSum = 0;
        int greenSum = 0;
        int blueSum = 0;
        for (int row = y, rz = y; row < y2; row++, rz += 2) {
            uchar *p = rgbClone.data + (row * rgb.cols + x) * 3;
            for (int col = x, cz = x; col < x2; col++, cz += 2) {
                const uchar r = p[0];
                const uchar g = p[1];
                const uchar b = p[2];
                cv::Scalar pixelColor(r, g, b);
                redSum += r;
                greenSum += g;
                blueSum += b;

                cv::rectangle(rgb, cv::Rect(cv::Point(cz, rz), zoom_px_size), pixelColor, -1);
                p += 3;
            }
        }
        if (redSum >= greenSum) {
            cv::rectangle(rgb, obj.rect, ccRed, 5*ZOOM);
        } else {
            cv::rectangle(rgb, obj.rect, ccGreen, 5*ZOOM);
        }
//        cv::rectangle(rgb, obj.rect, cc, 5*ZOOM);

//        cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)), cc, -1);

//        cv::Scalar textcc = (color[0] + color[1] + color[2] >= 381) ? cc_black : cc_white;

//        cv::putText(rgb, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, fontScale, textcc, thickness);
    }
//    if (color_index > 0) {
//        __android_log_print(ANDROID_LOG_INFO, "ncnn", "count=%d, max prob=%.0f%%", color_index, max_prob * 100);
//    }

    return 0;
}
