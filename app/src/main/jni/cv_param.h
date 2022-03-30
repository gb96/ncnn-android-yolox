//
// Created by Greg Bowering on 23/03/2022.
//
// Parameters for Computer Vision traffic light detection application
#ifndef NCNN_ANDROID_YOLOX_CV_PARAM_H
#define NCNN_ANDROID_YOLOX_CV_PARAM_H

// Camera resolution settings
// 1920x1184 = 2.27MP
// 2560x1440 = 3.69MP not supported
// 3840x2160 = 8.29MP 4K TV
// 5060x3164 = 32.02MP
// 10120x6328 = 64.04MP
#define CAMERA_RES_WIDTH 3840
#define CAMERA_RES_HEIGHT 2160

#define ZOOM 2

//#define CROP_FACTOR 1
//#define CROP_W (CAMERA_RES_WIDTH / CROP_FACTOR)
//#define CROP_H (CAMERA_RES_HEIGHT / CROP_FACTOR)
//#define CROP_X0 ((CAMERA_RES_WIDTH - CROP_W) >> 1)
//#define CROP_X1 ((CAMERA_RES_WIDTH + CROP_W) >> 1)
//#define CROP_Y0 ((CAMERA_RES_HEIGHT - CROP_H) >> 1)
//#define CROP_Y1 ((CAMERA_RES_HEIGHT + CROP_H) >> 1)


// Coordinate values of center of camera image
#define CAMERA_CENTER_X (0.5f * CAMERA_RES_WIDTH)
#define CAMERA_CENTER_Y (0.5f * CAMERA_RES_HEIGHT)

// Detections need to have probability above this threshold to make the cut
#define DETECTION_THRESHOLD 0.04f

// Overlapping detections with overlap greater than this IoU ratio result in
// the lower scoring of the overlapping detections to be discarded
#define NMS_THRESHOLD 0.5f

// Only interested in one target object class, this is the class index for that
#define TARGET_OBJECT_CLASS_IDX 9

// Limit number of simultaneous detections for the target class
#define TARGET_OBJECT_MAX_DETECT_COUNT 8

#define TARGET_OBJECT_BORDER_THICKNESS (9*ZOOM)
#endif //NCNN_ANDROID_YOLOX_CV_PARAM_H
