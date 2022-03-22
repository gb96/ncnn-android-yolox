//
// Created by Greg Bowering on 23/03/2022.
//
// Parameters for Computer Vision traffic light detection application
#ifndef NCNN_ANDROID_YOLOX_CV_PARAM_H
#define NCNN_ANDROID_YOLOX_CV_PARAM_H

// Camera resolution settings
#define CAMERA_RES_WIDTH 2560
#define CAMERA_RES_HEIGHT 1440

// Coordinate values of center of camera image
#define CAMERA_CENTER_X (0.5f * CAMERA_RES_WIDTH)
#define CAMERA_CENTER_Y (0.5f * CAMERA_RES_HEIGHT)

// Only interested in one target object class, this is the class index for that
#define TARGET_OBJECT_CLASS_IDX 9

// Limit number of simultaneous detections for the target class
#define TARGET_OBJECT_MAX_DETECT_COUNT 3

#endif //NCNN_ANDROID_YOLOX_CV_PARAM_H
