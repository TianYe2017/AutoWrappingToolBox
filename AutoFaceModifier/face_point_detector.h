#pragma once
#ifndef FACE_POINT_DETECTOR_H
#define FACE_POINT_DETECTOR_H

#include <dlib/opencv.h>
#include "opencv.hpp"
#include "highgui.hpp"
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <iostream>
#include <vector>
#include "stdlib.h"
#include "stdio.h"

#define FPD_SELECT_MAX 0
#define FPD_SELECT_ALL 1

using namespace std;
using namespace dlib;

struct FACE
{
	std::vector<std::vector<float>> allPoints;
	float area = -1.0f;
};

void UpdateFace(FACE& face);
frontal_face_detector InitFaceDetector(void);
shape_predictor InitPoseDetector(void);
std::vector<FACE> ShapeDetect(cv::Mat img, frontal_face_detector ffd, shape_predictor pm, int flag, bool debug);
void TestFPD(void);


#endif // !FACE_POINT_DETECTOR_H
