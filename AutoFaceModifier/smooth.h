#pragma once
#ifndef SMOOTH_H
#define SMOOTH_H

#include "opencv.hpp"
#include "highgui.hpp"

#include "stdlib.h"
#include "stdio.h"
#include <vector>
#include <string>

using namespace std;

cv::Mat EPF(cv::Mat data, int r, float threshold);
cv::Mat SmoothSkin(cv::Mat data, int size1, int size2, float threshold, float stddev2, float opacity, float luminance);

void TestEPF(void);
void TestSmooth(void);


#endif // !SMOOTH_H
