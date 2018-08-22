#pragma once
#ifndef LOACL_WRAPPING_H
#define LOCAL_WRAPPING_H


#include <opencv.hpp>
#include <highgui.hpp>
#include <vector>
#include <string>
#include "stdlib.h"
#include "stdio.h"
#include "bilinear_interpolation.h"
#include "cubic_solver.h"
#include "math.h"

cv::Mat Local_Translation(cv::Mat data, std::vector<float> c, std::vector<float> m, float rmax);
cv::Mat Local_Scale(cv::Mat data, std::vector<float> c, float a, float rmax);
cv::Mat Local_Rotation(cv::Mat data, std::vector<float> c, float alpha, float rmax);
void TestLT(void);
void TestLS(void);
void TestLR(void);



#endif