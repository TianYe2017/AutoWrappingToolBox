#pragma once
#ifndef HOMOGRAPHIC_TRANSFORMATION_H
#define HOMOGRAPHIC_TRANSFORMATION_H

#include <vector>
#include <iostream>
#include <stdlib.h>
#include <opencv.hpp>
#include "bilinear_interpolation.h"

using namespace std;

struct HT_TRANSFORM_PAIRS 
{
	std::vector<std::vector<float>> originalPoints;
	std::vector<std::vector<float>> targetPoints;
};


cv::Mat ComputeForwardMatrix(HT_TRANSFORM_PAIRS htPairs);
void Test_ComputeForwardMatrix(void);
std::vector<float> ComputeOriginalPos(std::vector<int> newPos, cv::Mat M);
bool IsThisPointInArea(std::vector<std::vector<float>> cluster, std::vector<float> curPoint);
bool IsRight(std::vector<float> p1, std::vector<float> p2, std::vector<float> curPoint);
bool IsThisPointInArea_Enhanced(std::vector<std::vector<float>> cluster, std::vector<float> curPoint);
float ComputeTriangleArea(std::vector<std::vector<float>> points);
bool IsThisPointInArea(std::vector<std::vector<float>> cluster, std::vector<float> curPoint);
cv::Mat Homographic_transformation(cv::Mat background, cv::Mat object, HT_TRANSFORM_PAIRS htPairs);
void Test_homographic_transformation(void);










#endif
