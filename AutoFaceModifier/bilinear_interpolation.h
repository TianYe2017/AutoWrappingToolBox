#pragma once
#ifndef BILINEAR_INTERPOLATION_H
#define BILINEAR_INTERPOLATION_H

#include <vector>
#include <iostream>
#include <stdlib.h>
#include <opencv.hpp>


using namespace std;


std::vector<float> FindBIValue(std::vector<float> pos, cv::Mat img, int bound_row, int bound_col);
void TestBI_BGR(void);
void TestBI_GRAY(void);


#endif
