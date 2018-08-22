#pragma once
#ifndef CUBIC_SOLVER
#define CUBIC_SOLVER

#include <opencv.hpp>
#include <vector>
#include <iostream>
#include <string>
#include "stdlib.h"
#include "stdio.h"


using namespace std;

std::vector<float> SolveCubic(std::vector<float>coef);
std::vector<float> SolveForInverseRange(float a, float range, float y);
void TestSCIR(void);


#endif