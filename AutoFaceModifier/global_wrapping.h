#pragma once

#ifndef GLOBAL_WRAPPING_H
#define GLOBAL_WRAPPING_H

#include <opencv.hpp>
#include <highgui.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <stdlib.h>
#include <stdio.h>

#include "bilinear_interpolation.h"
#include "homographic_transformation.h"


#define WP_AFFINE 0
#define WP_SIMILARITY 1
#define WP_RIGID 2




using namespace std;

struct WP_INPUT 
{
	cv::Mat img;
	std::vector<std::vector<float>> originalPoints;
	std::vector<std::vector<float>> targetPoints;
};





class WRAPPING 
{
public:
	WRAPPING();
	WRAPPING(WP_INPUT data, int ModeIn, float AlphaIn);

	void Reset(void);
	void LoadNewData(WP_INPUT dataIn, int ModeIn, float AlphaIn);

	cv::Mat MLS(int Mode, float step);


private:
	cv::Mat data;
	cv::Mat output;
	float Alpha;

	int numChannels = -1;
	
	//mode control
	int Mode = 0; //default: AFFINE
	
	//store original & target points
	std::vector<std::vector<float>> originalPoints;
	std::vector<std::vector<float>> targetPoints;

	//weight 
	cv::Mat weight; //1 * #of ref pairs

	//center of p&q
	cv::Mat p_center = cv::Mat(cv::Size(2, 1), CV_32F); // 1 * 2
	cv::Mat q_center = cv::Mat(cv::Size(2, 1), CV_32F); // 1 * 2

	//T basis
	cv::Mat t = cv::Mat(cv::Size(2, 1), CV_32F);

	//p hat & q hat
	cv::Mat p_hat = cv::Mat(cv::Size(2, 1), CV_32F);
	cv::Mat q_hat = cv::Mat(cv::Size(2, 1), CV_32F);

	//Transfer Matrix pp & qq  M = pp^-1 * qq
	cv::Mat pp = cv::Mat(cv::Size(2, 2), CV_32F);
	cv::Mat pq = cv::Mat(cv::Size(2, 2), CV_32F);
	cv::Mat ipp = cv::Mat(cv::Size(2, 2), CV_32F);
	cv::Mat ipq = cv::Mat(cv::Size(2, 2), CV_32F);
	cv::Mat M = cv::Mat(cv::Size(2, 2), CV_32F);
	cv::Mat iM = cv::Mat(cv::Size(2, 2), CV_32F);

	//size of new image
	int rowMin = 0, rowMax = 0, colMin = 0, colMax = 0;

	//step
	int stepW = 5;

	//us for similarity transform
	float us;

	//ur for rigid transform
	float ur;

	bool ComputeWeight(std::vector<float> pos);
	void ComputeUs(void);
	void ComputeUr(void);
	void ComputePCenter(void); 
	void ComputeQCenter(void);
	void UpdateMAffine(void); 
	void UpdateMSimilarity(void);
	void UpdateMRigid(void);



	std::vector<std::vector<std::vector<float>>> Grid(cv::Mat img, int step);
	std::vector<std::vector<std::vector<float>>> ComputeForwardBlocks(std::vector<std::vector<std::vector<float>>> blocks, int Mode);



	std::vector<float> ComputeMappingAFFINE(std::vector<float> pos);
	std::vector<float> ComputeMappingSIMILARITY(std::vector<float> pos);
	std::vector<float> ComputeMappingRIGID(std::vector<float> pos);

};

void TestMLS(int Mode);

#endif