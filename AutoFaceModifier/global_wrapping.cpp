#include "global_wrapping.h"


WRAPPING::WRAPPING() 
{

}

WRAPPING::WRAPPING(WP_INPUT dataIn, int ModeIn, float AlphaIn = 1.0f)
{
	originalPoints = dataIn.originalPoints;
	targetPoints = dataIn.targetPoints;
	data = dataIn.img;
	Mode = ModeIn;
	numChannels = data.channels();
	Alpha = AlphaIn;
	weight = cv::Mat(cv::Size(originalPoints.size(), 1), CV_32F);
}

void WRAPPING::Reset(void)
{
	numChannels = -1;
	Mode = 0;
	Alpha = 1.0f;

}

void WRAPPING::LoadNewData(WP_INPUT dataIn, int ModeIn, float AlphaIn = 1.0f)
{
	originalPoints = dataIn.originalPoints;
	targetPoints = dataIn.targetPoints;
	data = dataIn.img;
	numChannels = data.channels();
	Mode = Mode;
	Alpha = AlphaIn;
	weight = cv::Mat(cv::Size(originalPoints.size(), 1), CV_32F);
}

bool WRAPPING::ComputeWeight(vector<float> pos)
{
	float x, y;
	for (int i = 0; i < originalPoints.size(); i++)
	{
		x = originalPoints[i][0] - pos[0];
		y = originalPoints[i][1] - pos[1];
		if (x == 0.0f && y == 0.0f) 
		{
			weight.at<float>(0, i) = 9999999.0f;
		}
		else 
		{
			float s = sqrt(x * x + y * y);
			s = powf(s, 2.0f*Alpha);
			weight.at<float>(0, i) = 1.0f / (x*x + y*y);
		}
	}
	return true;
}

void WRAPPING::ComputeUs(void) 
{
	us = 0.0f;

	for (int i = 0; i < originalPoints.size(); i++) 
	{
		float p1 = originalPoints[i][0] - p_center.at<float>(0, 0);
		float p2 = originalPoints[i][1] - p_center.at<float>(0, 1);
		us += weight.at<float>(0, i) * (p1*p1 + p2*p2);
	}

	return;
}

void WRAPPING::ComputeUr(void) 
{
	cv::Mat q = cv::Mat(1, 2, CV_32F);
	cv::Mat p = cv::Mat(1, 2, CV_32F);

	float tmp1 = 0.0f,tmp2 = 0.0f;
	for (int i = 0; i < originalPoints.size(); i++) 
	{
		p.at<float>(0, 0) = originalPoints[i][0] - p_center.at<float>(0, 0);
		p.at<float>(0, 1) = originalPoints[i][1] - p_center.at<float>(0, 1);
		q.at<float>(0, 0) = targetPoints[i][0] - q_center.at<float>(0, 0);
		q.at<float>(0, 1) = targetPoints[i][1] - q_center.at<float>(0, 1);
		tmp1 += (q.at<float>(0, 0)*p.at<float>(0, 0) + q.at<float>(0, 1)*p.at<float>(0, 1))*weight.at<float>(0, i);
		tmp2 += (q.at<float>(0, 0)*-p.at<float>(0, 1) + q.at<float>(0, 1)*p.at<float>(0, 0))*weight.at<float>(0, i);
	}
	ur = sqrtf(tmp1 * tmp1 + tmp2 * tmp2);
	return;
}

void WRAPPING::ComputePCenter(void) 
{
	p_center.at<float>(0, 0) = 0.0f;
	p_center.at<float>(0, 1) = 0.0f;
	float sumW = 0.0f;
	for (int i = 0; i < originalPoints.size(); i++) 
	{
		sumW += weight.at<float>(0, i);
	}
	for (int i = 0; i < originalPoints.size(); i++)
	{
		p_center.at<float>(0, 0) += weight.at<float>(0, i) * originalPoints[i][0];
		p_center.at<float>(0, 1) += weight.at<float>(0, i) * originalPoints[i][1];
	}
	p_center.at<float>(0, 0) /= sumW;
	p_center.at<float>(0, 1) /= sumW;
}

void WRAPPING::ComputeQCenter(void)
{
	q_center.at<float>(0, 0) = 0.0f;
	q_center.at<float>(0, 1) = 0.0f;
	float sumW = 0.0f;
	for (int i = 0; i < targetPoints.size(); i++)
	{
		sumW += weight.at<float>(0, i);
	}
	for (int i = 0; i < targetPoints.size(); i++)
	{
		q_center.at<float>(0, 0) += weight.at<float>(0, i) * targetPoints[i][0];
		q_center.at<float>(0, 1) += weight.at<float>(0, i) * targetPoints[i][1];
	}
	q_center.at<float>(0, 0) /= sumW;
	q_center.at<float>(0, 1) /= sumW;
}

void WRAPPING::UpdateMAffine(void)
{
	cv::Mat tmp1 = cv::Mat(cv::Size(1, 2), CV_32F);
	cv::Mat tmp2 = cv::Mat(cv::Size(2, 1), CV_32F);
	cv::Mat tmp = cv::Mat(cv::Size(2, 2), CV_32F);

	pp = cv::Mat::zeros(2, 2, CV_32F);
	for (int i = 0; i < originalPoints.size(); i++)
	{
		tmp1.at<float>(0, 0) = originalPoints[i][0] - p_center.at<float>(0, 0);
		tmp1.at<float>(1, 0) = originalPoints[i][1] - p_center.at<float>(0, 1);
		tmp2.at<float>(0, 0) = tmp1.at<float>(0, 0);
		tmp2.at<float>(0, 1) = tmp1.at<float>(1, 0);
		tmp = tmp1*tmp2;
		tmp.at<float>(0, 0) *= weight.at<float>(0, i);
		tmp.at<float>(0, 1) *= weight.at<float>(0, i);
		tmp.at<float>(1, 0) *= weight.at<float>(0, i);
		tmp.at<float>(1, 1) *= weight.at<float>(0, i);
		pp = pp + tmp;
	}
	float det = pp.at<float>(0, 0) * pp.at<float>(1, 1) - pp.at<float>(0, 1) * pp.at<float>(1, 0);
	if (det < 0.000001f)
	{
		M.at<float>(0, 0) = -999.9f;
		M.at<float>(0, 1) = -999.9f;
		M.at<float>(1, 0) = -999.9f;
		M.at<float>(1, 1) = -999.9f;
		cout << "NO INVERSE WARNING" << endl;
		return; //eraly stop to indicate M do not has an inverse
	}
	else
	{
		ipp = pp.inv(cv::DECOMP_SVD);
	}
	pq = cv::Mat::zeros(2, 2, CV_32F);
	for (int i = 0; i < originalPoints.size(); i++)
	{
		tmp1.at<float>(0, 0) = originalPoints[i][0] - p_center.at<float>(0, 0);
		tmp1.at<float>(1, 0) = originalPoints[i][1] - p_center.at<float>(0, 1);
		tmp2.at<float>(0, 0) = targetPoints[i][0] - q_center.at<float>(0, 0);
		tmp2.at<float>(0, 1) = targetPoints[i][1] - q_center.at<float>(0, 1);
		tmp = tmp1*tmp2;
		tmp.at<float>(0, 0) *= weight.at<float>(0, i);
		tmp.at<float>(0, 1) *= weight.at<float>(0, i);
		tmp.at<float>(1, 0) *= weight.at<float>(0, i);
		tmp.at<float>(1, 1) *= weight.at<float>(0, i);
		pq = pq + tmp;
	}
	M = ipp * pq;
}

void WRAPPING::UpdateMSimilarity(void)
{
	cv::Mat pp = cv::Mat(2, 2, CV_32F);
	cv::Mat qq = cv::Mat(2, 2, CV_32F);
	cv::Mat p = cv::Mat(1, 2, CV_32F);
	cv::Mat q = cv::Mat(1, 2, CV_32F);
	cv::Mat sum = cv::Mat::zeros(2, 2, CV_32F);
	for (int i = 0; i < originalPoints.size(); i++) 
	{
		p.at<float>(0, 0) = originalPoints[i][0] - p_center.at<float>(0, 0);
		p.at<float>(0, 1) = originalPoints[i][1] - p_center.at<float>(0, 1);
		q.at<float>(0, 0) = targetPoints[i][0] - q_center.at<float>(0, 0);
		q.at<float>(0, 1) = targetPoints[i][1] - q_center.at<float>(0, 1);

		pp.at<float>(0, 0) = p.at<float>(0, 0);
		pp.at<float>(0, 1) = p.at<float>(0, 1);
		pp.at<float>(1, 0) = p.at<float>(0, 1);
		pp.at<float>(1, 1) = -p.at<float>(0, 0);

		qq.at<float>(0, 0) = q.at<float>(0, 0);
		qq.at<float>(0, 1) = q.at<float>(0, 1);
		qq.at<float>(1, 0) = q.at<float>(0, 1);
		qq.at<float>(1, 1) = -q.at<float>(0, 0);
		pp = pp*qq;

		pp.at<float>(0, 0) *= weight.at<float>(0, i);
		pp.at<float>(0, 1) *= weight.at<float>(0, i);
		pp.at<float>(1, 0) *= weight.at<float>(0, i);
		pp.at<float>(1, 1) *= weight.at<float>(0, i);
		sum = sum + pp;
	}
	M.at<float>(0, 0) = sum.at<float>(0, 0) / us;
	M.at<float>(0, 1) = sum.at<float>(0, 1) / us;
	M.at<float>(1, 0) = sum.at<float>(1, 0) / us;
	M.at<float>(1, 1) = sum.at<float>(1, 1) / us;
	return;
}

void WRAPPING::UpdateMRigid(void)
{
	cv::Mat pp = cv::Mat(2, 2, CV_32F);
	cv::Mat qq = cv::Mat(2, 2, CV_32F);
	cv::Mat p = cv::Mat(1, 2, CV_32F);
	cv::Mat q = cv::Mat(1, 2, CV_32F);
	cv::Mat sum = cv::Mat::zeros(2, 2, CV_32F);
	for (int i = 0; i < originalPoints.size(); i++)
	{
		p.at<float>(0, 0) = originalPoints[i][0] - p_center.at<float>(0, 0);
		p.at<float>(0, 1) = originalPoints[i][1] - p_center.at<float>(0, 1);
		q.at<float>(0, 0) = targetPoints[i][0] - q_center.at<float>(0, 0);
		q.at<float>(0, 1) = targetPoints[i][1] - q_center.at<float>(0, 1);

		pp.at<float>(0, 0) = p.at<float>(0, 0);
		pp.at<float>(0, 1) = p.at<float>(0, 1);
		pp.at<float>(1, 0) = p.at<float>(0, 1);
		pp.at<float>(1, 1) = -p.at<float>(0, 0);

		qq.at<float>(0, 0) = q.at<float>(0, 0);
		qq.at<float>(0, 1) = q.at<float>(0, 1);
		qq.at<float>(1, 0) = q.at<float>(0, 1);
		qq.at<float>(1, 1) = -q.at<float>(0, 0);
		pp = pp*qq;

		pp.at<float>(0, 0) *= weight.at<float>(0, i);
		pp.at<float>(0, 1) *= weight.at<float>(0, i);
		pp.at<float>(1, 0) *= weight.at<float>(0, i);
		pp.at<float>(1, 1) *= weight.at<float>(0, i);
		sum = sum + pp;
	}
	M.at<float>(0, 0) = sum.at<float>(0, 0) / ur;
	M.at<float>(0, 1) = sum.at<float>(0, 1) / ur;
	M.at<float>(1, 0) = sum.at<float>(1, 0) / ur;
	M.at<float>(1, 1) = sum.at<float>(1, 1) / ur;
	return;
}

vector<float> WRAPPING::ComputeMappingSIMILARITY(vector<float> pos)
{
	vector<float> ans;
	ComputeWeight(pos);
	ComputePCenter();
	ComputeQCenter();
	ComputeUs();
	UpdateMSimilarity();
	cv::Mat x = cv::Mat(cv::Size(2, 1), CV_32F);
	x.at<float>(0, 0) = pos[0];
	x.at<float>(0, 1) = pos[1];
	cv::Mat y = (x - p_center) * M + q_center;
	ans.push_back(y.at<float>(0, 0));
	ans.push_back(y.at<float>(0, 1));
	return ans;
}

vector<float> WRAPPING::ComputeMappingRIGID(vector<float> pos)
{
	vector<float> ans;
	ComputeWeight(pos);
	ComputePCenter();
	ComputeQCenter();
	ComputeUr();
	UpdateMRigid();
	cv::Mat x = cv::Mat(cv::Size(2, 1), CV_32F);
	x.at<float>(0, 0) = pos[0];
	x.at<float>(0, 1) = pos[1];
	cv::Mat y = (x - p_center) * M + q_center;
	ans.push_back(y.at<float>(0, 0));
	ans.push_back(y.at<float>(0, 1));
	return ans;
}

vector<float> WRAPPING::ComputeMappingAFFINE(vector<float> pos) 
{
	vector<float> ans;
	//update weight
	ComputeWeight(pos);
	//update P & Q center
	ComputePCenter();
	ComputeQCenter();
	//update M
	UpdateMAffine();
	//compute forward coordinate
	cv::Mat x = cv::Mat(cv::Size(2, 1), CV_32F);
	x.at<float>(0, 0) = pos[0];
	x.at<float>(0, 1) = pos[1];
	//if current point do not have valid M, use alternative transform
	if (M.at<float>(0, 0) == -999.9f && M.at<float>(0, 1) == -999.9f && M.at<float>(1, 0) == -999.9f && M.at<float>(1, 1) == -999.9f) 
	{
		ans.push_back(pos[0] + q_center.at<float>(0, 0) - p_center.at<float>(0, 0));
		ans.push_back(pos[1] + q_center.at<float>(0, 1) - p_center.at<float>(0, 1));
		return ans;
	}
	cv::Mat y = (x - p_center) * M + q_center;
	ans.push_back(y.at<float>(0, 0));
	ans.push_back(y.at<float>(0, 1));
	return ans;
}

vector<vector<vector<float>>> WRAPPING::Grid(cv::Mat img, int step)
{
	vector<vector<vector<float>>> ans;
	int stride = step - 1;
	int nRow = (img.rows - 1) / stride - 1;
	int nCol = (img.cols - 1) / stride - 1;

	vector<vector<float>> tmp;
	vector<float> point0;
	vector<float> point1;
	vector<float> point2;
	vector<float> point3;
	for (int row = 0; row <= nRow; row++) 
	{
		for (int col = 0; col <= nCol; col++) 
		{
			point0 = { (float)(stride * row), (float)(stride * col) };
			point1 = { (float)(stride * row), (float)(stride * col + stride) };
			point2 = { (float)(stride * row + stride), (float)(stride * col) };
			point3 = { (float)(stride * row + stride), (float)(stride * col + stride) };
			tmp = { point0,point1,point2,point3 };
			ans.push_back(tmp);
		}
	}
	return ans;
}

vector<vector<vector<float>>> WRAPPING::ComputeForwardBlocks(vector<vector<vector<float>>> blocks, int Mode)
{
	cout << "computing new coordinates..." << endl;
	int N = blocks.size() - 1;

	rowMax = -99999.9f;
	colMax = -99999.9f;
	rowMin = 99999.9f;
	colMin = 99999.9f;

	vector<vector<vector<float>>> ans;

	for (int i = 0; i <= N; i++) 
	{
		//cout << "Already finished: " << (float)(i) / (float)N * 100.0 << "%" << endl;
		vector<vector<float>> tmp;
		for (int k = 0; k < 4; k++) 
		{
			vector<float> newPos;
			if (Mode == 0) 
			{
				newPos = ComputeMappingAFFINE(blocks[i][k]);
			}
			else if (Mode == 1) 
			{
				newPos = ComputeMappingSIMILARITY(blocks[i][k]);
			}
			else if (Mode == 2) 
			{
				newPos = ComputeMappingRIGID(blocks[i][k]);
			}
			else 
			{
				newPos = ComputeMappingAFFINE(blocks[i][k]);
			}
			tmp.push_back(newPos);
			//cout << newPos[0] << " " << newPos[1] << endl;
			if (newPos[0] > rowMax) 
			{
				rowMax = newPos[0];
			}
			if (newPos[0] < rowMin) 
			{
				rowMin = newPos[0];
			}
			if (newPos[1] > colMax) 
			{
				colMax = newPos[1];
			}
			if (newPos[1] < colMin) 
			{
				colMin = newPos[1];
			}
		}
		ans.push_back(tmp);
	}
	for (int i = 0; i <= N; i++)
	{
		for (int k = 0; k < 4; k++)
		{
			ans[i][k][0] -= rowMin;
			ans[i][k][1] -= colMin;
			ans[i][k][0] += 1.0f;
			ans[i][k][1] += 1.0f;
		}
	}
	return ans;
}

cv::Mat WRAPPING:: MLS(int Mode, float step = 5.0f)
{
	vector<vector<vector<float>>> blocks_ori = Grid(data, stepW);
	vector<vector<vector<float>>> blocks_tar = ComputeForwardBlocks(blocks_ori, Mode);
	//create a empty background
	cv::Mat background;
	background.create((int)(rowMax - rowMin + 5.0f), (int)(colMax - colMin + 5.0f), data.type());
	//homographic transformation
	cout << "wrapping..." << endl;
	for (int i = 0; i < blocks_ori.size(); i++) 
	{
		//cout << "Wrapping...Already Finished: " << (float)i / (float)blocks_ori.size() * 100.0 << "%" << endl;
		HT_TRANSFORM_PAIRS htPairs;
		htPairs.originalPoints = blocks_ori[i];
		htPairs.targetPoints = blocks_tar[i];
		background = Homographic_transformation(background, data, htPairs);
	}
	return background;
}


void TestMLS(int Mode)
{
	vector<vector<float>> original = { {244,266},{243,317},{219,278},{165,253},{162,324},{ 9,9 },{ 9,630 },{ 468,9 },{ 470,628 },{9,315},{470,315},{240,9},{240,630} };
	vector<vector<float>> target = { {244,266},{227,334},{217,286},{165,253},{162,324} ,{ 9,9 },{ 9,630 },{ 468,9 },{ 470,628 } ,{ 9,315 },{ 470,315 },{ 240,9 },{ 240,630 } };
	//vector<vector<float>> target = { { 244,266 },{ 243,317 },{ 219,278 },{ 165,253 },{ 162,324 },{ 9,9 },{ 9,630 },{ 468,9 },{ 470,628 } };

	//vector<vector<float>> original = { { 244,266 },{ 243,317 },{ 219,278 },{ 165,253 },{ 162,324 }};
	//vector<vector<float>> target = { { 244,266 },{ 227,334 },{ 217,286 },{ 165,253 },{ 162,324 }};

	cv::Mat img = cv::imread("./MonaLisa.jpg");
	//cv::cvtColor(img, img, CV_RGB2GRAY);
	cv::imshow("original", img);
	cv::waitKey(20);
	
	WP_INPUT DATA;
	DATA.img = img;
	DATA.originalPoints = original;
	DATA.targetPoints = target;

	WRAPPING* wp = new WRAPPING();
	wp->LoadNewData(DATA, Mode, 1.0f);

	cv::Mat result = wp->MLS(Mode, 5.0f);

	string name = "./demo/MLS";
	if (Mode == WP_AFFINE) 
	{
		name += "_affine.jpg";
	}
	if (Mode == WP_SIMILARITY)
	{
		name += "_similarity.jpg";
	}
	if (Mode == WP_RIGID)
	{
		name += "_rigid.jpg";
	}
	cv::imwrite(name, result);
	cv::imshow(name, result);
	cv::waitKey(20);
}



