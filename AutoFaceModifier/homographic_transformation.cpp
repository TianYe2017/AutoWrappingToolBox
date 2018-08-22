#include "homographic_transformation.h"



cv::Mat ComputeForwardMatrix(HT_TRANSFORM_PAIRS htPairs) 
{
	cv::Mat M = cv::Mat(cv::Size(8, 8), CV_32F);
	cv::Mat b = cv::Mat(cv::Size(1, 8), CV_32F);
	for (int i = 0; i < 4; i++) 
	{
		float x1 = htPairs.originalPoints[i][0], y1 = htPairs.originalPoints[i][1];
		float x2 = htPairs.targetPoints[i][0], y2 = htPairs.targetPoints[i][1];
		vector<float> coefs1 = {-x1,-y1,-1,0,0,0,x1*x2,y1*x2};
		vector<float> coefs2 = {0,0,0,-x1,-y1,-1,x1*y2,y1*y2};
		for (int j = 0; j < 8; j++) 
		{
			M.at<float>(2 * i, j) = coefs1[j];
			M.at<float>(2 * i + 1, j) = coefs2[j];
		}
		b.at<float>(2 * i, 0) = -x2;
		b.at<float>(2 * i + 1, 0) = -y2;
	}
	M = M.inv(cv::DECOMP_SVD);
	cv::Mat tmp = M*b;
	cv::Mat ans = cv::Mat(cv::Size(3, 3), CV_32F);
	ans.at<float>(0, 0) = tmp.at<float>(0, 0);
	ans.at<float>(0, 1) = tmp.at<float>(1, 0);
	ans.at<float>(0, 2) = tmp.at<float>(2, 0);
	ans.at<float>(1, 0) = tmp.at<float>(3, 0);
	ans.at<float>(1, 1) = tmp.at<float>(4, 0);
	ans.at<float>(1, 2) = tmp.at<float>(5, 0);
	ans.at<float>(2, 0) = tmp.at<float>(6, 0);
	ans.at<float>(2, 1) = tmp.at<float>(7, 0);
	ans.at<float>(2, 2) = 1.0f;
	return ans;
}

void Test_ComputeForwardMatrix(void) 
{
	HT_TRANSFORM_PAIRS htPairs;
	vector<vector<float>> ori = { { 1,1 },{ 3,3 },{ 5,5 },{ 7,7 } };
	vector<vector<float>> target = { { 12,12 },{ 14,16 },{ 16,19 },{ 26,20 } };
	htPairs.originalPoints = ori;
	htPairs.targetPoints = target;

	cout << ori.size() << " " << ori[0].size() << endl;
	cv::Mat ans = ComputeForwardMatrix(htPairs);
	for (int i = 0; i < 3; i++) 
	{
		for (int j = 0; j < 3; j++) 
		{
			cout << ans.at<float>(i, j) << " ";
		}
		cout << "" << endl;
	}
}

vector<float> ComputeOriginalPos(vector<int> newPos, cv::Mat M) 
{
	float x2 = (float)newPos[0];
	float y2 = (float)newPos[1];
	float a1 = M.at<float>(0, 0);
	float a2 = M.at<float>(0, 1);
	float a3 = M.at<float>(0, 2);
	float a4 = M.at<float>(1, 0);
	float a5 = M.at<float>(1, 1);
	float a6 = M.at<float>(1, 2);
	float a7 = M.at<float>(2, 0);
	float a8 = M.at<float>(2, 1);
	float C1 = a1 - x2 * a7; 
	float C2 = a2 - x2 * a8; 
	float D1 = x2 - a3; 
	float C3 = a4 - y2 * a7; 
	float C4 = a5 - y2 * a8; 
	float D2 = y2 - a6; 
	float x1 = (D2 * C2 - D1 * C4) / (C2 * C3 - C1 * C4); 
	float y1 = (C1 * D2 - D1 * C3) / (C1 * C4 - C2 * C3); 
	vector<float> ans;
	ans.push_back(x1);
	ans.push_back(y1);
	return ans;
}

float ComputeTriangleArea(vector<vector<float>> points) 
{
	float a = points[0][0], b = points[0][1];
	float c = points[1][0], d = points[1][1];
	float e = points[2][0], f = points[2][1];
	return abs(a*d + b*e + c*f - e*d - a*f - b*c) / 2.0f;
}


bool IsThisPointInArea(vector<vector<float>> cluster, vector<float> curPoint) 
{
	vector<vector<float>> AP1 = { cluster[0],cluster[1],curPoint };
	vector<vector<float>> AP2 = { cluster[1],cluster[3],curPoint };
	vector<vector<float>> AP3 = { cluster[2],cluster[3],curPoint };
	vector<vector<float>> AP4 = { cluster[0],cluster[2],curPoint };
	vector<vector<float>> Ref1 = { cluster[0],cluster[1],cluster[2] };
	vector<vector<float>> Ref2 = { cluster[1],cluster[2],cluster[3] };
	float S = ComputeTriangleArea(AP1) + ComputeTriangleArea(AP2) + ComputeTriangleArea(AP3) + ComputeTriangleArea(AP4);
	float R = ComputeTriangleArea(Ref1) + ComputeTriangleArea(Ref2);
	//cout << "S" << S << " " << "R" << R << endl;
	if (abs(S - R) <= 0.00001f) 
	{
		//cout << "IN" << endl;
		return true;
	}
	else 
	{
		//cout << "OUT" << endl;
		return false;
	}
}


bool IsRight(vector<float> p1, vector<float> p2, vector<float> curPoint) 
{
	float x1 = p1[0], y1 = p1[1];
	float x2 = p2[0], y2 = p2[1];
	float x = curPoint[0], y = curPoint[1];

	float sum = (y1 - y2)*x + (x2 - x1)*y + x1*y2 - x2*y1;
	if (sum > 0) 
	{
		return false;
	}
	else 
	{
		return true;
	}
}

bool IsThisPointInArea_Enhanced(vector<vector<float>> cluster, vector<float> curPoint) 
{
	vector<float> p0 = { cluster[0][0], cluster[0][1] };
	vector<float> p1 = { cluster[1][0], cluster[1][1] };
	vector<float> p2 = { cluster[2][0], cluster[2][1] };
	vector<float> p3 = { cluster[3][0], cluster[3][1] };
	if (!IsRight(p0, p1, curPoint)) 
	{
		return false;
	}
	if (!IsRight(p1, p3, curPoint))
	{
		return false;
	}
	if (!IsRight(p3, p2, curPoint))
	{
		return false;
	}
	if (!IsRight(p2, p0, curPoint))
	{
		return false;
	}
	return true;
}


cv::Mat Homographic_transformation(cv::Mat background, cv::Mat object, HT_TRANSFORM_PAIRS htPairs) 
{
	
	float rowlimitDown = (float)(object.rows - 1);
	float collimitRight = (float)(object.cols - 1);


	if (background.channels() != object.channels()) 
	{
		cout << "Object Must have the same depth as Background." << endl;
		while (1) 
		{
			cv::waitKey(20);
		}
	}
	cv::Mat ans = background;

	int boundRowUp = (int)min(min(min(htPairs.targetPoints[0][0], htPairs.targetPoints[1][0]), htPairs.targetPoints[2][0]), htPairs.targetPoints[3][0]) - 1;
	int boundRowDown = (int)max(max(max(htPairs.targetPoints[0][0], htPairs.targetPoints[1][0]), htPairs.targetPoints[2][0]), htPairs.targetPoints[3][0]) + 1;
	int boundColLeft = (int)min(min(min(htPairs.targetPoints[0][1], htPairs.targetPoints[1][1]), htPairs.targetPoints[2][1]), htPairs.targetPoints[3][1]) - 1;
	int boundColRight = (int)max(max(max(htPairs.targetPoints[0][1], htPairs.targetPoints[1][1]), htPairs.targetPoints[2][1]), htPairs.targetPoints[3][1]) + 1;

	cv::Mat forwardMatrix = ComputeForwardMatrix(htPairs);
	if (background.channels() == 1) 
	{
		for (int row = boundRowUp; row <= boundRowDown; row++) 
		{
			for (int col = boundColLeft; col <= boundColRight; col++) 
			{
				if (row < 0 || row > object.rows - 1) 
				{
					continue;
				}
				if (col < 0 || col > object.cols - 1) 
				{
					continue;
				}
				vector<float> curPoint = {(float)row,(float)col};
				if (!IsThisPointInArea_Enhanced(htPairs.targetPoints, curPoint)) 
				{
					continue;
				}
				else
				{
					vector<int> curPointInt = { row,col };
					vector<float> pos = ComputeOriginalPos(curPointInt, forwardMatrix);
					if (pos[0] < 0.0f)
					{
						pos[0] = 0.0f;
					}
					if (pos[0] > rowlimitDown)
					{
						pos[0] = rowlimitDown;
					}
					if (pos[1] < 0.0f)
					{
						pos[1] = 0.0f;
					}
					if (pos[1] > collimitRight)
					{
						pos[1] = collimitRight;
					}
					vector<float> value = FindBIValue(pos, object, object.rows, object.cols);
					ans.at<unsigned char>(row, col) = (unsigned char)value[0];
				}
			}
		}
	}
	else if (background.channels() == 3)
	{
		for (int row = boundRowUp; row <= boundRowDown; row++)
		{
			for (int col = boundColLeft; col <= boundColRight; col++)
			{
				if (row < 0 || row > object.rows - 1)
				{
					continue;
				}
				if (col < 0 || col > object.cols - 1)
				{
					continue;
				}
				vector<float> curPoint = { (float)row,(float)col };
				if (!IsThisPointInArea_Enhanced(htPairs.targetPoints, curPoint))
				{
					continue;
				}
				else
				{
					vector<int> curPointInt = { row,col };
					vector<float> pos = ComputeOriginalPos(curPointInt, forwardMatrix);
					//cout << pos[0] << " " << pos[1] << endl;
					if (pos[0] < 0.0f) 
					{
						pos[0] = 0.0f;
					}
					if (pos[0] > rowlimitDown) 
					{
						pos[0] = rowlimitDown;
					}
					if (pos[1] < 0.0f) 
					{
						pos[1] = 0.0f;
					}
					if (pos[1] > collimitRight) 
					{
						pos[1] = collimitRight;
					}
					vector<float> value = FindBIValue(pos, object, object.rows, object.cols);
					ans.at<cv::Vec3b>(row, col)[0] = (unsigned char)value[0];
					ans.at<cv::Vec3b>(row, col)[1] = (unsigned char)value[1];
					ans.at<cv::Vec3b>(row, col)[2] = (unsigned char)value[2];
				}
			}
		}
	}
	else 
	{
		cout << "FATAL ERROR: Do not support current format." << endl;
		while (1)
		{
			cv::waitKey(20);
		}
	}
	return ans;
}

void Test_homographic_transformation(void) 
{
	cv::Mat background = cv::imread("./MonaLisa.jpg");
	//cv::cvtColor(background, background, CV_RGB2GRAY);

	cv::Mat object = cv::imread("./IE.jpg");
	//cv::cvtColor(object, object, CV_RGB2GRAY);

	cv::imshow("background", background);
	cv::waitKey(20);

	cv::imshow("object", object);
	cv::waitKey(20);

	vector<vector<float>> original = { {70,132},{70,490},{378,132},{378,490} };
	vector<vector<float>> target = { {85,250},{85,437},{313,143},{313,451} };

	HT_TRANSFORM_PAIRS htPairs;
	htPairs.originalPoints = original;
	htPairs.targetPoints = target;

	cv::Mat ans = Homographic_transformation(background, object, htPairs);

	cv::imshow("result", ans);
	cv::waitKey(20);

	cv::imwrite("./demo/test_homographic.jpg", ans);
}