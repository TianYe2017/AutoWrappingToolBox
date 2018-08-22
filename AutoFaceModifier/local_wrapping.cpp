#include "local_wrapping.h"

cv::Mat Local_Translation(cv::Mat data, vector<float> c, vector<float> m, float rmax = 10.0f) 
{
	cv::Mat result;
	data.copyTo(result);
	int rows = data.rows;
	int cols = data.cols;

	vector<float> vmc = {m[0] - c[0] , m[1] - c[1]};
	vector<float> pos;
	float mc = (m[0] - c[0])*(m[0] - c[0]) + (m[1] - c[1])*(m[1] - c[1]);
	float rmc, ct;

	for (int row = c[0] - rmax; row <= c[0] + rmax; row++) 
	{
		for (int col = c[1] - rmax; col <= c[1] + rmax; col++) 
		{
			//make sure current point is valid
			if (row < 0 || row > rows - 1 || col < 0 || col > cols - 1) 
			{
				continue;
			}
			//check if the point is in range
			float dis = sqrtf(((float)row - c[0]) * ((float)row - c[0]) + ((float)col - c[1]) * ((float)col - c[1]));
			if (dis >= rmax) 
			{
				continue;
			}
			//compute corresponding pos
			rmc = rmax * rmax - dis * dis;
			ct = rmc / (rmc + mc);
			ct = ct * ct;
			pos = {(float)row - ct * vmc[0], (float)col - ct * vmc[1]};
			//cout << pos[0] << " " << pos[1] << endl;
			vector<float> ans = FindBIValue(pos, data, rows, cols);
			//assign value
			if (data.channels() == 1) 
			{
				result.at<unsigned char>(row, col) = (unsigned char)ans[0];
			}
			else if (data.channels() == 3) 
			{
				result.at<cv::Vec3b>(row, col)[0] = (unsigned char)ans[0];
				result.at<cv::Vec3b>(row, col)[1] = (unsigned char)ans[1];
				result.at<cv::Vec3b>(row, col)[2] = (unsigned char)ans[2];
			}
			else 
			{
				cout << "FATAL ERROR: Do not support current format." << endl;
				while (1)
				{
					cv::waitKey(20);
				}
			}
		}
	}
	return result;
}


void TestLT(void) 
{
	cv::Mat img = cv::imread("./MonaLisa.jpg");
	cv::imshow("original", img);
	cv::waitKey(20);

	vector<float> c = { 248,286 };
	vector<float> m = { 300,288 };

	cv::Mat result = Local_Translation(img, c, m, 50.0f);
	cv::imwrite("./demo/testLT.jpg", result);
	cv::imshow("LT",result);
	cv::waitKey(20);

}

cv::Mat Local_Scale(cv::Mat data, vector<float> c, float a = -0.3f, float rmax = 20.0f) 
{
	cv::Mat result;
	data.copyTo(result);

	if (a == 0.0f) 
	{
		return result;
	}

	int rows = data.rows;
	int cols = data.cols;

	vector<float> pos;
	float R,A;


	for (int row = c[0] - rmax; row <= c[0] + rmax; row++)
	{
		for (int col = c[1] - rmax; col <= c[1] + rmax; col++)
		{		
			//make sure current point is valid
			if (row < 0 || row > rows - 1 || col < 0 || col > cols - 1)
			{
				continue;
			}
			//check if the point is in range
			float dis = sqrtf(((float)row - c[0]) * ((float)row - c[0]) + ((float)col - c[1]) * ((float)col - c[1]));
			if (dis >= rmax)
			{
				continue;
			}
			//check if the point is worth computed
			if (dis < 0.99f) 
			{
				continue;
			}
			//Update A
			A = -a / rmax * dis + a;
			//compute R
			vector<float> tmp = SolveForInverseRange(A, rmax, dis);
			if (tmp.size() == 0) 
			{
				continue;
			}
			else if (tmp.size() == 1) 
			{
				R = tmp[0];
			}
			else 
			{
				R = tmp[0];
				cout << "WARNING: MULTI SOLUTIONS FOR CUBIC EQUATION." << endl;
			}
			//compute pos
			pos = { R / dis * ((float)row - c[0]) + c[0], R / dis*((float)col - c[1]) + c[1] };
			//assign value
			vector<float> ans = FindBIValue(pos, data, rows, cols);
			if (data.channels() == 1)
			{
				result.at<unsigned char>(row, col) = (unsigned char)ans[0];
			}
			else if (data.channels() == 3)
			{
				result.at<cv::Vec3b>(row, col)[0] = (unsigned char)ans[0];
				result.at<cv::Vec3b>(row, col)[1] = (unsigned char)ans[1];
				result.at<cv::Vec3b>(row, col)[2] = (unsigned char)ans[2];
			}
			else
			{
				cout << "FATAL ERROR: Do not support current format." << endl;
				while (1)
				{
					cv::waitKey(20);
				}
			}
		}
	}
	return result;
}

void TestLS(void) 
{
	cv::Mat img = cv::imread("./MonaLisa.jpg");
	cv::imshow("original", img);
	cv::waitKey(20);

	vector<float> c = { 164,255 };
	cv::Mat result = Local_Scale(img, c, -0.3f, 30.0f);

	c = { 164,327 };
	result = Local_Scale(result, c, -0.3f, 30.0f);

	cv::imshow("LS", result);
	cv::waitKey(20);

	cv::imwrite("./demo/testLS.jpg", result);
}


cv::Mat Local_Rotation(cv::Mat data, vector<float> c, float alpha, float rmax = 30.0f) 
{
	cv::Mat result;
	data.copyTo(result);
	if (alpha == 0.0f)
	{
		return result;
	}
	int rows = data.rows;
	int cols = data.cols;

	float angle_cur, angle_inv, dangle;
	vector<float> pos;

	for (int row = c[0] - rmax; row <= c[0] + rmax; row++)
	{
		for (int col = c[1] - rmax; col <= c[1] + rmax; col++)
		{
			//make sure current point is valid
			if (row < 0 || row > rows - 1 || col < 0 || col > cols - 1)
			{
				continue;
			}
			//check if the point is in range
			float dis = sqrtf(((float)row - c[0]) * ((float)row - c[0]) + ((float)col - c[1]) * ((float)col - c[1]));
			if (dis >= rmax)
			{
				continue;
			}
			//check if the point is worth computed
			if (dis < 0.99f)
			{
				continue;
			}
			//compute current angle
			angle_cur = atan2f((float)row - c[0],(float)col - c[1]);
			//angle_cur = atan2f((float)col - c[1], (float)row - c[0]);
			dangle = ((1.0f - dis*dis / (rmax*rmax))) * ((1.0f - dis*dis / (rmax*rmax))) * alpha;
			angle_inv = angle_cur - dangle;
			//compute inv pos
			pos = {c[0] + dis * sinf(angle_inv), c[1] + dis * cosf(angle_inv)};
			//assign value
			vector<float> ans = FindBIValue(pos, data, rows, cols);
			if (data.channels() == 1)
			{
				result.at<unsigned char>(row, col) = (unsigned char)ans[0];
			}
			else if (data.channels() == 3)
			{
				result.at<cv::Vec3b>(row, col)[0] = (unsigned char)ans[0];
				result.at<cv::Vec3b>(row, col)[1] = (unsigned char)ans[1];
				result.at<cv::Vec3b>(row, col)[2] = (unsigned char)ans[2];
			}
			else
			{
				cout << "FATAL ERROR: Do not support current format." << endl;
				while (1)
				{
					cv::waitKey(20);
				}
			}
		}
	}
	return result;
}

void TestLR(void) 
{
	cv::Mat img = cv::imread("./MonaLisa.jpg");
	cv::imshow("original", img);
	cv::waitKey(20);

	vector<float> c = { 164,255 };
	cv::Mat result = Local_Rotation(img,c,-0.3f,30.0f);

	c = { 164,327 };
	result = Local_Rotation(result, c, 0.3f, 30.0f);

	cv::imwrite("./demo/testLR.jpg", result);
	cv::imshow("LR", result);
	cv::waitKey(20);

}