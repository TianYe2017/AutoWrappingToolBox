#include "bilinear_interpolation.h"


vector<float> FindBIValue(vector<float> pos, cv::Mat img, int bound_row, int bound_col) 
{
	vector<float> ans;
	float x = pos[0], y = pos[1];
	float X = (float)((int)x), Y = (float)((int)y);
	if (img.channels() == 1)
	{
		//如果直接就在点上
		if (x == X && y == Y) 
		{
			ans.push_back((float)img.at<unsigned char>((int)x, (int)y));
			return ans;
		}
		//如果行（x）满足但是列（y）不满足（x1 = x2）
		if (x == X && y != Y) 
		{
			if ((int)ceilf(y) == bound_col) 
			{
				y -= 1.0f;
			}
			float y1 = floorf(y);
			float y2 = ceilf(y);
			float V1 = (float)img.at<unsigned char>((int)x, (int)y1);
			float V2 = (float)img.at<unsigned char>((int)x, (int)y2);
			float tmp = V1 + (V2 - V1) * (y - y1) / (y2 - y1);
			ans.push_back(tmp);
			return ans;
		}
		//如果行（x）不满足但是列（y）满足（y1 = y2）
		if (x != X && y == Y) 
		{
			if ((int)ceilf(x) == bound_row) 
			{
				x -= 1.0f;
			}
			float x1 = floorf(x);
			float x2 = ceilf(x);
			float V1 = (float)img.at<unsigned char>((int)x1, (int)y);
			float V3 = (float)img.at<unsigned char>((int)x2, (int)y);
			float tmp = V1 + (V3 - V1) * (x - x1) / (x2 - x1);
			ans.push_back(tmp);
			return ans;
		}
		//如果行（x）不满足且列（y）不满足
		if ((int)ceilf(y) == bound_col)
		{
			y -= 1.0f;
		}
		if ((int)ceilf(x) == bound_row)
		{
			x -= 1.0f;
		}
		float x1 = floorf(x);
		float y1 = floorf(y);
		float x2 = ceilf(x);
		float y2 = ceilf(y);
		float V1 = (float)img.at<unsigned char>((int)x1, (int)y1);
		float V2 = (float)img.at<unsigned char>((int)x1, (int)y2);
		float V3 = (float)img.at<unsigned char>((int)x2, (int)y1);
		float V4 = (float)img.at<unsigned char>((int)x2, (int)y2);
		float tmpY1 = V1 + (V3 - V1) * (x - x1) / (x2 - x1);
		float tmpY2 = V2 + (V4 - V2) * (x - x1) / (x2 - x1);
		float tmp = tmpY1 + (tmpY2 - tmpY1) * (y - y1) / (y2 - y1);
		ans.push_back(tmp);
		return ans;
	}
	else if(img.channels() == 3)
	{
		//如果直接就在点上
		if (x == X && y == Y)
		{
			ans.push_back((float)img.at<cv::Vec3b>((int)x, (int)y)[0]);
			ans.push_back((float)img.at<cv::Vec3b>((int)x, (int)y)[1]);
			ans.push_back((float)img.at<cv::Vec3b>((int)x, (int)y)[2]);
			return ans;
		}
		//如果行（x）满足但是列（y）不满足（x1 = x2）
		if (x == X && y != Y)
		{
			if ((int)ceilf(y) == bound_col)
			{
				y -= 1.0f;
			}
			float y1 = floorf(y);
			float y2 = ceilf(y);
			for (int c = 0; c < 3; c++) 
			{
				float V1 = (float)img.at<cv::Vec3b>((int)x, (int)y1)[c];
				float V2 = (float)img.at<cv::Vec3b>((int)x, (int)y2)[c];
				float tmp = V1 + (V2 - V1) * (y - y1) / (y2 - y1);
				ans.push_back(tmp);
			}			
			return ans;
		}
		//如果行（x）不满足但是列（y）满足（y1 = y2）
		if (x != X && y == Y)
		{
			if ((int)ceilf(x) == bound_row)
			{
				x -= 1.0f;
			}
			float x1 = floorf(x);
			float x2 = ceilf(x);
			for (int c = 0; c < 3; c++) 
			{
				float V1 = (float)img.at<cv::Vec3b>((int)x1, (int)y)[c];
				float V3 = (float)img.at<cv::Vec3b>((int)x2, (int)y)[c];
				float tmp = V1 + (V3 - V1) * (x - x1) / (x2 - x1);
				ans.push_back(tmp);
			}	
			return ans;
		}
		//如果行（x）不满足且列（y）不满足
		if ((int)ceilf(y) == bound_col)
		{
			y -= 1.0f;
		}
		if ((int)ceilf(x) == bound_row)
		{
			x -= 1.0f;
		}
		float x1 = floorf(x);
		float y1 = floorf(y);
		float x2 = ceilf(x);
		float y2 = ceilf(y);
		for (int c = 0; c < 3; c++)
		{
			float V1 = (float)img.at<cv::Vec3b>((int)x1, (int)y1)[c];
			float V2 = (float)img.at<cv::Vec3b>((int)x1, (int)y2)[c];
			float V3 = (float)img.at<cv::Vec3b>((int)x2, (int)y1)[c];
			float V4 = (float)img.at<cv::Vec3b>((int)x2, (int)y2)[c];
			float tmpY1 = V1 + (V3 - V1) * (x - x1) / (x2 - x1);
			float tmpY2 = V2 + (V4 - V2) * (x - x1) / (x2 - x1);
			float tmp = tmpY1 + (tmpY2 - tmpY1) * (y - y1) / (y2 - y1);
			ans.push_back(tmp);
		}
		return ans;
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


void TestBI_BGR(void) 
{
	cv::Mat img = cv::imread("./MonaLisa.jpg");

	cv::imshow("original", img);
	cv::waitKey(100);
	
	cv::Mat B = cv::Mat(cv::Size(500, 1000), CV_8U);
	cv::Mat G = cv::Mat(cv::Size(500, 1000), CV_8U);
	cv::Mat R = cv::Mat(cv::Size(500, 1000), CV_8U);

	for (int row = 0; row < 1000; row++)
	{
		for (int col = 0; col < 500; col++)
		{
			vector<float> pos = {(float)row * (480.0f / 1000.0f),(float)col * (640.0f / 500.0f)};
			vector<float> tmp = FindBIValue(pos,img,img.rows,img.cols);
			B.at<unsigned char>(row, col) = (unsigned char)tmp[0];
			G.at<unsigned char>(row, col) = (unsigned char)tmp[1];
			R.at<unsigned char>(row, col) = (unsigned char)tmp[2];
		}
	}
	vector<cv::Mat> cluster;
	cluster.push_back(B);
	cluster.push_back(G);
	cluster.push_back(R);

	cv::Mat ans;
	cv::merge(cluster, ans);

	cv::imshow("new",ans);
	cv::waitKey(100);

	cv::imwrite("./demo/bilinear_interpolation.jpg",ans);
	cv::waitKey(100);
}

void TestBI_GRAY(void)
{
	cv::Mat img = cv::imread("./MonaLisa.jpg");

	cv::cvtColor(img, img, CV_RGB2GRAY);
	cv::imshow("original", img);
	cv::waitKey(100);

	cv::Mat ans = cv::Mat(cv::Size(300, 300), CV_8U);
	
	for (int row = 0; row < 300; row++)
	{
		for (int col = 0; col < 300; col++)
		{
			//cout << row << " " << col << endl;
			vector<float> pos = { (float)row * (480.0f / 300.0f),(float)col * (640.0f / 300.0f) };
			vector<float> tmp = FindBIValue(pos, img, img.rows, img.cols);
			ans.at<unsigned char>(row, col) = (unsigned char)tmp[0];
	
		}
	}
	cv::imshow("new", ans);
	cv::waitKey(100);
}