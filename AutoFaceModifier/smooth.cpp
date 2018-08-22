#include "smooth.h"



cv::Mat EPF(cv::Mat data, int r = 5, float threshold = 50.0f) 
{
	cv::Mat ans;
	ans.create(data.rows,data.cols,data.type());
	int rows = data.rows, cols = data.cols;
	cv::Mat edata;
	cv::copyMakeBorder(data,edata,r,r,r,r,IPL_BORDER_REPLICATE);
	float w,sum,sumWeight,cur;
	for (int row = r; row < r + rows; row++) 
	{
		for (int col = r; col < r + cols; col++) 
		{
			sumWeight = 0.0f;
			sum = 0.0f;
			cur = edata.at<float>(row,col);
			for (int i = row - r; i <= row + r; i++) 
			{
				for (int j = col - r; j <= col + r; j++) 
				{
					w = max(0.0f,1.0f - abs(edata.at<float>(i, j) - cur) / threshold);
					sumWeight += w;
					sum += w * edata.at<float>(i, j);
				}
			}
			ans.at<float>(row - r, col - r) = sum / sumWeight;
		}
	}
	return ans;
}


void TestEPF(void) 
{
	cv::Mat ori = cv::imread("./bloody_awful.jpg");
	cv::imshow("ORIGINAL", ori);
	cv::waitKey(20);

	vector<cv::Mat> cluster;
	vector<cv::Mat> ans;
	cv::Mat empty;

	int rows = ori.rows, cols = ori.cols;

	cv::split(ori,cluster);
	for (int i = 0; i < ori.channels(); i++) 
	{
		cluster[i].convertTo(cluster[i], CV_32F);
		ans.push_back(empty);
	}
	for (int i = 0; i < ori.channels(); i++)
	{
		ans[i] = EPF(cluster[i], 5, 50.0f);
	}
	for (int row = 0; row < rows; row++)
	{
		for (int col = 0; col < cols; col++)
		{
			//cout << ori[0].at<float>(row, col) << " " << ori[1].at<float>(row, col) << " " << ori[2].at<float>(row, col) << " " << R3[0].at<float>(row, col) << " " << R3[1].at<float>(row, col) << " " << R3[2].at<float>(row, col) << endl;
			cout << ans[0].at<float>(row, col) << " " << ans[1].at<float>(row, col) << " " << ans[2].at<float>(row, col) << endl;
		}
	}
	for (int i = 0; i < ori.channels(); i++)
	{
		ans[i].convertTo(ans[i], CV_8U);
	}
	cv::Mat result;
	cv::merge(ans, result);
	
	cv::imshow("AFTER", result);
	cv::waitKey(20);
}


cv::Mat SmoothSkin(cv::Mat data, int size1 = 5, int size2 = 3, float threshold = 50.0f, float stddev2 = 0.0f,float opacity = 0.5f, float luminance = 128.0f) 
{
	int rows = data.rows;
	int cols = data.cols;

	// split original image if apply
	vector<cv::Mat> ori;
	if (data.channels() == 3) 
	{
		cv::split(data, ori);
	}
	else if (data.channels() == 1) 
	{
		ori.push_back(data);
	}
	else 
	{
		cout << "FATAL ERROR: Do not support current format." << endl;
		while (1)
		{
			cv::waitKey(20);
		}
	}
	//processing begin here
	
	vector<cv::Mat> R1;
	vector<cv::Mat> R2;
	vector<cv::Mat> R3;
	vector<cv::Mat> processed;
	//allocate space
	cout << "allocate space..." << endl;
	for (int i = 0; i < ori.size(); i++) 
	{
		ori[i].convertTo(ori[i], CV_32F);
		R1.push_back(cv::Mat(cv::Size(cols, rows), CV_32F));
		R2.push_back(cv::Mat(cv::Size(cols, rows), CV_32F));
		R3.push_back(cv::Mat(cv::Size(cols, rows), CV_32F));
		processed.push_back(cv::Mat(cv::Size(cols, rows), CV_32F));
	}
	//Get R1
	cout << "R1..." << endl;
	for (int i = 0; i < ori.size(); i++)
	{
		R1[i] = EPF(ori[i], size1, threshold);
	}
	//Get R2
	cout << "R2..." << endl;
	for (int i = 0; i < ori.size(); i++)
	{
		for (int row = 0; row < rows; row++) 
		{
			for (int col = 0; col < cols; col++) 
			{
				R2[i].at<float>(row, col) = R1[i].at<float>(row, col) - ori[i].at<float>(row, col) + luminance;
			}
		}
	}
	//Get R3
	cout << "R3..." << endl;
	for (int i = 0; i < ori.size(); i++)
	{
		cv::GaussianBlur(R2[i], R3[i], cv::Size(size2,size2), stddev2, stddev2);
	}
	//Get result
	cout << "Gathering..." << endl;
	for (int i = 0; i < ori.size(); i++)
	{
		for (int row = 0; row < rows; row++)
		{
			for (int col = 0; col < cols; col++)
			{
				processed[i].at<float>(row, col) = (1.0f - opacity) * ori[i].at<float>(row, col) + opacity * (ori[i].at<float>(row,col) + 2*R3[i].at<float>(row,col)-256.0f);
			}
		}
	}
	cv::Mat ans;
	if (data.channels() == 3)
	{
		processed[0].convertTo(processed[0], CV_8U);
		processed[1].convertTo(processed[1], CV_8U);
		processed[2].convertTo(processed[2], CV_8U);
		cv::merge(processed, ans);
	}
	else 
	{
		processed[0].convertTo(processed[0], CV_8U);
		ans = processed[0];
	}
	return ans;
}


void TestSmooth(void) 
{
	cv::Mat img = cv::imread("./bloody_awful.jpg");
	cv::imshow("original", img);
	cv::waitKey(20);
	cv::Mat ans = SmoothSkin(img, 13, 3, 70.0f, 0, 0.5f, 128.0f);
	cv::imshow("processed", ans);
	cv::waitKey(20);
	cv::imwrite("./demo_smooth_out.jpg", ans);
	cout << "done" << endl;
}