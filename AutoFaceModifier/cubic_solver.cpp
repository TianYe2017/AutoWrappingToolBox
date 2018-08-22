#include "cubic_solver.h"



using namespace std;

vector<float> SolveCubic(vector<float>coef) 
{
	vector<float> ans;
	cv::solveCubic(coef, ans);
	return ans;
}

vector<float> SolveForInverseRange(float a, float range, float y) 
{
	vector<float> coef = { a / (range*range), 2.0f * a / range, a - 1.0f, y };
	vector<float> candiates;
	cv::solveCubic(coef, candiates);
	vector<float> ans;
	for (int i = 0; i < candiates.size(); i++)
	{
		if (candiates[i] > 0.01f && candiates[i] < range)
		{
			ans.push_back(candiates[i]);
		}
	}
	return ans;
}

void TestSCIR(void) 
{
	for (float a = -1.0f; a <= 1.0f; a += 0.1f) 
	{
		for (float y = 0.0f; y <= 20.0f; y += 1.0f) 
		{
			vector<float> v = SolveForInverseRange(a, 20.0f, y);
			cout << "a: " << a << " y: " << y << endl;
			for (int i = 0; i < v.size(); i++) 
			{
				cout << v[i] << " ";
			}
			cout << "" << endl;
		}
	}
}