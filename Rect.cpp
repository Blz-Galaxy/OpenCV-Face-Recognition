/***********************************
   数字图像处理实验-人脸识别

文件：Rect.cpp
简介：实现了三种人脸识别方式，给出了
所有测试图片与预测图片的对比，并通过
判断匹配结果正确与否，统计出总的识别
率。

                          by KC-Mei
                          2014/6/15
************************************/


#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <vector>
#include "opencv2/contrib/contrib.hpp"
#include <windows.h>
#include <fstream>

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	//学习训练样本
	vector<Mat> images;
	vector<int> labels;
	
	cout<<"Loading images for training...";
	for (int i = 0; i < 280; i++)
	{    
		char path[100];  
		sprintf(path,"..//..//人脸识别库2//TrainDatabase//%d.jpg",i+1);
		images.push_back(imread(path,0));
        labels.push_back(i);
	}
	cout<<"Done!"<<endl;
	
	while(1)
	{
		int c;
		cout<<"Select the recognize algorithm(input number)：\n 1）Eigen Face;\n 2）Fisher Face;\n 3）LBPH Face;\n Other）Exit.\n";
		cin>>c;
		fstream _file;
		Ptr<FaceRecognizer> model;

		switch(c)
		{
		case 1:
			model = createEigenFaceRecognizer();
			_file.open("EigenFace.data",ios::in);
			if(_file)
			{
				model->load("EigenFace.data");
			}
			else
			{
				cout<<"Training...";
				model->train(images, labels);
				//model->save("EigenFace.data");
				cout<<"Done!"<<endl;
			}
			break;

		case 2:
			model = createFisherFaceRecognizer();
			_file.open("FisherFace.data",ios::in);
			if(_file)
			{
				model->load("FisherFace.data");
			}
			else
			{
				cout<<"Training...";
				model->train(images, labels);
				//model->save("FisherFace.data");
				cout<<"Done!"<<endl;
			}
			break;

		case 3:
			model = createLBPHFaceRecognizer();
			_file.open("LBPHFace.data",ios::in);
			if(_file)
			{
				model->load("LBPHFace.data");
			}
			else
			{
				cout<<"Training...";
				model->train(images, labels);
				//model->save("LBPHFace.data");
				cout<<"Done!"<<endl;
			}
			break;

		default:
			return c;
		}
	
		//下面对测试图像进行预测，predictedLabel是预测标签结果
		vector<Mat> images2;
		vector<int> labels2;

		cout<<"Loading images for testing...";
		for (int i = 0; i < 280; i++)
		{    
			char path[100];  
			sprintf(path,"..//..//人脸识别库2//TestDatabase//%d.jpg",i+1);
			images2.push_back(imread(path,0));
			labels2.push_back(i);		
		}
		cout<<"Done!"<<endl<<endl;

		namedWindow("Test",CV_WINDOW_AUTOSIZE);
		namedWindow("Predict",CV_WINDOW_AUTOSIZE);

		int rate = 0;
		for(int i = 0; i<280; i++)
		{
			int predictedLabel = -1;
			double confidence = 0.0;
			model->predict(images2[i], predictedLabel, confidence);	
			if(predictedLabel/10 == i/10)
			{
				cout<<"正确 ";
				rate++;
			}
			else
				cout<<"错误 ";
			cout<<"Test No."<<i+1<<"; Predicted No."<<predictedLabel<<" ;Confidence:"<<confidence<<endl;
			imshow("Test",images2[i]);
			imshow("Predict",images[predictedLabel]);
			waitKey();
		}

		cout<<"识别率:"<<rate/280.0<<endl;
		waitKey();
	}
}