/***********************************
   ����ͼ����ʵ��-����ʶ��

�ļ���Output.cpp
��飺���������ʶ��ʽ�Ļ����ϣ�ͳ
���˵�ѵ��ʱ������Ԥ��ʱ�䣬���⻹ͳ
���˸���ʽ��ÿ��������ʶ��������Ա�
��������������

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
	//ѧϰѵ������
	vector<Mat> images;
	vector<int> labels;
	
	cout<<"Loading images for training...";
	for (int i = 0; i < 280; i++)
	{    
		char path[100];  
		sprintf(path,"..//..//����ʶ���2//TrainDatabase//%d.jpg",i+1);
		images.push_back(imread(path,0));
        labels.push_back(i);
	}
	cout<<"Done!"<<endl;
	
	clock_t  t_start,t_end;

	while(1)
	{
		int c;
		cout<<"Select the recognize algorithm(input number)��\n 1��Eigen Face;\n 2��Fisher Face;\n 3��LBPH Face;\n Other��Exit.\n";
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
				t_start = clock() ;
				model->train(images, labels);
				t_end = clock();
				cout<<"ѵ��ʱ�䣺"<< (double)((double)(t_end - t_start) / CLOCKS_PER_SEC)<<endl;
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
				t_start = clock() ;
				model->train(images, labels);
				t_end = clock();
				cout<<"ѵ��ʱ�䣺"<< (double)((double)(t_end - t_start) / CLOCKS_PER_SEC)<<endl;
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
				t_start = clock() ;
				model->train(images, labels);
				t_end = clock();
				cout<<"ѵ��ʱ�䣺"<< (double)((double)(t_end - t_start) / CLOCKS_PER_SEC)<<endl;
				//model->save("LBPHFace.data");
				cout<<"Done!"<<endl;
			}
			break;

		default:
			return c;
		}
	
		//����Բ���ͼ�����Ԥ�⣬predictedLabel��Ԥ���ǩ���
		vector<Mat> images2;
		vector<int> labels2;

		cout<<"Loading images for testing...";
		for (int i = 0; i < 280; i++)
		{    
			char path[100];  
			sprintf(path,"..//..//����ʶ���2//TestDatabase//%d.jpg",i+1);
			images2.push_back(imread(path,0));
			labels2.push_back(i);		
		}
		cout<<"Done!"<<endl<<endl;
		
		int rate = 0;
		int each = 0;

		stringstream ss;
		string str;
		ss<<c;
		ss>>str;
		ofstream f(str + ".txt");
		int cc = 0;

		t_start = clock() ;
		if (f)
		for(int i = 0; i<280; i++)
		{
			int predictedLabel = -1;
			double confidence = 0.0;
			model->predict(images2[i], predictedLabel, confidence);	
			if(predictedLabel/10 == i/10)
			{
				rate++;
				each++;
			}
			if(i%10 == 9)
			{
				f<<each<<endl;
				each=0;
				cc++;
			}
		}
		f.close();
		t_end = clock();

		cout<<"Ԥ��ʱ�䣺"<<(double)((double)(t_end - t_start) / CLOCKS_PER_SEC)<<endl;
	
		cout<<cc<<endl;
		cout<<"ʶ����:"<<rate/280.0<<endl;
		waitKey();
	}
}