#ifndef SVMDETECTOR_H
#define SVMDETECTOR_H

#endif // SVMDETECTOR_H
#ifndef Traffic_H
#define Traffic_H

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/objdetect.hpp>
#include <dirent.h>
#include <string>

using namespace cv;
using namespace ml;
using namespace std;


class Traffic
{
private:

    Ptr<SVM> svm;
    string model;  // model file
    string data;  // data train folder
    string sample; // seeding file

    vector<string> labels; // labels

    void load();

public:

    int isTrain;
    Mat trackImg;

    double resizeRatio;

    Scalar lower, upper;

    Mat gray, hsv, img;

    void induct(Mat gray, Mat hsv, Mat img);

    int mc, mo;

    Mat closing, opening;

    Traffic();

    Size kernel;

    int vals[100];

    int taquy();

    void preprocess(Mat &img, Mat &img2, Mat &hsv, Mat &gray);

    Rect pooling(Mat &mask, Mat &out);

    vector<Rect> poolingMult(Mat &mask, vector<Mat> &outs);

    int predict(Mat &test);

    string label(int &id);

    int train();

    int detect();

    vector<int> detectMult();

    void testvid(string vid, int wv);

    Mat draw(Mat frame, vector<Rect> boxes, String label);

    void lsdirs(string path, vector<string> &folders);

    void lsfiles(string path, vector<string> &files);

    void slider(int &val, int max, string title, string wname);

    void conversion(Mat frame, Mat &gray, Mat &hsv);

};


#endif // Traffic_H
