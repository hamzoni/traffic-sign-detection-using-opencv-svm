#include <traffic.h>
#include <iostream>
#include <opencv2/ml/ml.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <dirent.h>
#include <time.h>
using namespace std;

double st = 0, et = 0, fps = 0;
double freq = getTickFrequency();

void testTrain() {
    Traffic *d;
    d = new Traffic();
    d->train();
}

void conversion(Mat frame, Mat &gray, Mat &hsv) {
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
}

void testCam(){

    cv::VideoCapture video(0);
    cv::Mat frame, gray;
    Traffic *d = new Traffic();

    while(true){


        st = getTickCount();
        video >> frame;
        if(frame.empty()) continue;

        Mat gray, hsv;
        conversion(frame, gray, hsv);

        d->induct(gray, hsv, frame);
        int id = d->taquy();

        cout << id << endl;

        int k = cv::waitKey(1) & 0xff;

        et = getTickCount();
        fps = 1.0 / ((et-st) / freq);
        cout << "FPS: "<< fps<< '\n';

        if(k == 27) break;
        if(k == 32) waitKey();
    }
}

int main(){
//    testTrain();
    testCam();
//    Traffic *dsvm = new Traffic();
//    dsvm->testvid("/home/taquy/Projects/python/svm-train/4.avi", 200);

//    string vid = "/home/taquy/Desktop/c/14.avi";
//    dsvm->testvid(vid, 200);
}
