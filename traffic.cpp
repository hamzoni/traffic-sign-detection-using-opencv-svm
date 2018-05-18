#include "traffic.h"

//#define dark
 #define debug
//#define single
 #define trackbar
Mat trackImg;

void Traffic::preprocess(Mat &img, Mat &img2, Mat &hsv, Mat &gray) {
    // cv::resize(img, img, Size(img.cols * this->resizeRatio, img.rows * this->resizeRatio), 0, 0, cv::INTER_LANCZOS4);
   cv::flip(img, img, 1);
    img.copyTo(img2);

//    cvtColor(img, img2, cv::COLOR_RGB2BGR);
    cvtColor(img2, gray, COLOR_BGR2GRAY);
    cvtColor(img2, hsv, COLOR_BGR2HSV);
}


Traffic::Traffic()
{
    this->isTrain = 0;
    this->kernel = Size(64, 48);

    this->trackImg = Mat();
    this->resizeRatio = 0.7;
    this->model = "/home/taquy/train.txt";
    #ifdef dark // dark
    this->mc = 3;
    this->mo = 0;
    this->lower = Scalar(0,0,0);
    this->upper = Scalar(255,255,114);
    #else // light
    this->mc = 10;
    this->mo = 0;
//    this->lower = Scalar(60, 105, 83);
//    this->upper = Scalar(255,255,255);

    this->lower = Scalar(55, 78, 0);
    this->upper = Scalar(255, 255, 255);

    // this->lower = Scalar(143, 23, 86);
    // this->upper = Scalar(255, 255, 255);
    #endif

    this->data = "/home/taquy/Projects/python/svm-train/data/";
    this->sample = "/home/taquy/Projects/python/svm-train/data/1/81.jpg";
    
    this->labels.push_back("Right"); // 0
    this->labels.push_back("Left"); // 1
    this->labels.push_back("Stop"); // 2
    this->labels.push_back("Box"); // 3
    this->labels.push_back("None"); // 4
    

    // create window for slider testing
    #ifdef debug
    namedWindow("mask", CV_WINDOW_AUTOSIZE);
    this->vals[0] = this->mo;  this->slider(this->vals[0], 200, "Opening", "mask");
    this->vals[1] = this->mc; this->slider(this->vals[1], 200, "Closing", "mask");

    this->vals[2] = this->lower[0]; this->slider(this->vals[2], 255, "Lower B", "mask");
    this->vals[3] = this->lower[1]; this->slider(this->vals[3], 255, "Lower R", "mask");
    this->vals[4] = this->lower[2]; this->slider(this->vals[4], 255, "Lower G", "mask");

    this->vals[5] = this->upper[0]; this->slider(this->vals[5], 255, "Upper B", "mask");
    this->vals[6] = this->upper[1]; this->slider(this->vals[6], 255, "Upper R", "mask");
    this->vals[7] = this->upper[2]; this->slider(this->vals[7], 255, "Upper G", "mask");
    #endif

    this->load();
    cout << this->kernel << endl;
}

void Traffic::induct(Mat gray, Mat hsv, Mat img){
    this->gray = gray;
    this->hsv = hsv;
    this->img = img;
}

int Traffic::taquy() {
    if (this->gray.empty()) {
        cout << "Error: lodge image required" << endl;
        return - 1;
    }

    #ifndef single
    vector<int> ids = this->detectMult();
    for (int i = 0; i < ids.size(); i++) {
        if (ids[i] >= 0 && ids[i] < this->labels.size()) return ids[i];
    }
    #else
    return this->detect();
    #endif
    return -1;
}

void Traffic::load() {
    if (this->isTrain == 0) {
        cout << "Loading model ..." << endl;
        try {
            svm = SVM::create();
            svm = SVM::load(this->model);
            cout << "Model loads successful." << endl;
        } catch (const std::exception& e) {
            cout << "Load model failed." << endl;
        }
    }
}


void filter(Mat &mask, Traffic *d){
    #ifndef debug
    inRange(d->hsv, cv::Scalar(60, 105, 83), cv::Scalar(255, 255, 255), mask);

    inRange(d->hsv, d->lower, d->upper, mask);
    Mat k(d->mo, d->mo, CV_8U, Scalar(1));
    Mat m(d->mc, d->mc, CV_8U, Scalar(1));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, k);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, m);
    #else

//    inRange(hsv, cv::Scalar(60, 105, 83), cv::Scalar(255, 255, 255), mask);

    inRange(d->hsv, Scalar(d->vals[2], d->vals[3], d->vals[4]), Scalar(d->vals[5], d->vals[6], d->vals[7]), mask);
    Mat k(d->vals[0], d->vals[0], CV_8U, Scalar(1));
    Mat m(d->vals[1], d->vals[1], CV_8U, Scalar(1));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, k);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, m);
    #endif
    medianBlur(mask, mask, 3);
}
    

Rect Traffic::pooling(Mat &mask, Mat &out){

    vector< vector<Point> > contours;
    vector<Vec4i> hierarchy;

    findContours(mask, contours, hierarchy, RETR_TREE , CHAIN_APPROX_SIMPLE, Point(0, 0) );

    if (contours.size() <= 0) return Rect();


    int largest_area = 0;
    int largest_contour = 0;

    Rect rect;

    for(unsigned int i = 0; i< contours.size(); i++) {
        double a = contourArea( contours[i],false);
        if( a > largest_area){
            largest_area = a;
            largest_contour = i;
            rect = boundingRect(contours[i]);
        }
    }

    if(rect.area() < mask.size().height * mask.size().width) {

        #ifdef debug
            cv::rectangle(mask, rect, Scalar(255, 255, 255), 1, 8, 0);
            cv::rectangle(this->gray, rect, Scalar(255, 255, 255), 1, 8, 0);
            imshow("gray", gray);
        #endif

        gray(rect).copyTo(out);
        return rect;
    }

}

vector<Rect> Traffic::poolingMult(Mat &mask, vector<Mat> &outs){
    vector<Rect> rects;
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

    findContours(mask, contours, hierarchy, RETR_TREE , CHAIN_APPROX_SIMPLE, Point(0, 0) );

    if (contours.size() <= 0) return rects;

    for(unsigned int i = 0; i< contours.size(); i++) {
        Rect rect = boundingRect(contours[i]);
        int minw = 30;
        int minh = 30;
        double rti = rect.width / rect.height;
        if(rect.area() < mask.size().height * mask.size().width
            && rect.area() > minw * minh
//            && rti >= 0.8 && rti <= 1.2
        ) {
            rects.push_back(rect);

            Mat out;
            gray(rect).copyTo(out);
            outs.push_back(out);
        }

    }

    #ifdef debug
    for (int i = 0; i < rects.size(); i++) {
        Rect rect = rects[i];
        cv::rectangle(mask, rect, Scalar(255, 255, 255), 1, 8, 0);
        cv::rectangle(this->gray, rect, Scalar(255, 255, 255), 1, 8, 0);
    }
    imshow("gray", this->gray);
    #endif
    return rects;
}

void on_track(int, void * x) {
    Traffic *d = (Traffic *) x;
    if (d->hsv.empty()) return;
    Mat mask;
    filter(mask, d);
    cv::imshow("mask", mask);

}

void Traffic::slider(int &val, int max, string title, string wname) {
    #ifdef trackbar
    title = title + " " + std::to_string(max);
    createTrackbar(title, wname, &val, max, on_track, this);
    #endif
}

int Traffic::predict(Mat &test) {
      resize(test,test,this->kernel);
      HOGDescriptor hog(this->kernel, Size(8,8), Size(4,4), Size(4,4), 9);
      vector<float> size_;
      hog.compute(test,size_);
      int col= size_.size();
      Mat testMat(1, col, CV_32FC1);
      vector<float> des;
      hog.compute(test,des);
      for (int i = 0; i < col; i++)
      testMat.at<float>(0,i)= des[i];
      return this->svm->predict(testMat);
}

string Traffic::label(int &id) {
    if (id < 0 || id >= this->labels.size())
        id = this->labels.size() - 1;
    return this->labels[id];
}

int Traffic::train(){

    if (this->isTrain) return -1;

    int m = 0;
    int numfiles;
    Mat img, img0;
    vector<string> files;
    vector<string> folders;

    string f = this->data;

    img0 = imread(this->sample, 1);
    cv::cvtColor(img0, img0, cv::COLOR_BGR2RGB);

    if(img0.empty()){
        cout << "Failed open image 0" << endl;
        return -1;
    }

    Mat re;

    resize(img0, re, this->kernel);

    vector<float> size_;

    HOGDescriptor hog(this->kernel, Size(8,8), Size(4,4), Size(4,4), 9);

    hog.compute(re, size_);

    this->lsdirs(f, folders);

    for(string s: folders) cout << s << endl;

    int num_imgs = 200;
    numfiles = num_imgs  * folders.size();
    Mat labels( numfiles, 1, CV_32S);
    Mat trainMat( numfiles, size_.size(), CV_32FC1);

    for(unsigned int x = 0; x < folders.size(); x++ ){
    this->lsfiles(folders[x], files);
    cout << folders[x] << endl;

    for(unsigned int y = 0; y< num_imgs; y++){
        cout << files[y] << endl;

        img = imread(files[y], 0);

        resize(img, img, this->kernel);

        vector<float> des;
        hog.compute(img,des);
        for (unsigned int i = 0; i < size_.size(); i++)
            trainMat.at<float>(m,i)= des[i];
            labels.at<int>(m, 0) = x;
            m++;
        }
    }

    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::LINEAR);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 10000, 1e-6));
    svm->train(trainMat, ROW_SAMPLE, labels);
    svm->save(f + "train.txt");
    cout << "FINISHED" << endl;

}


int Traffic::detect() {

    Mat mask, out;

    // must be BGR
    filter(mask, this);

    #ifdef debug
    hsv.copyTo(this->trackImg);
    imshow("mask", mask);
    #endif

    // detect object
    Rect box = this->pooling(mask, out);

    if(!out.empty()){

        // classification
        int id = predict(out);

        #ifdef debug

            cv::resize(out, out, this->kernel);

            vector<Rect> boxes;
            boxes.push_back(box);

            string lbl = this->label(id);
            cout << this->label(id) << endl;

            Mat rsl;
            cv::resize(out, rsl, Size(out.cols * 2, out.rows * 2));


            if (id != this->labels.size() - 1) {
                this->img = this->draw(this->img, boxes, lbl);

                imshow("region", rsl);
            } else {
                imshow("region", rsl);
            }

        #endif

        return id;
    }

    return -1;
}

vector<int> Traffic::detectMult() {
    Mat mask, out;
    vector<int> ids;

    filter(mask, this);

    #ifdef debug
    this->hsv.copyTo(this->trackImg);
    imshow("mask", mask);
    #endif

    vector<Mat> outs;
    vector<Rect> objs = this->poolingMult(mask, outs);


    if(!objs.size() == 0) {
        for (int i = 0; i < outs.size(); i++) {
            Mat out = outs[i];
            // classification
            int id = predict(out);
            ids.push_back(id);

            #ifdef debug
            vector<Rect> boxes;
            boxes.push_back(objs[i]);

            string lbl = this->label(id);
            cout << this->label(id) << endl;


            if (id != this->labels.size() - 1) {
                this->img = this->draw(this->img, boxes, lbl);
            }
            #endif
        }
    }

    #ifdef debug
    cv::imshow("result", this->img);
    #endif

    return ids;
}

void Traffic::conversion(Mat frame, Mat &gray, Mat &hsv) {
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
}

void Traffic::testvid(string vid, int wv) {

    cv::VideoCapture video(vid);

    int frame_counter = 0;
    double st = 0, et = 0, fps = 0;
    double freq = getTickFrequency();

    cv::Mat frame, gray, hsv;
    while(true){

        if (frame_counter == video.get(CV_CAP_PROP_FRAME_COUNT)) {
            frame_counter = 0 ;
            video.set(CV_CAP_PROP_POS_FRAMES, 0);
        }


        st = getTickCount();
        video >> frame;
        this->conversion(frame, gray, hsv);

        if(frame.empty()) continue;
        this->induct(gray, hsv, frame);

        int id = this->taquy();
        cout << id << endl;

        int k = cv::waitKey(wv) & 0xff;

        et = getTickCount();
        fps = 1.0 / ((et-st) / freq);
        cout << "FPS: "<< fps<< '\n';

        if(k == 27) break;
        if(k == 38) wv += 10;
        if(k == 32) waitKey();
    }
}

Mat Traffic::draw(Mat frame, vector<Rect> boxes, String label) {

    // draw rects
    for( size_t i = 0; i < boxes.size(); i++ )
    {
        int x = boxes[i].x;
        int y = boxes[i].y;
        Point a(x, y);
        Point b(x + boxes[i].width, y + boxes[i].height);

        rectangle(frame, a, b, Scalar(0, 255, 0), 3);

        putText(frame, label, a, FONT_HERSHEY_SIMPLEX, 1, (0, 125, 255), 3, 0, false);
    }
    return frame;
}

void Traffic::lsdirs(string path, vector<string> &folders){
    folders.clear();
    struct dirent *entry;

    DIR *dir = opendir(path.c_str());

    while((entry = readdir(dir))!= NULL){
        if ((strcmp(entry->d_name, ".") != 0) && (strcmp(entry->d_name, "..") != 0)) {
            string s =  string(path)  + string(entry->d_name) ;
            folders.push_back(s);
        }
    }

    closedir(dir);
    sort(folders.begin(),folders.end());
}

void Traffic::lsfiles(string path, vector<string> &files) {

    files.clear();

    struct dirent *entry;

    DIR *dir = opendir(path.c_str());

    while((entry = readdir(dir))!= NULL){
        if ((strcmp(entry->d_name, ".") != 0) && (strcmp(entry->d_name, "..") != 0)) {
            string s =  string(path) + "/" + string(entry->d_name) ;
            files.push_back(s);
        }
    }

    closedir(dir);
    sort(files.begin(),files.end());
}
