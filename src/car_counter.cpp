// Denis Kotnik, november 2020
// https://www.kotnik.si
// Code not clean.

#include <opencv2/video/background_segm.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/highgui.hpp>
#include "opencv2/opencv.hpp"
#include <opencv2/video.hpp>
#include <opencv2/features2d.hpp>
#include <stdio.h>
#include <iostream>
#include <sstream>

using namespace cv;
using namespace std;

#define WINDOW_NAME_BACKGROUND "Background subtraction"
#define WINDOW_NAME_ORIGINAL "Original"
#define WINDOW_NAME_MASK "Mask - POI"
#define MODE_OPENING 1
#define MODE_CLOSING 2
#define MAX_SIZE 6
// Morphology.
int EROSION_SIZE = 2;
int DILATION_SIZE = 2;

// Points for lines for which cars need to cross in order to increase car coutner.
Point firstLineStart(0,0);
Point firstLineEnd(0,0);
Point secondLineStart(0,0);
Point secondLineEnd(0,0);

int keyboard;
int modeOfChoosing = 1;

vector <cv::Point> poiPoints;
Mat videoFrame, maskFrame, maskedFrame, background;
Ptr<BackgroundSubtractorKNN> backgSubtrKNN;

////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////

string intToString (int number) {
    ostringstream oss;
    oss << number;
    return oss.str();
}

// Create a convex hull from a points which we choose with mouse clicks.
Mat convexMask() {
    // Calculate convex hull from a vector of points.
    vector<Point> hull;
    convexHull(Mat(poiPoints), hull, true);

    // Approximation of points into lines and curves.
    vector<cv::Point> polyright;
    approxPolyDP(Mat(hull), polyright, 0.001, true);

    Mat mask = Mat(videoFrame.rows, videoFrame.cols, CV_8UC1);
    mask = 0;
    // Fill the empty space - create a mask.
    fillConvexPoly(mask, &polyright[0], polyright.size(), Scalar(255,255,255), 8, 0);
    Mat dst;
    videoFrame.copyTo(dst, mask);    

    return dst;
}

// Morphology (dilate, erode).
Mat doMorphology(Mat src, int mode, int structElement) {
    Mat erode_element = getStructuringElement(structElement, Size(2 * EROSION_SIZE + 1, 2 * EROSION_SIZE + 1), Point(EROSION_SIZE, EROSION_SIZE));
    Mat dilate_element = getStructuringElement(structElement, Size(2 * DILATION_SIZE + 1, 2 * DILATION_SIZE + 1), Point(DILATION_SIZE, DILATION_SIZE));

    Mat tmp, dst;
    switch (mode) {
        case MODE_OPENING:
            erode(src, tmp, erode_element);
            dilate(tmp, dst, dilate_element);
            break;
        case MODE_CLOSING:
            dilate(src, tmp, dilate_element);
            erode(tmp, dst, erode_element);
            break;
    }
    return dst;
}

// Check on which side of the line the point of the car appears (check the +/- of the determinant).
bool isLeft(Point a, Point b, Rect r){
    // Calculate the cener of the square (car).
    Point2f c = Point2f(r.x + r.width / 2, r.y + r.height / 2);
    return ((b.x - a.x)*(c.y - a.y) - (b.y - a.y)*(c.x - a.x)) > 0;
}

bool betweenTwoPoints(Point a, Point b, Rect r) {
    // Calculate the cener of the square (car).
    Point2f c = Point2f(r.x + r.width / 2, r.y + r.height / 2);
    
    Point leftPoint, rightPoint;
    if (a.x < b.x) {
        leftPoint = a;
        rightPoint = b;
    }
    else {
        leftPoint = b;
        rightPoint = a;
    }
    return c.x > leftPoint.x && c.x < rightPoint.x;
}

void mouse(int event, int x, int y, int flags, void* data) {
    if(event == EVENT_LBUTTONDOWN) {
        if (modeOfChoosing == 1) {
            // Selecting the mask
            circle(maskFrame, Point(x, y), 10, Scalar(255, 0, 0, 100), -1);
            poiPoints.push_back(cv::Point(x,y));
        }
        else if (modeOfChoosing == 2) {
            // Selecting the first line
            if (firstLineStart == Point(0,0))
                firstLineStart = Point(x, y);
            else {
                firstLineEnd = Point(x, y);
                line(videoFrame, firstLineStart, firstLineEnd, Scalar(0, 0, 255),2);
            }
        }
        else if (modeOfChoosing == 3) {
            // Selecting the second line
            if (secondLineStart == Point(0,0))
                secondLineStart = Point(x, y);
            else {
                secondLineEnd = Point(x, y);
                line(videoFrame, secondLineStart, secondLineEnd, Scalar(0, 0, 255),2);
            }
        }
    }
}

int main(int argc, char* argv[])
{
    // Open video or camera
    VideoCapture capture;
    if (argc > 1){
        capture = VideoCapture(argv[1]);
    }
    else{
        capture = VideoCapture(0);
    }
    if(!capture.isOpened()){
        cerr << "Error opening the video file: " << argv[1] << endl;
        exit(EXIT_FAILURE);
    }

    // Create window for selecting the mask (POI)
    namedWindow(WINDOW_NAME_MASK);

    if(!capture.read(videoFrame)) {
        cerr << "Error capturing the next image" << endl;
        exit(EXIT_FAILURE);
    }
    ////////////////////////////////////////////////////
    // Selecting the mask and lines for car counting
    ////////////////////////////////////////////////////

    // Optional
    if (2 < argc && string(argv[2]) == "-small") {
        resize(videoFrame, videoFrame, Size(640, 480), 0, 0, INTER_CUBIC);
    }

    setMouseCallback(WINDOW_NAME_MASK, mouse, NULL);
        
    while(modeOfChoosing != 4) {
        Mat maskTemp = Mat::zeros(videoFrame.rows, videoFrame.cols, videoFrame.type());
        if (poiPoints.size() > 0) {
            maskTemp = convexMask();
        }
        // Mix (add) two different images: original + mask
        addWeighted(videoFrame, 0.5, maskTemp, 1.0, 0.0, maskFrame);

        imshow(WINDOW_NAME_MASK, maskFrame);
        
        keyboard = waitKey(10);
        switch((char)keyboard) {
            case '1':
                modeOfChoosing = 1;
                break;
            case '2':
                modeOfChoosing = 2;
                break;
            case '3':
                modeOfChoosing = 3;
                break;
            case '4':
                modeOfChoosing = 4;
                break;
            case 'c':
                firstLineStart = Point(0,0);
                firstLineEnd = Point(0,0);
                secondLineStart = Point(0,0);
                secondLineEnd = Point(0,0);
                poiPoints.clear();
                break;
        }
    }

    // Destroy the window for selecting the mask (POI)
    destroyWindow(WINDOW_NAME_MASK);

    ////////////////////////////////////////////////////
    // GUI
    ////////////////////////////////////////////////////

    namedWindow(WINDOW_NAME_ORIGINAL);
    namedWindow(WINDOW_NAME_BACKGROUND);
    createTrackbar("Erosion size", WINDOW_NAME_BACKGROUND, &EROSION_SIZE, MAX_SIZE);
    createTrackbar("Dilation size", WINDOW_NAME_BACKGROUND, &DILATION_SIZE, MAX_SIZE);

    // Ustvarimo Background Subtractor model (KNN) z ustreznimi nastavitvami.
    // Create background subtraction model (KNN) with custom settings
    backgSubtrKNN = createBackgroundSubtractorKNN();
    backgSubtrKNN->setDetectShadows(true);
    backgSubtrKNN->setShadowThreshold(0.5);
    // Remove shadows
    backgSubtrKNN->setShadowValue(0);

    // For blobs - cars in time t-1
    vector<Rect> rect_car_old;
    // For blobs - cars in time t
    vector<Rect> rect_car_new;
    
    int carCounterBack = 0, carCounterForward = 0;
    bool throughFirstLine = false, throughSecondLine = false;

    // Repeat until 'ESC'
    while((char)keyboard != 27 ) {
        if(!capture.read(videoFrame)) {
            cerr << "Error capturing the next image" << endl;
            exit(EXIT_FAILURE);
        }
        
        // Optional
        if (2 < argc && string(argv[2]) == "-small") {
            resize(videoFrame, videoFrame, Size(640, 480), 0, 0, INTER_CUBIC);
        }

        // Select appropriate color (green or red) for appropriate line
        Scalar firstLineColor, secondLineColor;
        if (throughFirstLine == true) {
            firstLineColor = Scalar(0, 255, 0);
            throughFirstLine = false;
        }
        else {
            firstLineColor = Scalar(0, 0, 255);
        }
        if (throughSecondLine == true) {
            secondLineColor = Scalar(0, 255, 0);
            throughSecondLine = false;
        }
        else {
            secondLineColor = Scalar(0, 0, 255);
        }

        // Draw the lines and car couters
        line(videoFrame, firstLineStart, firstLineEnd, firstLineColor, 2);
        putText(videoFrame, intToString(carCounterBack), Point(20, 40), FONT_HERSHEY_SIMPLEX, 1, firstLineColor, 1, 8);
        if (secondLineStart != Point(0,0)) {
            line(videoFrame, secondLineStart, secondLineEnd, secondLineColor, 2);
            putText(videoFrame, intToString(carCounterForward), Point(videoFrame.cols - 40, 40), FONT_HERSHEY_SIMPLEX, 1, secondLineColor, 1, 8);
        }
        
        ////////////////////////////////////////////////////
        // Image processing
        ////////////////////////////////////////////////////
        // Create mask (POI)
        maskedFrame = convexMask();

        blur(maskedFrame, maskedFrame, Size(5,5));
        
        // Update the background sub. model
        backgSubtrKNN->apply(maskedFrame, background);
        
        // Do morphology on binary image
        background = doMorphology(background, MODE_OPENING, MORPH_ELLIPSE);
        
        ////////////////////////////////////////////////////
        // Blob/contours searching - cars
        ////////////////////////////////////////////////////
        vector<vector<Point>> contours_raw;

        // RETR_EXTERNAL: only outer blobs, CHAIN_APPROX_SIMPLE: magic
        findContours(background.clone(), contours_raw, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        
        // Reliable blobs (convex hull)
        vector<vector<Point>> contours(contours_raw.size());

        // Create a convex hull around founded blobs
        for(unsigned int i = 0; i < contours_raw.size(); i++ ) {
            convexHull(Mat(contours_raw[i]), contours[i], true);
        }

        // Filtering according to size
        for(unsigned int i = 0; i < contours.size(); i++){
            double size = contourArea(contours[i], false);
            if (size > 500 && size < 2500){
                rect_car_new.push_back(boundingRect(contours[i]));
            }
        }

        ////////////////////////////////////////////////////
        // Drawing the squares around founded blobs - cars
        ////////////////////////////////////////////////////
        for (unsigned int i = 0; i < rect_car_new.size(); i++) {
            Point2f middle = Point2f(rect_car_new[i].x + rect_car_new[i].width / 2, rect_car_new[i].y + rect_car_new[i].height / 2);
            rectangle(videoFrame, rect_car_new[i],  Scalar(0,255,0),2, 8,0);
            circle(videoFrame, middle, 5, CV_RGB(0, 255, 0), -1);
        }

        ////////////////////////////////////////////////////
        // Finding pair of blobs - cars in time t and t-1 - shortest distance
        ////////////////////////////////////////////////////
        // Vector of tuples of indexes (i, j) of old and new blobs
        vector<pair<int, int>> old_new;
        // Finding the shortest distances
        for (unsigned int i = 0; i < rect_car_old.size(); i++) {
            double minDistance = 100;
            int indexOfMin = -1;

            for (unsigned int j = 0; j < rect_car_new.size(); j++) {
                // Distance between new and old blob
                double distance = sqrt(pow(rect_car_old[i].x - rect_car_new[j].x, 2) + pow(rect_car_old[i].y-rect_car_new[j].y, 2));
                
                if (distance < minDistance) {
                    minDistance = distance;
                    indexOfMin = j;
                }
            }
            if (indexOfMin != -1) {
                old_new.push_back(make_pair(i, indexOfMin));
            }
        }

        ////////////////////////////////////////////////////
        // For each pair of blob (cars) check if blob from time t-1 is under line AND blob from time t is on top of the line
        // If so, increment counter
        ////////////////////////////////////////////////////
        for (unsigned int i = 0; i < old_new.size(); i++) {
            cout << "Coordinates of car in time t-1: (" << rect_car_old[old_new[i].first].x << ", " << 
            rect_car_old[old_new[i].first].y << ") and time t: (" << rect_car_new[old_new[i].second].x << 
            ", " << rect_car_new[old_new[i].second].y << ")." << endl;

            // !isLeft = on the top of the line
            if (isLeft(firstLineStart, firstLineEnd, rect_car_old[old_new[i].first]) &&
                betweenTwoPoints(firstLineStart, firstLineEnd, rect_car_old[old_new[i].first]) &&
                !isLeft(firstLineStart, firstLineEnd, rect_car_new[old_new[i].second]) &&
                betweenTwoPoints(firstLineStart, firstLineEnd, rect_car_new[old_new[i].second])) {
                carCounterBack += 1;
                throughFirstLine = true;
            }
            if (secondLineStart != Point(0, 0)) {
                if (isLeft(secondLineStart, secondLineEnd, rect_car_old[old_new[i].first]) &&
                    betweenTwoPoints(secondLineStart, secondLineEnd, rect_car_old[old_new[i].first]) && 
                    !isLeft(secondLineStart, secondLineEnd, rect_car_new[old_new[i].second]) &&
                    betweenTwoPoints(secondLineStart, secondLineEnd, rect_car_new[old_new[i].second])) {
                    carCounterForward += 1;
                    throughSecondLine = true;
                }
            }
        }

        ////////////////////////////////////////////////////
        // New blobs from time t becomes old ones from time t-1, delete new ones
        ////////////////////////////////////////////////////
        rect_car_old.clear();
        for (unsigned int i = 0; i < rect_car_new.size(); i++) {
            rect_car_old.push_back(rect_car_new[i]);
        }
        rect_car_new.clear();
        contours.clear();

        ////////////////////////////////////////////////////
        // Show image and mask
        ////////////////////////////////////////////////////
        imshow(WINDOW_NAME_ORIGINAL, videoFrame);
        imshow(WINDOW_NAME_BACKGROUND, background);

        keyboard = waitKey(10);
    }

    capture.release();
    destroyAllWindows();

    return EXIT_SUCCESS;
}