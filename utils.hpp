#ifndef _UTILS_H
#define _UTILS_H

#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "card.hpp"

using namespace cv;
using namespace std;

void create_masks(void);
Mat get_outer_contours(Mat img, vector<vector<Point>> &contours, Mat &th_mask, bool approx, double ratio_epsilon);
vector<vector<Card>> get_sets(const vector<Card> &deck);

#endif