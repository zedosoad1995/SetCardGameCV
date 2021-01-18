#include <iostream>
#include <string>
#include <dirent.h>
#include <mutex>
#include <algorithm>

#include <tbb/parallel_for.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/features2d.hpp>

#include "card.hpp"
#include "utils.hpp"


using namespace std;
using namespace cv;

std::mutex mtx;

struct triangle_color{
  Scalar color;
  int count = 0;
  vector<Card> set;
};

double normalize_0_180(double angle);
bool check_contour_angle(const vector<Point> & contour, float desired_angle, float error = 45);
void transform_contours(vector<vector<Point>> &contours, const Mat &projective_matrix);
Point mean_point_contour(const vector<Point> &contour);
Scalar get_triangle_color(const vector<Card> &set, vector<triangle_color> &triangle_colors, int n_max = 5);

int main()
{

    vector<Scalar> predefined_triangle_colors {Scalar(255,0,0), Scalar(0,255,0), Scalar(0,0,255), Scalar(255,255,0), Scalar(255,0,255),
                                                Scalar(0,255,255), Scalar(255,255,255), Scalar(0,0,0),
                                                Scalar(122,0,0), Scalar(0,122,0), Scalar(0,0,122),
                                                Scalar(122,122,0), Scalar(122,0,122), Scalar(0,122,122),
                                                Scalar(122,122,122),
                                                Scalar(255,122,0), Scalar(122,255,0), Scalar(0,255,122),
                                                Scalar(0,122,255), Scalar(122,0,255), Scalar(255,0,122)};
    vector<triangle_color> triangle_colors;
    triangle_color triangle_color_aux;
    for(size_t i = 0; i < 21; i++){
        triangle_color_aux.color = predefined_triangle_colors[i];
        triangle_colors.push_back(triangle_color_aux);
    }

    float zoom = 0.33;
    Size card_size(560*zoom, 865*zoom);

    Point2f ur(560*zoom, 0), ul(0, 0), dl(0, 865*zoom), dr(560*zoom, 865*zoom);

    // Card coordinates
    vector<Point2f> card_points;
    card_points.push_back(ur);
    card_points.push_back(ul);
    card_points.push_back(dl);
    card_points.push_back(dr);

    vector<Point2f> card_points_rotate;
    card_points_rotate.push_back(ul);
    card_points_rotate.push_back(dl);
    card_points_rotate.push_back(dr);
    card_points_rotate.push_back(ur);

    Mat transform_matrix_rotate_90 = getPerspectiveTransform(card_points, card_points_rotate);

    //VideoCapture cap("https://XXX.XXX.XXX.XXX:8080/video");
    VideoCapture cap("opencv_test/data/templates/video8.mp4");
    	 
    if(!cap.isOpened())
        return -1; 

    Mat img;

    int counter = 0;

    while(1){ 
        
        // uncomment this if IP camera is having delay, and comment cap >> img
        //cap.grab();
        //if(counter%5 != 0) continue;
        //cap.retrieve(img);

        counter++;

	    cap >> img;

	    if (img.empty())
            break;

        ////////////////////
        // PRE-PROCESSING //
        ////////////////////
        // to grayscale
        Mat gray_img, gray_img_;
        cvtColor(img, gray_img_, COLOR_BGR2GRAY);

        // Gaussian smoothing
        GaussianBlur(gray_img_, gray_img_, Size(7, 7), 0);
        GaussianBlur(gray_img_, gray_img, Size(7, 7), 0);
        addWeighted(gray_img_, 1.37, gray_img, -0.37, 0, gray_img);

        // Histogram linear stretching
        normalize(gray_img, gray_img, 0, 255, cv::NORM_MINMAX);

        // canny edge detector
        Mat img_canny;
        int lowThreshold = 5;
        const int ratio = 3;
        const int kernel_size = 3;
        Canny(gray_img, img_canny, lowThreshold, lowThreshold*ratio, kernel_size);

        // Find the contours
        vector<vector<Point> > contours, contours2;
        vector<Vec4i> hierarchy;

        findContours(img_canny, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);

        // APAGAR
        int n_rectangles_detected = 0;

        vector<Card> deck;

        // parallel computing
        // Check all cards in the contours
        tbb::parallel_for(size_t (0), contours.size(),
                       [&](size_t i)
        {
            //auto start = high_resolution_clock::now(); 

            double contour_len = arcLength(contours[i], true);
            double contour_area = contourArea(contours[i]);

            // Simplify contours
            approxPolyDP( contours[i], contours[i], 0.08*contour_len, true );

            // Only allow for contours that are closed, have 4 vertices (rectagle/card shape), are big enough and all angles are around 90 degrees
            if(contours[i].size() == 4 && contour_area > 2300 && contour_area > contour_len && check_contour_angle(contours[i], 90)){
                
                vector<Point2f> contours2f;
                std::transform(contours[i].begin(), contours[i].end(),
                            std::back_inserter(contours2f),
                            [](const Point& p) { return (Point2f)p; });

                // Performs a projective transformation of the card
                Mat projected_card, projected_card_original_size;
                Mat transform_matrix = getPerspectiveTransform(contours2f, card_points);
                warpPerspective(img, projected_card, transform_matrix, card_size);

                // Gets contours of tokens in the card (symbols)
                Mat gray_projected_card;

                cvtColor(projected_card, gray_projected_card, COLOR_BGR2GRAY);
                normalize(gray_projected_card, gray_projected_card, 0, 255, cv::NORM_MINMAX);
                adaptiveThreshold(gray_projected_card, gray_projected_card, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 181, 10);

                Mat tokens_mask(gray_projected_card.rows, gray_projected_card.cols, CV_8UC1, Scalar::all(0));

                vector<vector<Point>> token_contours_aux, token_contours_approx_aux, tokens_contours, tokens_contours_approx;
                vector<Vec4i> hierarchy;
                findContours(gray_projected_card, token_contours_aux, hierarchy, RETR_TREE, CHAIN_APPROX_TC89_L1);

                token_contours_approx_aux = token_contours_aux;

                for(int c = 0; c < token_contours_aux.size(); c++){
                    if(hierarchy[c][3] == -1){
                        int cc = hierarchy[c][2];
                        while(cc != -1){
                            contour_len = arcLength(token_contours_aux[cc], true);
                            approxPolyDP(token_contours_aux[cc], token_contours_approx_aux[cc], 0.025*contour_len, true);
                            if(contourArea(token_contours_approx_aux[cc]) > 5000*zoom){
                                drawContours(tokens_mask, token_contours_aux, cc, Scalar(255, 255, 255), -1, 8);
                                tokens_contours.push_back(token_contours_aux[cc]);
                                tokens_contours_approx.push_back(token_contours_approx_aux[cc]);
                            }
                            cc = hierarchy[cc][0];
                        }
                    }
                }
                
                Point top_point(0, 0), bottom_point(100000, 100000);

                if(tokens_contours.empty()){
                    return;
                }

                // Gets top and bottom points of token
                for(const Point &point: tokens_contours[0]){
                    if(point.y > top_point.y) top_point = point;
                    if(point.y < bottom_point.y) bottom_point = point;
                }

                // check if card is turned in the wrong rotation (horizontal intead of vertical)
                if(norm(top_point - bottom_point) > 400*zoom){     

                    transform_contours(tokens_contours, transform_matrix_rotate_90);
                    for(size_t c = 0; c < tokens_contours.size(); c++){
                        contour_len = arcLength(tokens_contours[c], true);
                        approxPolyDP(tokens_contours[c], tokens_contours_approx[c], 0.025*contour_len, true);
                        //drawContours(tokens_mask2, tokens_contours_approx, c, Scalar(255, 255, 255), -1, 8);
                    }
                    warpPerspective(projected_card, projected_card, transform_matrix_rotate_90, card_size);
                    warpPerspective(tokens_mask, tokens_mask, transform_matrix_rotate_90, card_size);
                }


                // Get average color value inside and outside contours
                cvtColor(projected_card, gray_projected_card, COLOR_BGR2GRAY);
                erode(tokens_mask, tokens_mask, getStructuringElement(MORPH_RECT, Size(35*zoom*2, 35*zoom*2)));
                Scalar average_gray_inside_tockens = mean(gray_projected_card, tokens_mask);
                bitwise_not(tokens_mask, tokens_mask);
                erode(tokens_mask, tokens_mask, getStructuringElement(MORPH_RECT, Size(41*zoom*2, 41*zoom*2)));
                Scalar average_gray_outside_tockens = mean(gray_projected_card, tokens_mask);

                // Choose fill
                Fill fill_enum;
                string fill;
                if(norm(average_gray_inside_tockens - average_gray_outside_tockens) > 70){
                    fill = "solid";
                    fill_enum = SOLID;
                }else if(norm(average_gray_inside_tockens - average_gray_outside_tockens) > 5){
                    fill = "dashed";
                    fill_enum = STRIPES;
                }else{
                    fill = "empty";
                    fill_enum = OUTLINE;
                }
                
                // Transform white
                Scalar average_white = mean(projected_card, tokens_mask);
                Mat average_scalar_rgb = Mat(1, 1, CV_8UC3, average_white);
                Vec3b pixel = average_scalar_rgb.at<Vec3b>(0, 0);
                
                for( int y = 0; y < projected_card.rows; y++ ) {
                    for( int x = 0; x < projected_card.cols; x++ ) {
                        for( int c = 0; c < projected_card.channels(); c++ ) {
                            projected_card.at<Vec3b>(y,x)[c] = saturate_cast<uchar>(255*projected_card.at<Vec3b>(y,x)[c]/pixel[c]);
                        }
                    }
                }
                
                // Get Colors
                Mat hsv, mask_red_high, mask_red_low, mask_red, mask_purple, mask_green;
                cvtColor(projected_card, hsv, COLOR_BGR2HSV);
                inRange(hsv, Scalar(220/2, 3*255/100, 3*255/100), Scalar(320/2, 100*255/100, 100*255/100), mask_purple);
                inRange(hsv, Scalar(63/2, 3*255/100, 3*255/100), Scalar(190/2, 100*255/100, 100*255/100), mask_green);
                inRange(hsv, Scalar(320/2, 3*255/100, 3*255/100), Scalar(360/2, 100*255/100, 100*255/100), mask_red_high);
                inRange(hsv, Scalar(0/2, 3*255/100, 3*255/100), Scalar(62/2, 100*255/100, 100*255/100), mask_red_low);

                bitwise_not(tokens_mask, tokens_mask);
                erode(tokens_mask, tokens_mask, getStructuringElement(MORPH_RECT, Size(7, 7)));
                bitwise_and(mask_purple,tokens_mask,mask_purple);
                bitwise_and(mask_green,tokens_mask,mask_green);
                mask_red = mask_red_high + mask_red_low;
                bitwise_and(mask_red,tokens_mask,mask_red);

                // Count number of pixels for each color
                int n_purple = countNonZero(mask_purple);
                int n_green = countNonZero(mask_green);
                int n_red = countNonZero(mask_red);

                string color_name;
                Color color_enum;
                if(n_purple > n_green && n_purple > n_red){
                    color_name = "purple";
                    color_enum = PURPLE;
                }
                else if(n_green > n_purple && n_green > n_red){
                    color_name = "green";
                    color_enum = GREEN;
                }
                else{
                    color_name = "red";
                    color_enum = RED;
                }
            

                // Get shape
                double vertices = 0, convexity = 0;
                for(int i = 0; i < tokens_contours_approx.size(); i++){
                    vertices += tokens_contours_approx[i].size();
                    if(isContourConvex(tokens_contours_approx[i])) convexity++;
                }

                Shape shape_enum;

                if(abs(vertices/tokens_contours_approx.size() - 4) < 1.1){
                    shape_enum = DIAMOND;
                    //cv::putText(projected_card, to_string(tokens_contours_approx.size()) + " " + fill + " " + color_name + " diamonds", Point(5,10), FONT_HERSHEY_COMPLEX_SMALL, 0.5, Scalar(255,0,0), 1);
                }
                else if(convexity/tokens_contours_approx.size() < 0.4){
                    shape_enum = WAVE;
                    //cv::putText(projected_card, to_string(tokens_contours_approx.size()) + " " + fill + " " + color_name  + " waves", Point(5,10), FONT_HERSHEY_COMPLEX_SMALL, 0.5, Scalar(255,0,0), 1);
                }
                else if(convexity/tokens_contours_approx.size() > 0.75 && abs(vertices/tokens_contours_approx.size() - 8) < 1.1){
                    shape_enum = CIRCULAR;
                    //cv::putText(projected_card, to_string(tokens_contours_approx.size()) + " " + fill + " " + color_name  + " capsules", Point(5,10), FONT_HERSHEY_COMPLEX_SMALL, 0.5, Scalar(255,0,0), 1);
                }else{
                    return;
                }

                // Get number
                Number number_enum;
                if(tokens_contours_approx.size() == 1){
                    number_enum = ONE;
                }
                else if(tokens_contours_approx.size() == 2){
                    number_enum = TWO;
                }
                else if(tokens_contours_approx.size() == 3){
                    number_enum = THREE;
                }

                mtx.lock();
                vector<vector<Point>> contour_aux {contours[i]};
                deck.push_back(Card(color_enum, fill_enum, shape_enum, number_enum, contour_aux, true));
                mtx.unlock();


                drawContours(img, contours, i, Scalar(255, 0, 0), 0, 8);


            }   
        });

        sort(deck.begin(), deck.end(), [](const auto& c1, const auto& c2)
            {
                return c1.getColor() < c2.getColor() ||
                        (c1.getColor() == c2.getColor() && c1.getFill() < c2.getFill()) ||
                        (c1.getColor() == c2.getColor() && c1.getFill() == c2.getFill() && c1.getNumber() < c2.getNumber()) ||
                        (c1.getColor() == c2.getColor() && c1.getFill() == c2.getFill() && c1.getNumber() == c2.getNumber() && c1.getShape() < c2.getShape());
            });

        // Get sets
        vector<vector<Card>> sets {get_sets(deck)};

        int color_num = 0;
        for(const auto &set: sets){
            vector<vector<Point>> triangle_set(1);
            for(const auto &card: set){
                card.print();
                triangle_set[0].push_back(mean_point_contour(card.getContour()[0]));
            }
            cout << endl;
            drawContours(img, triangle_set, 0, get_triangle_color(set, triangle_colors), 3, 8);
        }
        for(auto & triangle_color: triangle_colors)
            triangle_color.count = max(triangle_color.count-1, 0);

        // Display outputs
        imshow("Display image output", img);

        char c = (char)waitKey(1);
	    if( c == 27 ) 
	        break;
    }

    cap.release();
	destroyAllWindows();

    return 0;
}

double normalize_0_180(double angle){
    if(angle < -180){
        angle += 360;
    }else if(angle < 0){
        angle += 180;
    }else if(angle > 180){
        angle -= 180;
    }
    return angle;
}

bool check_contour_angle(const vector<Point> & contour, float desired_angle, float error){
    for(int i = 0; i < contour.size(); i++){
        Point P1 = contour[(i+1)%contour.size()];
        Point P2 = contour[i];
        Point P3 = contour[(i+2)%contour.size()];
        double angle = (atan2(P3.y - P1.y, P3.x - P1.x) - atan2(P2.y - P1.y, P2.x - P1.x)) * (180.0/3.141592653589793238463);
        angle = normalize_0_180(angle);
        if(abs(angle - 90) > 45){
            return false;
        }
    }
    return true;
}

void transform_contours(vector<vector<Point>> &contours, const Mat &projective_matrix){
    for(int i = 0; i < contours.size(); i++){
        for(int j = 0; j < contours[i].size(); j++){
            double point_array[3] = {(double)(contours[i][j].x), (double)(contours[i][j].y), 1};
            Mat point_mat = Mat(3, 1, CV_64FC1, point_array);
            Mat xy_points_mat = projective_matrix*point_mat;
            contours[i][j].x = xy_points_mat.at<double>(0, 0);
            contours[i][j].y = xy_points_mat.at<double>(1, 0);
        }
    }
}

Point mean_point_contour(const vector<Point> &contour){
    double x = 0, y = 0;
    for(const Point &point: contour){
        x += point.x;
        y += point.y;
    }
    return Point(x/contour.size(), y/contour.size());
}

Scalar get_triangle_color(const vector<Card> &set, vector<triangle_color> &triangle_colors, int n_max){
    int first_empty = -1;
    for(size_t i = 0; i < triangle_colors.size(); i++){
        if(triangle_colors[i].set.empty() || triangle_colors[i].count == 0){
            if(first_empty == -1){
                first_empty = i;
            }
            continue;
        }

        if(set == triangle_colors[i].set){
            triangle_colors[i].count = n_max;
            return triangle_colors[i].color;
        }
    }
    if(first_empty == -1)
        first_empty = 0;
    triangle_colors[first_empty].set = set;
    triangle_colors[first_empty].count = n_max;
    return triangle_colors[first_empty].color;
}