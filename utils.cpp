#include "utils.hpp"

#include <iostream>
#include <string>
#include <dirent.h>

using namespace std;

Mat get_outer_contours(Mat img, vector<vector<Point>> &contours, Mat &th_mask, bool approx = false, double ratio_epsilon = 0.05){
    
    Mat gray_img;
    cvtColor(img, gray_img, COLOR_BGR2GRAY);
    normalize(gray_img, gray_img, 0, 255, cv::NORM_MINMAX);
    //threshold(gray_img, gray_img, 180, 255, THRESH_BINARY);
    adaptiveThreshold(gray_img, gray_img, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 181, 10);
    th_mask = gray_img;

    Mat dst(gray_img.rows, gray_img.cols, CV_8UC1, Scalar::all(0));
    vector<vector<Point>> contours_all;
    vector<Vec4i> hierarchy;
    findContours(gray_img, contours_all, hierarchy, RETR_TREE, CHAIN_APPROX_TC89_L1);

    for( int i = 0; i < contours_all.size(); i++ ){
        //cout << hierarchy[i];
        Scalar color(255, 255, 255);
        if(hierarchy[i][3] == -1){
            int j = hierarchy[i][2];
            while(j != -1){
                double len = arcLength(contours_all[j], true);
                if(approx) approxPolyDP( contours_all[j], contours_all[j], ratio_epsilon*len, true);
                if(contourArea(contours_all[j]) > 5000){
                    drawContours(dst, contours_all, j, color, -1, 8);
                    contours.push_back(contours_all[j]);
                }
                j = hierarchy[j][0];
            }
        }
    }
    return dst;
}

void create_mask(string path_img, string base_path, string dst_path_template, string dst_path_mask){

    // Get image
    Mat img = imread(base_path + path_img, IMREAD_COLOR);
    if(img.empty()){
        cout << "Could not read the image: " << path_img << endl;
        return;
    }
    rotate(img, img, ROTATE_90_CLOCKWISE);

    vector<vector<Point>> contours;
    Mat random_mat;
    Mat dst = get_outer_contours(img, contours, random_mat);

    Mat cropped_img, cropped_mask, gray_dst;
    gray_dst = dst > 128;
    /**
    Rect crop_region(30, 150, 240, 140);
    cropped_mask = gray_dst(crop_region);
    cropped_img = img(crop_region);
    */
    //cropped_mask = gray_dst;
    //cropped_img = img;

    imshow("Mask", gray_dst);
    imshow("Image", img);

    imwrite(dst_path_template + path_img, img);
    imwrite(dst_path_mask + path_img, gray_dst);
    cout << dst_path_template + path_img << endl;
    return;
}

void create_masks(){

    
    DIR *dpdf;
    struct dirent *epdf;

    string base_dir = "opencv_test/data/labeled/";
    dpdf = opendir("opencv_test/data/labeled/");
    if (dpdf != NULL){
        while (epdf = readdir(dpdf)){
            create_mask(epdf->d_name, base_dir, "opencv_test/data/templates/imgs/", "opencv_test/data/templates/masks/");
            //waitKey(0);
        }
    }
    closedir(dpdf);

}

bool check_set(const Card &card1, const Card &card2, const Card &card3){

    if(!((card1.getColor() == card2.getColor() && card2.getColor() == card3.getColor() && card1.getColor() == card3.getColor()) || 
    (card1.getColor() != card2.getColor() && card2.getColor() != card3.getColor() && card1.getColor() != card3.getColor()))){
        return false;
    }

    if(!((card1.getFill() == card2.getFill() && card2.getFill() == card3.getFill() && card1.getFill() == card3.getFill()) || 
    (card1.getFill() != card2.getFill() && card2.getFill() != card3.getFill() && card1.getFill() != card3.getFill()))){
        return false;
    }

    if(!((card1.getNumber() == card2.getNumber() && card2.getNumber() == card3.getNumber() && card1.getNumber() == card3.getNumber()) || 
    (card1.getNumber() != card2.getNumber() && card2.getNumber() != card3.getNumber() && card1.getNumber() != card3.getNumber()))){
        return false;
    }

    if(!((card1.getShape() == card2.getShape() && card2.getShape() == card3.getShape() && card1.getShape() == card3.getShape()) || 
    (card1.getShape() != card2.getShape() && card2.getShape() != card3.getShape() && card1.getShape() != card3.getShape()))){
        return false;
    }

    return true;
}

vector<vector<Card>> get_sets(const vector<Card> &deck){
    vector<vector<Card>> sets;

    if(deck.size() >= 3){
        for(int i = 0; i < deck.size() - 2; i++){
            for(int j = i + 1; j < deck.size() - 1; j++){
                for(int k = j + 1; k < deck.size(); k++){
                    if(check_set(deck[i], deck[j], deck[k])){
                        vector<Card> set {deck[i], deck[j], deck[k]};
                        sets.push_back(set);
                    }
                }
            }
        }
    }

    return sets;
}