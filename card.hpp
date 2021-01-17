#ifndef _CARD_H
#define _CARD_H

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <string>

using namespace cv;
using namespace std;

enum Color {RED, GREEN, PURPLE};
enum Fill {SOLID, OUTLINE, STRIPES};
enum Shape {DIAMOND, WAVE, CIRCULAR};
enum Number {ONE, TWO, THREE};

class Card{
private:
    Mat template_img;
    Color color;
    Fill fill;
    Shape shape;
    Number number;
    vector<vector<Point>> card_bounding_box;
public:
    Card(Color col, Fill fil, Shape sh, Number n, vector<vector<Point>> box = vector<vector<Point>>(), bool get_img = false, string base_path = "opencv_test/data/templates/imgs/");
    Mat getTemplate()const{return this->template_img;}
    Color getColor()const{return this->color;}
    Fill getFill()const{return this->fill;}
    Shape getShape()const{return this->shape;}
    Number getNumber()const{return this->number;}
    vector<vector<Point>> getContour()const{return this->card_bounding_box;}
    void print()const;
    bool operator==(const Card& other) const;
};

#endif