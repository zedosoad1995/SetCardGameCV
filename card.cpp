#include "card.hpp"

#include <iostream>

using namespace std;

Card::Card(Color col, Fill fil, Shape sh, Number n, vector<vector<Point>>  box, bool get_img, string base_path):
color {col}, fill {fil}, shape {sh}, number {n}, card_bounding_box {box}
{

    string color_name, number_name, fill_name, shape_name;

    if(color == RED) color_name = "red";
    else if(color == GREEN) color_name = "green";
    else if(color == PURPLE) color_name = "purple";

    if(number == ONE) number_name = "single";
    else if(number == TWO) number_name = "double";
    else if(number == THREE) number_name = "triple";

    if(fill == SOLID) fill_name = "solid";
    else if(fill == OUTLINE) fill_name = "outline";
    else if(fill == STRIPES) fill_name = "stripes";

    if(shape == CIRCULAR) shape_name = "capsule";
    else if(shape == WAVE) shape_name = "squiggle";
    else if(shape == DIAMOND) shape_name = "diamond";

    if(get_img){
        string img_name = color_name + "-" + number_name + "-" + fill_name + "-" + shape_name + ".jpg";

        // Get image
        Mat img = imread(base_path + img_name, IMREAD_COLOR);
        if(img.empty()){
            cout << "Could not read the image " << img_name << endl;
            return;
        }
        rotate(img, img, ROTATE_90_CLOCKWISE);
        template_img = img;
    }

}

void Card::print() const{

    string color_name, fill_name, shape_name, number_name;

    if(color == RED) color_name = "red";
    else if(color == GREEN) color_name = "green";
    else if(color == PURPLE) color_name = "purple";

    if(number == ONE) number_name = "one";
    else if(number == TWO) number_name = "two";
    else if(number == THREE) number_name = "three";

    if(fill == SOLID) fill_name = "solid";
    else if(fill == OUTLINE) fill_name = "outline";
    else if(fill == STRIPES) fill_name = "stripes";

    if(shape == CIRCULAR) shape_name = "circular";
    else if(shape == WAVE) shape_name = "wave";
    else if(shape == DIAMOND) shape_name = "diamond";

    cout << color_name << " " << number_name << " " << fill_name << " " << shape_name << endl;
}

bool Card::operator==(const Card& card) const{
    return card.color == color && card.fill == fill && card.number == number && card.shape == shape;
}
