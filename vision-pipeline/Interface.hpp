#pragma once

#include <opencv.hpp>
#include <json.hpp>

using json = nlohmann::json;

struct Data
{
    Data(cv::Mat img, cv::Mat _mask) : image(img), mask(_mask){};
    ~Data(){
        // image.release();
        // mask.release();
    };
    cv::Mat image;
    cv::Mat mask;
};

struct BboxData
{
    BboxData(cv::Mat img, json _bbox) : image(img), bbox(_bbox){};
    ~BboxData(){
        // image.release();
        // bbox.erase(bbox.begin(), bbox.end());
    };
    cv::Mat image;
    json bbox;
};