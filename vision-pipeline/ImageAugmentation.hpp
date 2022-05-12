//
// Voyance Vision
//
// Voyance Vision C++ Library
// ImageAugmentation.hpp
//
// Functions in this section borrowed and modified from https://jovian.ai/aakanksha-ns/road-signs-bounding-box-prediction/v/2#C20

#ifndef ImageAugmentation_hpp
#define ImageAugmentation_hpp

#include "ImagePreprocessing.hpp"
#include "Interface.hpp"

#include <random>

#include <filesystem>
namespace fs = std::filesystem;

class ImageAugmentation
{
private:
    const uint r_pix = 8;

public:
    /**
     * This class handles image Augmentation
     *
     * Constructor
     * @param imagePath image absolute path
     */
    ImageAugmentation(){};

    /**
     * Dealloctor
     *
     */
    ~ImageAugmentation(){};

    /**
     * cv::Mat Method (centerCrop)
     * crop image from center point
     *
     * @param src input 1-, 3-, or 4-channel image; when ksize is 3 or 5, the image depth should be CV_8U, CV_16U, or CV_32F, for larger aperture sizes, it can only be CV_8U.
     * @return array of the same size and type as src.
     *
     */
    cv::Mat centerCrop(cv::Mat src);

    /**
     * cv::Mat Method (rotate_cv)
     * rotate image
     *
     * @param src input 1-, 3-, or 4-channel image; when ksize is 3 or 5, the image depth should be CV_8U, CV_16U, or CV_32F, for larger aperture sizes, it can only be CV_8U.
     * @param rdeg rotation degree.
     * @param y fill destination image pixels, if outliers set to zero (default=false)
     * @param mode border reflection (default=cv::BORDER_REFLECT)
     * @param interpolation resampling using pixel area relation (default=cv::INTER_AREA)
     *
     * @return array of the same size and type as src.
     */
    cv::Mat rotate_cv(cv::Mat image, double rdeg, bool y = false, uint mode = cv::BORDER_REFLECT, uint interpolation = cv::INTER_AREA);

    /**
     * void Method (randomCropXY)
     * random crop image
     *
     * @param image input 1-, 3-, or 4-channel image; when ksize is 3 or 5, the image depth should be CV_8U, CV_16U, or CV_32F, for larger aperture sizes, it can only be CV_8U.
     * @param mask input 1-, 3-, or 4-channel image; when ksize is 3 or 5, the image depth should be CV_8U, CV_16U, or CV_32F, for larger aperture sizes, it can only be CV_8U.
     *
     * @return (image mask)
     */
    Data randomCropXY(Data data);

    /**
     * static json Method (transformsXY)
     * random crop image
     *
     * @param image input 1-, 3-, or 4-channel image; when ksize is 3 or 5, the image depth should be CV_8U, CV_16U, or CV_32F, for larger aperture sizes, it can only be CV_8U.
     * @param bbox json bbox array
     * @param transforms tranform image (rotate, flip image)
     *
     * @return Mat image and json object of new bounding box
     *
     */
    static BboxData transformsXY(BboxData data, bool transforms);

    /**
     * static cv::Mat Method (createMask)
     * create a masking layer from image using bbox
     *
     * @param image input 1-, 3-, or 4-channel image; when ksize is 3 or 5, the image depth should be CV_8U, CV_16U, or CV_32F, for larger aperture sizes, it can only be CV_8U.
     * @param bbox json bbox array
     *
     * @return array of the same size and type as src.
     *
     */
    static cv::Mat createMask(json bbox, cv::Mat image);

    /**
     * static json Method (maskToBbox)
     * create a bbox from mask
     *
     * @param mask input 1-, 3-, or 4-channel image; when ksize is 3 or 5, the image depth should be CV_8U, CV_16U, or CV_32F, for larger aperture sizes, it can only be CV_8U.
     *
     * @return json object of new bounding box
     *
     */
    static json maskToBbox(cv::Mat mask);

    /**
     * static json Method (resizeImageBbox)
     * random crop image
     *
     * @param imagePath absolute path to image file.
     * @param outputPath absolute path to output folder.
     * @param bbox json bbox array
     * @param size resized image size
     * @param *newImagePath resized image absolute path
     *
     * @return json object of new bounding box
     *
     */
    static json resizeImageBbox(std::string imagePath, std::string outputPath, json bbox, uint32_t size, std::string *newImagePath);
};

#endif // ImageAugmentation_hpp
