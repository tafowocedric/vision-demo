//
// Voyance Vision
//
// Voyance Vision C++ Library
// ImagePreprocessing.hpp
//

#ifndef ImagePreprocessing_hpp
#define ImagePreprocessing_hpp

#include <opencv.hpp>
#include "utils.hpp"

class ImagePreprocessing
{
public:
    /**
     * @brief Construct a new Image Preprocessing object
     *
     * @param imagePath image absolute path
     */
    ImagePreprocessing(std::string imagePath);

    /**
     * @brief covert image to grayscale (2D)
     *
     * @param src input 1-, 3-, or 4-channel image; when ksize is 3 or 5, the image depth should be CV_8U, CV_16U, or CV_32F, for larger aperture sizes, it can only be CV_8U.
     * @param dst destination array of the same size and type as src.
     */
    static void covertToGray(cv::Mat src, cv::Mat *dst);

    /**
     * @brief load image from abs path
     *
     * @param imagePath image absolute location
     * @return cv::Mat image
     */
    static cv::Mat readImage(std::string imagePath);

    /**
     * @brief save image to abs path
     *
     * @param imagePath image absolute destination
     * @param image image data array
     */
    static void writeImage(std::string imagePath, cv::Mat image);

    /**
     * @brief Use bbox annotation and draw rectangle on our point of interest
     *
     * @param image Mat image
     * @param bbox json bbox array
     */
    void drawBboxRect(cv::Mat image, json bbox);

    /**
     * @brief extract our point of interest as the main image
     *
     * @param src input 1-, 3-, or 4-channel image; when ksize is 3 or 5, the image depth should be CV_8U, CV_16U, or CV_32F, for larger aperture sizes, it can only be CV_8U.
     * @param dst destination array of the same size and type as src.
     * @param region Rect region of interest extracted bbox annotation
     */
    static void cropImage(cv::Mat src, cv::Mat *dst, cv::Rect region);

    /**
     * @brief Adjust image text to 90 degree
     *
     * @param src input 1-, 3-, or 4-channel image; when ksize is 3 or 5, the image depth should be CV_8U, CV_16U, or CV_32F, for larger aperture sizes, it can only be CV_8U.
     * @param dst destination array of the same size and type as src.
     * @param thetaRad threshold radius angle to skew the image
     */
    void deskewImage(cv::Mat src, cv::Mat *dst, double thetaRad = 0.0);

    /**
     * @brief compute adaptive threshold
     *
     * @param src input 1-, 3-, or 4-channel image; when ksize is 3 or 5, the image depth should be CV_8U, CV_16U, or CV_32F, for larger aperture sizes, it can only be CV_8U.
     * @param dst destination array of the same size and type as src.
     * @param ksizeBlur blur threshold
     * @param maxValue maximum image value (255)
     * @param adaptiveMethod adaptive method type
     * @param thresholdType binary threshold inv type
     * @param blockSize
     * @param C constant
     */
    void adpativeThreshold(cv::Mat src, cv::Mat *dst, uint8_t ksizeBlur, uint8_t maxValue, cv::AdaptiveThresholdTypes adaptiveMethod, uint8_t thresholdType, size_t blockSize, size_t C);

    /**
     * @brief remove noise from image
     *
     * @param src input 1-, 3-, or 4-channel image; when ksize is 3 or 5, the image depth should be CV_8U, CV_16U, or CV_32F, for larger aperture sizes, it can only be CV_8U.
     * @param dst destination array of the same size and type as src.
     * @param templateWindowSize window size
     * @param searchWindowSize
     * @param h
     */
    void noiseReducer(cv::Mat src, cv::Mat *dst, uint8_t templateWindowSize, uint8_t searchWindowSize, uint8_t h);

    /**
     * @brief image processor
     *
     * @param imagePath image absolute destination
     * @param bbox json bbox array
     */
    static void processor(std::string imagePath, json bbox);

    /**
     * @brief image resize preprocessor pipeline and save to a new annotation file
     *
     * @param dataDir data directory
     * @param annotations absolute path to annotation file
     * @param size resize image size
     * @param dst destination root folder
     * @param dstAnnotations new annotation json file
     */
    static void imageResizePipeline(std::string dataDir, std::string annotations, uint size, std::string dst, std::string dstAnnotations);

    /**
     * @brief Get the Image object
     *
     * @return cv::Mat
     */
    cv::Mat getImage();

    /**
     * @brief Get the Crop Region object
     *
     * @return cv::Rect
     */
    cv::Rect getCropRegion();

private:
    cv::Mat image;
    cv::Rect cropRegion;

    /**
     * @brief extact and compute draw points and crop region from bbox annotation
     *
     * @param bbox bbox json annotation array
     * @param start draw starting point ptr
     * @param end end draw point ptr
     */
    void extractBbox(json bbox, cv::Point *start, cv::Point *end);

    /**
     * @brief feature processing before hough transform
     *
     * @param src input 1-, 3-, or 4-channel image; when ksize is 3 or 5, the image depth should be CV_8U, CV_16U, or CV_32F, for larger aperture sizes, it can only be CV_8U.
     * @param dst destination array of the same size and type as src.
     */
    void processes(cv::Mat src, cv::Mat *dst);

    /**
     * @brief hough tranform algorithm
     *
     * @param image input 1-, 3-, or 4-channel image; when ksize is 3 or 5, the image depth should be CV_8U, CV_16U, or CV_32F, for larger aperture sizes, it can only be CV_8U.
     * @param skew skew angle
     */
    void houghTransform(cv::Mat image, double *skew);
};

#endif // ImagePreprocessing_hpp