//
// Voyance Vision
//
// Voyance Vision C++ Library
// ImagePreprocessing.cpp
//

#include "ImagePreprocessing.hpp"

ImagePreprocessing::ImagePreprocessing(std::string imagePath)
{
    // read image
    this->image = readImage(imagePath);
};

cv::Mat ImagePreprocessing::readImage(std::string imagePath)
{
    return cv::imread(imagePath, cv::IMREAD_COLOR);
};

void ImagePreprocessing::writeImage(std::string imagePath, cv::Mat image)
{
    cv::imwrite(imagePath, image);
};

void ImagePreprocessing::covertToGray(cv::Mat src, cv::Mat *dst)
{
    cv::cvtColor(src, *dst, cv::COLOR_BGR2GRAY);
};

void ImagePreprocessing::extractBbox(json bbox, cv::Point *start, cv::Point *stop)
{
    *start = cv::Point((int)bbox[0], (int)bbox[1]);
    *stop = cv::Point((int)bbox[0] + (int)bbox[2], (int)bbox[1] + (int)bbox[3]);

    cv::Rect crop_region((int)bbox[0], (int)bbox[1], (int)bbox[2], (int)bbox[3]);
    this->cropRegion = crop_region;
};

void ImagePreprocessing::drawBboxRect(cv::Mat image, json bbox)
{
    cv::Point start, stop;
    cv::Rect cropRegion;

    // compute start and stop
    this->extractBbox(bbox, &start, &stop);
    cv::rectangle(image, start, stop, cv::Scalar(0, 0, 255), 3);
};

void ImagePreprocessing::cropImage(cv::Mat src, cv::Mat *dst, cv::Rect region)
{
    *dst = src(region);
};

void ImagePreprocessing::processes(cv::Mat src, cv::Mat *dst)
{
    // https://stackoverflow.com/questions/45913057/deskewing-image-opencv
    // 1) assume white on black and does local thresholding
    // 2) only allow voting top is white and buttom is black(buttom text line)
    cv::Mat thresh;
    thresh = 255 - src;

    // thresh = src.clone();
    adaptiveThreshold(thresh, thresh, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 15, -2);

    cv::Mat wb = cv::Mat::zeros(src.size(), CV_8UC1);

    for (int x = 1; x < thresh.cols - 1; x++)
    {
        for (int y = 1; y < thresh.rows - 1; y++)
        {
            bool toprowblack = thresh.at<uchar>(y - 1, x) == 0 || thresh.at<uchar>(y - 1, x - 1) == 0 || thresh.at<uchar>(y - 1, x + 1) == 0;
            bool belowrowblack = thresh.at<uchar>(y + 1, x) == 0 || thresh.at<uchar>(y + 1, x - 1) == 0 || thresh.at<uchar>(y + 1, x + 1) == 0;

            uchar pix = thresh.at<uchar>(y, x);
            if ((!toprowblack && pix == 255 && belowrowblack))
            {
                wb.at<uchar>(y, x) = 255;
            }
        }
    }

    *dst = wb;
};

void ImagePreprocessing::houghTransform(cv::Mat image, double *skew)
{
    // https://stackoverflow.com/questions/45913057/deskewing-image-opencv

    // compute processed image
    cv::Mat processedImage;
    processes(image, &processedImage);

    double max_r = sqrt(pow(.5 * processedImage.cols, 2) + pow(.5 * processedImage.rows, 2));
    int angleBins = 180;

    cv::Mat acc = cv::Mat::zeros(cv::Size(2 * max_r, angleBins), CV_32SC1);
    int cenx = processedImage.cols / 2;
    int ceny = processedImage.rows / 2;

    for (int x = 1; x < processedImage.cols - 1; x++)
    {
        for (int y = 1; y < processedImage.rows - 1; y++)
        {
            if (processedImage.at<uchar>(y, x) == 255)
            {
                for (int t = 0; t < angleBins; t++)
                {
                    double r = (x - cenx) * cos((double)t / angleBins * CV_PI) + (y - ceny) * sin((double)t / angleBins * CV_PI);
                    r += max_r;
                    acc.at<int>(t, int(r))++;
                }
            }
        }
    }

    cv::Mat thresh;
    cv::normalize(acc, acc, 255, 0, cv::NORM_MINMAX);
    cv::convertScaleAbs(acc, acc);

    /** debug
    cv::Mat cmap;
    applyColorMap(acc, cmap, COLORMAP_JET);
    imshow("cmap", cmap);
    imshow("acc", acc); */

    cv::Point maxLoc;
    cv::minMaxLoc(acc, 0, 0, 0, &maxLoc);

    double theta = (double)maxLoc.y / angleBins * CV_PI;
    double rho = maxLoc.x - max_r;

    if (abs(sin(theta)) < 0.000001) // check vertical
    {
        // when vertical, line equation becomes
        // x = rho
        double m = -cos(theta) / sin(theta);

        cv::Point2d p1 = cv::Point2d(rho + processedImage.cols / 2, 0);
        cv::Point2d p2 = cv::Point2d(rho + processedImage.cols / 2, processedImage.rows);

        line(image, p1, p2, cv::Scalar(0, 0, 255), 1);
        *skew = 90;
    }
    else
    {
        // convert normal form back to slope intercept form
        // y = mx + b
        double m = -cos(theta) / sin(theta);
        double b = rho / sin(theta) + processedImage.rows / 2. - m * processedImage.cols / 2.;

        cv::Point2d p1 = cv::Point2d(0, b);
        cv::Point2d p2 = cv::Point2d(processedImage.cols, processedImage.cols * m + b);

        line(image, p1, p2, cv::Scalar(0, 0, 255), 1);

        double skewangle;
        skewangle = p1.x - p2.x > 0 ? (atan2(p1.y - p2.y, p1.x - p2.x) * 180. / CV_PI) : (atan2(p2.y - p1.y, p2.x - p1.x) * 180. / CV_PI);
        *skew = skewangle;
    }
};

void ImagePreprocessing::deskewImage(cv::Mat src, cv::Mat *dst, double thetaRad)
{
    // https://stackoverflow.com/questions/45913057/deskewing-image-opencv
    if (thetaRad == 0.0)
        houghTransform(src, &thetaRad);

    cv::Mat rotated;

    double rskew = thetaRad * CV_PI / 180;
    double nw = abs(sin(thetaRad)) * src.rows + abs(cos(thetaRad)) * src.cols;
    double nh = abs(cos(thetaRad)) * src.rows + abs(sin(thetaRad)) * src.cols;

    cv::Mat rot_mat = cv::getRotationMatrix2D(cv::Point2d(nw * .5, nh * .5), thetaRad * 180 / CV_PI, 1);

    cv::Mat pos = cv::Mat::zeros(cv::Size(1, 3), CV_64FC1);
    pos.at<double>(0) = (nw - src.cols) * .5;
    pos.at<double>(1) = (nh - src.rows) * .5;

    cv::Mat res = rot_mat * pos;
    rot_mat.at<double>(0, 2) += res.at<double>(0);
    rot_mat.at<double>(1, 2) += res.at<double>(1);

    cv::warpAffine(src, rotated, rot_mat, cv::Size(nw, nh), cv::INTER_LANCZOS4);
    *dst = rotated;
};

void ImagePreprocessing::adpativeThreshold(cv::Mat src, cv::Mat *dst, uint8_t ksizeBlur, uint8_t maxValue, cv::AdaptiveThresholdTypes adaptiveMethod, uint8_t thresholdType, size_t blockSize, size_t C)
{
    cv::Mat threshold_image, blur;

    medianBlur(src, blur, ksizeBlur);
    adaptiveThreshold(blur, *dst, maxValue, adaptiveMethod, thresholdType, blockSize, C);
};

void ImagePreprocessing::noiseReducer(cv::Mat src, cv::Mat *dst, uint8_t templateWindowSize, uint8_t searchWindowSize, uint8_t h)
{
    cv::fastNlMeansDenoising(src, *dst, templateWindowSize, searchWindowSize, h);
    cv::threshold(src, *dst, 150, 255, cv::THRESH_BINARY);
};

void ImagePreprocessing::processor(std::string imagePath, json bbox)
{
    ImagePreprocessing imageProcessor = ImagePreprocessing(imagePath);

    // Read image as grayscale
    cv::Mat image = imageProcessor.getImage();

    if (!image.data)
    {
        std::cout << "Could not open or find the image" << std::endl;
        return;
    };
    cv::imshow("original", image);

    // IMAGE PROCESSING

    imageProcessor.covertToGray(image, &image);

    // Invert image
    bitwise_not(image, image);

    // draw bbox on image
    imageProcessor.drawBboxRect(image, bbox);

    // crop point of interest using bbox
    imageProcessor.cropImage(image, &image, imageProcessor.getCropRegion());

    // Deskew image
    imageProcessor.deskewImage(image, &image);

    // Applying Adaptive Threshold with kernel :- 21 X 21
    imageProcessor.adpativeThreshold(image, &image, 1, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV, 3, 4);

    // Noise Removal
    imageProcessor.noiseReducer(image, &image, 30, 7, 21);

    cv::imshow("no noise", image);
    cv::waitKey(0);
};

// Getters
cv::Mat ImagePreprocessing::getImage()
{
    return this->image;
};

cv::Rect ImagePreprocessing::getCropRegion()
{
    return this->cropRegion;
};