//
// Voyance Vision
//
// Voyance Vision C++ Library
// ImageAugmentation.cpp
//

#include "ImageAugmentation.hpp"

template <typename T>
T RandomNum(T _min, T _max)
{
    T temp;
    if (_min > _max)
    {
        temp = _max;
        _max = _min;
        _min = temp;
    }
    return rand() / (double)RAND_MAX * (_max - _min) + _min;
}

cv::Mat ImageAugmentation::centerCrop(cv::Mat src)
{
    uint width = src.size[0];
    uint height = src.size[1];

    int c_pix = std::round(r_pix * height / width);
    cv::Rect region(r_pix, c_pix, width - 2 * c_pix, height - 2 * c_pix);

    cv::Mat dst;
    ImagePreprocessing::cropImage(src, &dst, region);

    return dst;
};

cv::Mat ImageAugmentation::rotate_cv(cv::Mat image, double rdeg, bool y, uint mode, uint interpolation)
{
    uint width = image.size[0];
    uint height = image.size[1];

    cv::Mat M = cv::getRotationMatrix2D(cv::Point2d(height / 2, width / 2), rdeg, 1);

    cv::Mat dst;
    if (y)
        cv::warpAffine(image, dst, M, cv::Size(height, width), 1, cv::BORDER_CONSTANT);
    else
        cv::warpAffine(image, dst, M, cv::Size(height, width), cv::WARP_FILL_OUTLIERS + interpolation, mode);

    return dst;
};

Data ImageAugmentation::randomCropXY(Data data)
{
    uint width = data.image.size[0];
    uint height = data.image.size[1];

    uint c_pix = std::round(r_pix * height / width);

    double rand_r = RandomNum<float>(0, 1);
    double rand_c = RandomNum<float>(0, 1);

    int start_r = std::floor(2 * rand_r * r_pix);
    int start_c = std::floor(2 * rand_c * c_pix);

    cv::Rect region(start_c, start_r, width - 2 * c_pix, height - 2 * c_pix);

    cv::Mat cropImage, cropMask;
    ImagePreprocessing::cropImage(data.image, &cropImage, region);
    ImagePreprocessing::cropImage(data.mask, &cropMask, region);

    return data;
};

BboxData ImageAugmentation::transformsXY(BboxData data, bool transforms)
{
    // smart pointer
    std::unique_ptr<ImageAugmentation> imageAugmentation = std::make_unique<ImageAugmentation>();

    // Normalize image (0 - 1)
    cv::cvtColor(data.image, data.image, cv::COLOR_BGR2RGB);
    cv::Mat channels[3], normalize_rgb;
    cv::split(data.image, channels);

    for (int i = 0; i < data.image.size().height; i++)
    {
        for (int j = 0; j < data.image.size().width; j++)
        {
            int b = (int)(data.image).at<cv::Vec3b>(i, j)[0];
            int g = (int)(data.image).at<cv::Vec3b>(i, j)[1];
            int r = (int)(data.image).at<cv::Vec3b>(i, j)[2];
            double sum = b + g + r;
            double bnorm = double(b) / sum * 255;
            double gnorm = double(g) / sum * 255;
            double rnorm = double(r) / sum * 255;
            channels[0].at<uchar>(i, j) = bnorm;
            channels[1].at<uchar>(i, j) = gnorm;
            channels[2].at<uchar>(i, j) = rnorm;
        }
    }

    cv::merge(channels, 3, normalize_rgb);
    data.image = normalize_rgb.clone();
    //  End of normalizer

    cv::Mat mask = imageAugmentation->createMask(data.bbox, data.image);

    // return data constructor
    Data *m_data = new Data(data.image, mask);
    if (transforms)
    {
        double rdeg = (RandomNum<float>(0, 1) - 0.50) * 20;
        cv::Mat r_src = imageAugmentation->rotate_cv(data.image, rdeg);
        cv::Mat r_mask = imageAugmentation->rotate_cv(mask, rdeg);

        // Horizontal Flip
        if (RandomNum<float>(0, 1) > 0.5)
        {
            cv::Mat fr_src = cv::Mat(r_src.rows, r_src.cols, CV_8UC3);
            cv::Mat fr_mask = cv::Mat(r_mask.rows, r_mask.cols, CV_8UC3);

            cv::flip(r_src, fr_src, 1);
            cv::flip(r_mask, fr_mask, 1);

            r_src = fr_src;
            r_mask = fr_mask;
        }

        // Vertical Flip
        if (RandomNum<float>(0, 1) > 0.5)
        {
            cv::Mat fr_src = cv::Mat(r_src.rows, r_src.cols, CV_8UC3);
            cv::Mat fr_mask = cv::Mat(r_mask.rows, r_mask.cols, CV_8UC3);

            cv::flip(r_src, fr_src, 0);
            cv::flip(r_mask, fr_mask, 0);

            r_src = fr_src;
            r_mask = fr_mask;
        }

        *m_data = imageAugmentation->randomCropXY(Data(r_src, r_mask));
    }
    else
    {
        *m_data = Data(imageAugmentation->centerCrop(data.image), imageAugmentation->centerCrop(mask));
    }

    BboxData n_data(m_data->image, imageAugmentation->maskToBbox(m_data->mask));

    // free memory
    data.image.release();
    data.bbox.clear();

    delete m_data;
    return n_data;
}

cv::Mat ImageAugmentation::createMask(json bbox, cv::Mat image)
{
    // Creates a mask for the bounding box of same shape as image
    cv::Mat mask = cv::Mat(image.rows, image.cols, CV_8UC1, cv::Scalar(0));
    std::vector<cv::Point> roi_vertices = {cv::Point((int)bbox[0], (int)bbox[1]),
                                           cv::Point((int)bbox[0] + (int)bbox[2], (int)bbox[1] + (int)bbox[3])};

    std::vector<cv::Point> roi_poly;
    cv::approxPolyDP(roi_vertices, roi_poly, 1.0, true);

    cv::rectangle(mask, roi_poly[0], roi_poly[1], cv::Scalar(255, 0, 255), CV_FILLED, 0);
    return mask;
};

json ImageAugmentation::maskToBbox(cv::Mat mask)
{
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    cv::findContours(mask, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_TC89_KCOS);
    static json bbox;
    bbox.clear();

    cv::Rect brect = cv::boundingRect(contours[contours.size() - 1]);
    bbox.push_back({brect.x, brect.y, brect.width, brect.height});

    return bbox[0];
};

json ImageAugmentation::resizeImageBbox(std::string imagePath, std::string outputPath, json bbox, uint32_t size, std::string *newImagePath)
{
    // // if file exists return
    // TODO: HANDLE RESIZE IMAGE AND ANNOTATION EXIST

    // Resize an image and its bounding box and write image to new path
    cv::Mat image = ImagePreprocessing::readImage(imagePath);

    // cv::Size resize((int)(1.49 * size), size);
    cv::Size resize(size, size);
    cv::Mat resizeImage, resizeBbox;

    cv::resize(image, resizeImage, resize);
    cv::resize(createMask(bbox, image), resizeBbox, resize);

    // save resized image to resized-folder
    // create directory if not exist
    if (!fs::is_directory(outputPath) || !fs::exists(outputPath))
        fs::create_directories(outputPath);

    *newImagePath = outputPath + imagePath.substr(imagePath.rfind("/"));
    ImagePreprocessing::writeImage(*newImagePath, resizeImage);

    return maskToBbox(resizeBbox);
}

void ImagePreprocessing::imageResizePipeline(std::string dataDir, std::string annotations, uint size, std::string dst, std::string dstAnnotations)
{
    // Resizing images and bounding boxes
    json annotationData = json_file_reader(annotations);
    ImageAugmentation imageAugmentation = ImageAugmentation();

    json resizedAnn;
    for (auto &annotation : annotationData["annotations"])
    {
        for (auto &image : annotationData["images"])
        {
            if (image["id"] == annotation["image_id"])
            {
                fs::path imageAbsPath = (fs::path)dataDir / (fs::path)image["file_name"].get<std::string>();
                std::string newImagePath;

                try
                {
                    json newBbox = imageAugmentation.resizeImageBbox(imageAbsPath, dst, annotation["bbox"], size, &newImagePath);

                    annotation["bbox"] = newBbox.size() < 2 ? newBbox[0] : newBbox;
                    image["file_name"] = newImagePath;
                    image["height"] = size;
                    image["width"] = size;

                    resizedAnn["annotations"].push_back(annotation);
                    resizedAnn["categories"] = annotationData["categories"];
                    resizedAnn["images"].push_back(image);

                    // clear variable
                    newBbox.clear();
                }
                catch (const std::exception &e)
                {
                    std::cerr << imageAbsPath << std::endl;
                    std::cerr << e.what() << '\n';
                }
            }
        }
    };

    // save new resize json image and bbox to file
    save_to_file(dstAnnotations, resizedAnn);
    auto *_ = split_annotation(dstAnnotations, true);
}
