#include <iostream>
#include <string>
#include <cstring>
#include <iterator>
#include <set>
#include <ctime>

#include "vision-model/Detector.hpp"
#include "vision-model/criterion.hpp"
#include "vision-pipeline/ImageAugmentation.hpp"

class VoyanceCOCODataset : public torch::data::datasets::Dataset<VoyanceCOCODataset>
{
public:
    explicit VoyanceCOCODataset(std::string data_path_, std::string split_, std::set<std::string> CLASSES_) : ann_(json_file_reader(data_path_)),
                                                                                                              classes(CLASSES_),
                                                                                                              split(split_)
    {
        std::transform(split.begin(), split.end(), split.begin(), ::toupper);
        std::set<std::string> phase{"TRAIN", "TEST", "VALIDATE"};
        assert(phase.find(split) != phase.end());

        n_classes = classes.size();
    }
    ~VoyanceCOCODataset() {}

    /**
     * torch::data::Example Method (get)
     * get method to load custom data.
     *
     * @param index iterator value.
     * @return torch Tensor (data, target).
     *
     */
    torch::data::Example<> get(size_t index) override
    {
        size_t imageId = ann_["annotations"][index]["image_id"].get<size_t>();
        json bbox = ann_["annotations"][index]["bbox"].get<json>();
        int cat_id = ann_["annotations"][index]["category_id"].get<int>();
        int difficulties = ann_["annotations"][index]["iscrowd"].get<int>();

        // get image by image id
        std::string imagePath;
        int image_w, image_h;
        for (auto image : ann_["images"].get<json>())
            if (image["id"].get<size_t>() == imageId)
            {
                imagePath = image["file_name"].get<std::string>();
                image_w = image["width"].get<int>();
                image_h = image["height"].get<int>();
            }

        // Mean and standard deviation of ImageNet data that our base VGG from torchvision was trained on
        // see: https://pytorch.org/docs/stable/torchvision/models.html
        std::vector<double> norm_mean = {0.485, 0.456, 0.406};
        std::vector<double> norm_std = {0.229, 0.224, 0.225};

        // Load image with OpenCV.
        cv::Mat image = ImagePreprocessing::readImage(imagePath);

        // OpenCV data manipulation
        bool transform = split == "TRAIN" ? true : false;
        std::shared_ptr<BboxData> data = std::make_shared<BboxData>(ImageAugmentation::transformsXY(BboxData(image, bbox), transform));

        if (data->bbox.empty())
        {
            std::cout << "=> NO BOX: " << imageId << std::endl;
            data = std::make_shared<BboxData>(image, bbox);
        }

        // Convert the image and label to a tensor.
        // Here we need to clone the data, as from_blob does not change the ownership of the underlying memory,
        // which, therefore, still belongs to OpenCV. If we did not clone the data at this point, the memory
        // would be deallocated after leaving the scope of this get method, which results in undefined behavior.
        std::vector<cv::Mat> channels(3);
        cv::split(data->image, channels);

        torch::Tensor R = torch::from_blob(channels[2].ptr(), {image_w, image_h}, torch::kUInt8);
        torch::Tensor G = torch::from_blob(channels[1].ptr(), {image_w, image_h}, torch::kUInt8);
        torch::Tensor B = torch::from_blob(channels[0].ptr(), {image_w, image_h}, torch::kUInt8);

        // create image tensor from channels and normalize data [0 - 1]
        torch::Tensor image_tensor = torch::cat({R, G, B}).view({3, image_w, image_h}).to(torch::kFloat).clone();
        image_tensor = torch::data::transforms::Normalize<>(norm_mean, norm_std)(image_tensor);

        // range-based for bbox
        std::array<int32_t, 4> vec_bbox{{data->bbox[0], data->bbox[1], data->bbox[2], data->bbox[3]}};

        auto option = torch::TensorOptions().dtype(at::kFloat);
        torch::Tensor bbox_tensor = torch::from_blob(vec_bbox.data(), {1, (long)vec_bbox.size()}, option).clone();

        // this also converts absolute boundary coordinates to their fractional form
        torch::Tensor image_dims = torch::tensor({image_w, image_h, image_w, image_h}, at::kFloat).unsqueeze(0);
        bbox_tensor = bbox_tensor / image_dims;

        //
        torch::Tensor cat_tensor = torch::tensor({cat_id, difficulties}, at::kFloat);
        cat_tensor = cat_tensor.unsqueeze(0);

        torch::Tensor target_tensor = torch::cat({bbox_tensor, cat_tensor}, 1);
        return {image_tensor.clone(), target_tensor.clone()};
    }

    /**
     * torch::optional<size_t> Method (size)
     * get data length.
     *
     * @return torch Tensor size_t.
     *
     */
    torch::optional<size_t> size() const override
    {
        std::cout << "Training Size: " << ann_["annotations"].size() << std::endl;
        return ann_["annotations"].size();
    }

private:
    const json ann_;
    const std::set<std::string> classes;
    int n_classes;
    std::string split;
};

void adjust_learning_rate(torch::optim::SGD &optimizer, float scale)
{
    for (auto &param_group : optimizer.param_groups())
    {
        param_group.options().set_lr(param_group.options().get_lr() * scale);
    }

    // printf("DECAYING learning rate.\n The new LR is %f\n", optimizer.param_groups().at(1).options().get_lr());
}

/**
 * @brief One epoch's training.
 *
 * @tparam datatype DataLoader Type
 * @param train_loader DataLoader for training data
 * @param model SSD300 model
 * @param criterion MultiBox loss
 * @param optimizer optimizer
 * @param epoch epoch number
 */
// template <typename datatype>
// void train(datatype &train_loader, SSD300 &model, MultiBoxLoss &criterion, torch::optim::SGD &optimizer, int &epoch, torch::Device &device);

struct AverageMeter
{
    int val, avg, sum, count;
    AverageMeter()
    {
        reset();
    }
    ~AverageMeter() {}

    void reset()
    {
        val = 0;
        avg = 0;
        sum = 0;
        count = 0;
    }

    void update(int val_, int n = 1)
    {
        val = val_;
        sum += val * n;
        count += n;
        avg = sum / count;
    }
};

int main(int argc, char *argv[])
{
    // Data parameters
    size_t image_size = 300;
    std::string root = "/Users/voyance/Documents/projects/voyancehq/visionhq/data";
    std::string data_folder = "/Users/voyance/Documents/projects/voyancehq/visionhq/data/passports";
    std::string annotations = "/Users/voyance/Documents/projects/voyancehq/visionhq/data/annotations.json";
    std::string resized_image_dir = root + "/output/resized-image-" + std::to_string(image_size) + "x" + std::to_string(image_size) + "x3";
    std::string resized_annotations = root + "/output/resized_annotations.json";

    // Model parameters
    // Not too many here since the SSD300 has a very specific structure
    const std::set<std::string> CLASSES = {"mrz"};
    int n_classes = CLASSES.size();

    // device
    torch::Device device = torch::Device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

    // Learning parameters
    std::string checkpoint_path;                    // path to model checkpoint, None if none
    int batch_size = 1;                             // batch size
    int iterations = 120000;                        // number of iterations to train
    int workers = 4;                                // number of workers for loading data in the DataLoader
    int print_freq = 200;                           // print training status every __ batches
    double lr = 1e-3;                               // learning rate
    std::vector<int> decay_lr_at = {80000, 100000}; // decay learning rate after these many iterations
    float decay_lr_to = 0.1F;                       // decay learning rate to this fraction of the existing learning rate
    float momentum = 0.9F;                          // momentum
    double weight_decay = 5e-4;                     // weight decay
    bool grad_clip;                                 // clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32) - you will recognize it by a sorting error in the MuliBox loss calculation
    SSD300 model = SSD300(n_classes, device);       // model

    torch::optim::SGDOptions optimizerOption = torch::optim::SGDOptions(lr * 2).momentum(momentum).weight_decay(weight_decay);
    torch::optim::SGD optimizer = torch::optim::SGD(model->parameters(), optimizerOption);

    // Training.
    int start_epoch, label_map, epoch;
    start_epoch = 0;
    // TODO: load model checkpoint
    // Initialize model or load checkpoint
    // if (checkpoint_path.empty())
    // {
    //     start_epoch = 0;
    //     model = SSD300(n_classes, device);
    //     // Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
    //     torch::OrderedDict<std::string, at::Tensor> model_dict = model->named_parameters();
    //     torch::OrderedDict<std::string, at::Tensor> biases;
    //     torch::OrderedDict<std::string, at::Tensor> not_biases;
    //     // TODO: model optimizer
    //     // for (auto &params : model_dict)
    //     // {
    //     //     // std::cout << params.key() << std::endl;
    //     //     // std::cout << params.value().requires_grad() << std::endl;
    //     //     if (params.value().requires_grad())
    //     //     {
    //     //         if (params.key().ends_with(".bias"))
    //     //             biases.insert(params.key(), params.value());
    //     //         else
    //     //             not_biases.insert(params.key(), params.value());
    //     //     }
    //     // }
    //     // torch::optim::SGDOptions optimizerOption = torch::optim::SGDOptions(lr * 2)
    //     //                                                .momentum(momentum)
    //     //                                                .weight_decay(weight_decay);
    //     // optimizer = &torch::optim::SGD(biases.values(), optimizerOption);
    //     // optimizer->param_groups().push_back(torch::optim::OptimizerParamGroup(not_biases.values()));
    // }
    // else
    // {
    //     // torch::jit::script::Module checkpoint = torch::jit::load(checkpoint_path);
    //     // start_epoch = checkpoint.get_property("epoch");
    //     // printf("\nLoaded checkpoint from epoch %d.\n", start_epoch);
    //     // model = (SSD300)torch::jit::load(checkpoint["model"].get<std::string>());
    //     // torch::load(optimizer, checkpoint["optimizer"].get<std::string>());
    // }

    // Move to default device
    model->to(device);
    MultiBoxLoss criterion = MultiBoxLoss(model->get_priors_cxcy(), device);
    criterion.to(device);

    /**
     * @brief
     *
     */
    // ImagePreprocessing::imageResizePipeline(data_folder, annotations, image_size, resized_image_dir, resized_annotations);

    std::string ann_train_path = root + "/output/train.json";
    std::string ann_val_path = root + "/output/val.json";
    std::string ann_test_path = root + "/output/test.json";

    auto train_dataset = VoyanceCOCODataset(ann_train_path, "train", CLASSES).map(torch::data::transforms::Stack<>());
    auto options = torch::data::DataLoaderOptions();
    options.drop_last(true);
    options.batch_size(batch_size);
    // options.workers(2);

    // Generate a data loader.
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(train_dataset), options);

    // Calculate total number of epochs to train and the epochs to decay learning rate at (i.e. convert iterations to epochs)
    // To convert iterations to epochs, divide iterations by the number of iterations per epoch
    // The paper trains for 120,000 iterations with a batch size of 32, decays after 80,000 and 100,000 iterations
    // epochs = iterations // (len(train_dataset) // 32)
    int epochs = (int)(iterations / (int)train_dataset.size().value() / batch_size);

    decay_lr_at = [decay_lr_at, train_dataset, batch_size]()
    {
        std::vector<int> holder;
        for (auto &val : decay_lr_at)
        {
            holder.push_back((int)(val / (int)train_dataset.size().value() / batch_size));
        }
        return holder;
    }();

    // Epochs
    for (int epoch = start_epoch; epoch < 1; epoch++)
    {
        // Decay learning rate at particular epochs
        if (std::find(decay_lr_at.begin(), decay_lr_at.end(), epoch) != decay_lr_at.end())
            adjust_learning_rate(optimizer, decay_lr_to);

        // One epoch's training
        // train(train_loader, model, criterion, optimizer, epoch, device);
        model->train(); // training mode enables dropout

        AverageMeter batch_time = AverageMeter(); // forward prop. + back prop. time
        AverageMeter data_time = AverageMeter();  // data loading time
        AverageMeter losses = AverageMeter();     // loss

        std::time_t start = std::time(NULL);

        /**
         * DATA LOADER PIN TO MEMORY,
         * COLLATE FN IF NOT POSSIBLE IN C++
         * RUN TRAINING ON THE DATASET
         */
        int batch_index = 0;
        for (auto batch : *train_loader)
        {
            auto data = batch.data.to(device);
            at::Tensor input;
            try
            {
                /* pin to cuda */
                input = at::empty(data.sizes()).pin_memory(device);
                data.copy_(data, true);
            }
            catch (const std::exception &e)
            {
                input = data;
                // std::cerr << e.what() << '\n';
                std::cerr << "\nDATA CAN NOT PIN TO MEMORY ON: " << device << '\n'
                          << std::endl;
            }

            // slice target(difficulties, bboxs, lables)
            auto bboxes = batch.target.index({"...", torch::indexing::Slice(torch::indexing::None, 4)}).to(torch::kFloat32).to(device);
            auto labels = batch.target.index({"...", torch::indexing::Slice(4, 5)}).to(torch::kLong).to(device);
            auto difficulties = batch.target.index({"...", torch::indexing::Slice(5, torch::indexing::None)}).to(torch::kLong).to(device);

            // Forward prop.
            PredictionConvolutionsFeatures prediction = model->forward(input); // (N, 8732, 4), (N, 8732, n_classes)

            // Loss
            torch::Tensor loss = criterion.forward(prediction.locs, prediction.class_score, bboxes, labels); // scalar
            std::cout << loss << std::endl;

            break;
        }
    }

    return 0;
}

// template <typename datatype>
// void train(datatype &train_loader, SSD300 &model, MultiBoxLoss &criterion, torch::optim::SGD &optimizer, int &epoch, torch::Device &device)
// {
//     model->train(); // training mode enables dropout

//     AverageMeter batch_time = AverageMeter(); // forward prop. + back prop. time
//     AverageMeter data_time = AverageMeter();  // data loading time
//     AverageMeter losses = AverageMeter();     // loss

//     std::time_t start = std::time(NULL);

//     /**
//      * DATA LOADER PIN TO MEMORY,
//      * COLLATE FN IF NOT POSSIBLE IN C++
//      * RUN TRAINING ON THE DATASET
//      */
//     for (auto batch : *train_loader)
//     {
//         // std::cout << batch.data.sizes() << std::endl;
//         // auto data = batch.data.to(device);
//         // at::Tensor input;
//         // try
//         // {
//         //     /* pin to cuda */
//         //     input = at::empty(data.sizes()).pin_memory(device);
//         //     data.copy_(data, true);
//         // }
//         // catch (const std::exception &e)
//         // {
//         //     input = data;
//         //     // std::cerr << e.what() << '\n';
//         //     std::cerr << "\nDATA CAN NOT PIN TO MEMORY ON: " << device << '\n'
//         //               << std::endl;
//         // }

//         // break;

//         // // slice target(class lable & bbox label)
//         // auto label = batch.target.index({"...", torch::indexing::Slice(4, torch::indexing::None)}).to(torch::kFloat32).to(device);
//         // label = label.view({label.sizes()[0], -1});

//         // auto bbox = batch.target.index({"...", torch::indexing::Slice(torch::indexing::None, 4)}).to(torch::kFloat32).to(device);
//         // bbox = bbox.view({bbox.sizes()[0], -1});
//     }
// }
