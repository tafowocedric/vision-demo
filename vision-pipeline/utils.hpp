
#include <fstream>
#include <iostream>
#include <json.hpp>

using json = nlohmann::json;

/**
 * @brief read json file
 *
 * @param filepath file path
 * @return json
 */
static json json_file_reader(std::string filepath)
{
    std::ifstream file(filepath, std::ifstream::binary);
    json data = json::parse(file);
    file.close();

    return data;
}

/**
 * @brief maps a key to an existing list of keys and return true or false, if key is found in the list of keys.
 *
 * @tparam T Dynamic type
 * @param key The key to map
 * @param keys List of keys
 * @return bool
 */
template <typename T>
static bool map_key_to_keys(T key, T keys)
{
    for (T val : keys)
    {
        if (val == key)
            return true;
    }
    return false;
}

/**
 * @brief Get object keys value object
 *
 * @param data The json data to iterate over
 * @param key type of key(s) to return
 * @return json of keys
 */
static json get_key_value(json data, std::string key)
{
    json keys = json::array();
    for (const json val : data)
    {
        keys.push_back(val[key]);
    }

    return keys;
}

/**
 * @brief filter an existing json base on list of filters and the filter key
 *
 * @param data The json data to filter
 * @param filter_array list of existing keys you want to filter out from the data
 * @param key json data filter key (the key to filter the data with)
 * @return json
 */
static json json_filter(json data, json filter_array, std::string key)
{
    json filtered;
    std::copy_if(data.begin(), data.end(), std::back_inserter(filtered), [filter_array, key](const json &item)
                 { return item.contains(key) && map_key_to_keys(item[key], filter_array); });

    return filtered;
}

/**
 * @brief save the data set to it respective file
 *
 * @tparam T Dynamic type
 * @param file The absolute filepath of the json file
 * @param dump_data data to dump
 */
template <typename T>
static void save_to_file(std::string file, T dump_data)
{
    std::ofstream output(file);
    output << dump_data;
    output.close();
};

/**
 * @brief save the coco data set to it respective file
 *
 * @param file The absolute filepath of the json file
 * @param data list json of images/annotations
 * @param categories list of json data categories
 */
static void save_coco(std::string file, json data, json categories)
{
    json dump_data;
    dump_data["images"] = data["images"];
    dump_data["annotations"] = data["annotations"];
    dump_data["categories"] = categories;

    save_to_file(file, dump_data);
}

/**
 * @brief split data in to train test
 *
 * @param images images
 * @param annotations annotations
 * @param split_size split size (0 - 1) equivalent to 0% ~ 100%
 * @return tuple<json, json>
 */
static json train_test_split(json images, json annotations, float split_size)
{
    int train_set_size = (int)(images.size() * split_size);
    json train_data = {{"annotations", json::array()}, {"images", json::array()}};
    json test_data = {{"annotations", json::array()}, {"images", json::array()}};

    for (int index = 0; index < images.size(); index++)
    {
        if (index < train_set_size)
        {
            train_data["annotations"].push_back(annotations[index]);
            train_data["images"].push_back(images[index]);
        }
        else
        {
            test_data["annotations"].push_back(annotations[index]);
            test_data["images"].push_back(images[index]);
        }
    }
    json result[2] = {train_data, test_data};
    return result;
}

/**
 * @brief load annotation data from file, create train, test, validation data, and write those data to a json file respectively.
 *
 * @param annotations_path The absolute filepath of the json file
 * @param having_annotations bool true or false if data images have annotations already
 * @param split split size for training (0 - 1) [default=0.6] equivalent to 0% ~ 100%
 * @return std::string * of size 3
 */
static std::string *split_annotation(std::string annotations_path, bool having_annotations, const float split = 0.6)
{
    // read json file
    json coco;

    try
    {
        coco = json_file_reader(annotations_path);
    }
    catch (const std::exception &e)
    {
        // TODO:: CATCH EXCEPTION HERES
        std::cerr << e.what() << '\n';
    }

    // annotated data
    json images = coco["images"];
    json annotations = coco["annotations"];
    json categories = coco["categories"];

    // create data paths
    std::filesystem::path dir_name = annotations_path;
    std::filesystem::path train_path = dir_name.parent_path().concat("/train.json");
    std::filesystem::path test_path = dir_name.parent_path().concat("/test.json");
    std::filesystem::path val_path = dir_name.parent_path().concat("/val.json");

    static std::string path_list[3] = {train_path, test_path, val_path};
    if (having_annotations)
    {
        // create an array of images id with annotations and filter images without annotations
        json images_ids_with_annotations = get_key_value(annotations, "image_id");
        images = json_filter(images, images_ids_with_annotations, "id");
    }

    // split dataset into train, test, validation datasets
    json data = train_test_split(images, annotations, split);
    json train = data[0];
    json rest = data[1];

    json data_yz = train_test_split(rest["images"], rest["annotations"], 0.65);
    json test = data_yz[0];
    json validation = data_yz[1];

    // save coco train, test, validation dataset in respective json files
    save_coco(train_path, train, categories);
    save_coco(test_path, test, categories);
    save_coco(val_path, validation, categories);

    return path_list;
};