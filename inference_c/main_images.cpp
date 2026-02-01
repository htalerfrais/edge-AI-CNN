#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <vector>
#include <algorithm>

#include "neural_network.h"
#include "process_frame.h"

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    const char* default_path = "../data/raw_images/4";
    fs::path source = argc > 1 ? fs::path(argv[1]) : fs::path(default_path);

    if (!fs::exists(source)) {
        std::cerr << "Source path does not exist: " << source << std::endl;
        return 1;
    }

    std::vector<fs::path> images;

    if (fs::is_regular_file(source)) {
        images.push_back(source);
    } else if (fs::is_directory(source)) {
        for (const auto& entry : fs::directory_iterator(source)) {
            if (entry.is_regular_file()) {
                images.push_back(entry.path());
            }
        }
    }

    if (images.empty()) {
        std::cerr << "No images found in: " << source << std::endl;
        return 1;
    }

    std::sort(images.begin(), images.end());
    fs::path output_dir("output/");
    fs::create_directories(output_dir);

    const char* model_path_cnn = "../models/cnn_model.txt";
    CNNModel* model_cnn = load_cnn_model(model_path_cnn);
    if (!model_cnn) {
        std::cerr << "Erreur chargement modèle CNN" << std::endl;
        return 1;
    }

    FrameProcessorState state;

    for (const auto& image_path : images) {
        std::cout << "Processing " << image_path << std::endl;
        cv::Mat frame = cv::imread(image_path.string());
        if (frame.empty()) {
            std::cerr << "  -> failed to read image" << std::endl;
            continue;
        }

        state.last_pred = -1;
        process_frame(frame, model_cnn, state, false);  /* full frame, no center ROI */

        fs::path out_path = output_dir / (image_path.stem().string() + "_processed.png");
        if (!cv::imwrite(out_path.string(), frame)) {
            std::cerr << "  -> failed to write " << out_path << std::endl;
        } else {
            std::cout << "  -> saved " << out_path << std::endl;
        }
    }

    return 0;
}
