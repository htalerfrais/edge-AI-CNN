/**
 * Pi 5 Camera Stream Passthrough
 * Reçoit flux H264, renvoie vers client
 */
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <csignal>
#include <atomic>
#include <algorithm>
#include "neural_network.h"
#include "contour_detection.h"
#include "contour_selector.h"
#include "digit_extractor.h"

std::atomic<bool> running(true);
void sigHandler(int) { running = false; }

static int last_pred = -1;

int main(int argc, char** argv) {
    signal(SIGINT, sigHandler);
    signal(SIGTERM, sigHandler);

    const char* inHost = argc > 1 ? argv[1] : "127.0.0.1";
    int inPort = argc > 2 ? std::stoi(argv[2]) : 5000;
    int outPort = argc > 3 ? std::stoi(argv[3]) : 8554;
    int w = argc > 4 ? std::stoi(argv[4]) : 1280;
    int h = argc > 5 ? std::stoi(argv[5]) : 720;

    float output[OUTPUT_SIZE];

    const char* model_path_mlp = "./mlp_model.txt";
    const char* model_path_cnn = "./cnn_model.txt";

    MLPModel* model_mlp = load_mlp_model(model_path_mlp);
    CNNModel* model_cnn = load_cnn_model(model_path_cnn);

    if (!model_cnn) {
        std::cerr << "Erreur chargement modèle" << std::endl;
        return 1;
    }

    bool digitDetected = false;
    int pred = -1;
    double elapsed = 0.0;
    float confidence = 0.0f;

    std::cout << "=== Pi5 Camera ===" << std::endl;
    std::cout << "In:" << inHost << ":" << inPort << " Out:" << outPort << " " << w << "x" << h << std::endl;

    std::string capPipe =
        std::string("tcpclientsrc host=") + inHost + " port=" + std::to_string(inPort) + " ! "
        "h264parse ! avdec_h264 ! videoconvert ! video/x-raw,format=BGR ! "
        "appsink drop=1 sync=0";

    std::string outPipe =
        "appsrc ! videoconvert ! video/x-raw,format=I420 ! "
        "x264enc tune=zerolatency speed-preset=ultrafast bitrate=4000 key-int-max=15 ! "
        "video/x-h264,profile=baseline ! h264parse config-interval=1 ! "
        "mpegtsmux ! tcpserversink host=0.0.0.0 port=" + std::to_string(outPort);

    cv::VideoCapture cap(capPipe, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        std::cerr << "Erreur: input" << std::endl;
        return 1;
    }

    cv::VideoWriter writer(outPipe, cv::CAP_GSTREAMER, 0, 60, cv::Size(w, h), true);
    if (!writer.isOpened()) {
        std::cerr << "Erreur: output" << std::endl;
        return 1;
    }

    cv::Mat frame;
    int count = 0;
    auto t0 = std::chrono::steady_clock::now();

    while (running && cap.read(frame)) {
        digitDetected = false;
        if (frame.empty()) continue;

        // CENTRAL ROI (60% of image)
        int roiW = frame.cols * 0.3;
        int roiH = frame.rows * 0.6;
        int roiX = (frame.cols - roiW) / 2;
        int roiY = (frame.rows - roiH) / 2;

        cv::Rect centerROI(roiX, roiY, roiW, roiH);
        //cv::rectangle(frame, centerROI, cv::Scalar(255,0,0), 2); // blue ROI
        cv::Mat roiFrame = frame(centerROI);

        // PREPROCESSING
        cv::Mat gray;
        cv::cvtColor(roiFrame, gray, cv::COLOR_BGR2GRAY);

        cv::Mat thresh;
        cv::adaptiveThreshold(
            gray, thresh, 255,
            cv::ADAPTIVE_THRESH_GAUSSIAN_C,
            cv::THRESH_BINARY_INV,
            15, 5
        );

        // CONTOUR DETECTION
        std::vector<std::vector<cv::Point>> contours;
        contour_det_find_all(thresh, contours);
        contour_det_draw(roiFrame, contours); // red contours

        int bestIdx = contour_sel_find_best(contours);

        // DIGIT FOUND
        if (bestIdx >= 0) {
            digitDetected = true;

            cv::Rect box = cv::boundingRect(contours[bestIdx]);

            //cv::rectangle(frame, box, cv::Scalar(0,255,0), 2); // green bounding box

            // --- Extract digit (28x28) ---
            cv::Mat digit28;
            digit_extr_extract(thresh, box, digit28);

            cv::Mat overlay;
            int scale = 10;
            cv::resize(digit28, overlay, cv::Size(28 * scale, 28 * scale), 0, 0, cv::INTER_NEAREST);
            cv::cvtColor(overlay, overlay, cv::COLOR_GRAY2BGR);

            // Show it in the video
            overlay.copyTo(frame(cv::Rect(10, 10, overlay.cols, overlay.rows)));

            float nn_input[784];
            digit_extr_to_nn_input(digit28, nn_input);
            
            auto t1 = std::chrono::steady_clock::now();
            forward_pass_cnn(model_cnn, nn_input, output);
            auto t2 = std::chrono::steady_clock::now();
            elapsed = std::chrono::duration<double, std::milli>(t2 - t1).count();

            pred = get_prediction(output);

            // Apply softmax
            float sum_exp = 0.0f;
            float prob[OUTPUT_SIZE];
            for (int i = 0; i < OUTPUT_SIZE; i++) {
                prob[i] = std::exp(output[i]);
                sum_exp += prob[i];
            }
            for (int i = 0; i < OUTPUT_SIZE; i++)
                prob[i] /= sum_exp;

            confidence = prob[pred];

            //const float CONF_THRESHOLD = 0.8f;

            if (digitDetected && pred != last_pred) {
                std::cout << "Prediction : " << pred 
                        << " in " << elapsed << " ms."
                        << " Confidence : " << confidence << std::endl;
                last_pred = pred;
            }

            // --- Draw bounding box ---
            // Global bounding box
            cv::Rect globalBox(
                box.x + centerROI.x,
                box.y + centerROI.y,
                box.width,
                box.height
            );
            cv::rectangle(frame, globalBox, cv::Scalar(0,255,0), 2);


            // --- Overlay prediction on bounding box ---
            char text[32];
            std::snprintf(text, sizeof(text), "%d (%.2f)", pred, confidence);
            cv::putText(frame, text,
                        cv::Point(globalBox.x + 2, globalBox.y + 20), // slightly offset from top-left
                        cv::FONT_HERSHEY_SIMPLEX, 0.7,                // font size
                        cv::Scalar(0,255,0),                          // green color
                        2);                                           // thickness
            
        }

        cv::rectangle(frame, centerROI, cv::Scalar(255,0,0), 2);

        writer.write(frame);
        count++;

        auto now = std::chrono::steady_clock::now();
        std::chrono::duration<double> dt = now - t0;
        if (dt.count() >= 1.0) {
            std::cout << "FPS: " << static_cast<int>(count / dt.count()) << std::endl;
            count = 0;
            t0 = now;
            if(pred == last_pred){
                std::cout << "Prediction : " << pred 
                << " in " << elapsed << " ms."
                << " Confidence : " << confidence << std::endl;
            }
        }
    }

    return 0;
}
