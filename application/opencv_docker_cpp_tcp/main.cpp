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

std::atomic<bool> running(true);
void sigHandler(int) { running = false; }

int main(int argc, char** argv) {
    signal(SIGINT, sigHandler);
    signal(SIGTERM, sigHandler);

    int inPort = argc > 1 ? std::stoi(argv[1]) : 5000;
    int outPort = argc > 2 ? std::stoi(argv[2]) : 8554;
    int w = argc > 3 ? std::stoi(argv[3]) : 1280;
    int h = argc > 4 ? std::stoi(argv[4]) : 720;

    const char* model_path = "./mlp_model.txt";

    float output[OUTPUT_SIZE];
    MLPModel* model = load_mlp_model(model_path);
    if (!model) return 1;



    std::cout << "=== Pi5 Camera ===" << std::endl;
    std::cout << "In:" << inPort << " Out:" << outPort << " " << w << "x" << h << std::endl;

    std::string capPipe = 
        "tcpclientsrc host=127.0.0.1 port=" + std::to_string(inPort) + " ! "
        "h264parse ! avdec_h264 ! videoconvert ! video/x-raw,format=BGR ! "
        "appsink drop=1 sync=0";

    std::string outPipe = 
        "appsrc ! videoconvert ! video/x-raw,format=I420 ! "
        "x264enc tune=zerolatency speed-preset=ultrafast bitrate=4000 key-int-max=15 ! "
        "video/x-h264,profile=baseline ! h264parse config-interval=1 ! "
        "mpegtsmux ! tcpserversink host=0.0.0.0 port=" + std::to_string(outPort);

    cv::VideoCapture cap(capPipe, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) { std::cerr << "Erreur: input" << std::endl; return 1; }

    cv::VideoWriter writer(outPipe, cv::CAP_GSTREAMER, 0, 60, cv::Size(w, h), true);
    if (!writer.isOpened()) { std::cerr << "Erreur: output" << std::endl; return 1; }

    cv::Mat frame;
    int count = 0;

    // Type explicite du time_point
    std::chrono::steady_clock::time_point t0 =
        std::chrono::steady_clock::now();

    while (running && cap.read(frame)) {
        if (frame.empty()) continue;

        // === TRAITEMENT OPENCV ICI ===
        // Grayscale
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // Threshold (white object on black)
        cv::Mat thresh;
        cv::threshold(gray, thresh, 0, 255,
                    cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

        // Find contours
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(thresh, contours,
                        cv::RETR_EXTERNAL,
                        cv::CHAIN_APPROX_SIMPLE);

        // Select largest contour
        int bestIdx = -1;
        double bestArea = 0.0;

        for (int i = 0; i < (int)contours.size(); i++) {
            double area = cv::contourArea(contours[i]);
            if (area > bestArea && area > 300) { // area threshold
                bestArea = area;
                bestIdx = i;
            }
        }

        // f an object was found
        if (bestIdx >= 0) {
            cv::Rect box = cv::boundingRect(contours[bestIdx]);

            // Draw bounding box
            cv::rectangle(frame, box, cv::Scalar(0,255,0), 2);

            // Crop object
            cv::Mat roi = thresh(box);

            // Resize to 28×28
            cv::Mat digit28;
            cv::resize(roi, digit28, cv::Size(28,28),
                    0, 0, cv::INTER_AREA);

            // OPTIONAL: debug view
            // cv::imshow("Digit28", digit28);

            // digit28 is now ready for your neural net

        // Flatten 28x28 -> 784 and normalize to [0,1]
        float nn_input[784];

        for (int y = 0; y < 28; y++) {
            const uchar* row = digit28.ptr<uchar>(y);
            for (int x = 0; x < 28; x++) {
                nn_input[y * 28 + x] = row[x] / 255.0f;
            }
        }

        forward_pass(model, nn_input, output);
        
        int pred = get_prediction(nn_input);
        static int last_pred = -1;
        if (pred != last_pred) {
            std::cout << "Prediction: " << pred << std::endl;
            last_pred = pred;
        }
        }

        writer.write(frame);
        count++;

        std::chrono::steady_clock::time_point now =
            std::chrono::steady_clock::now();

        std::chrono::duration<double> dt = now - t0;

        if (dt.count() >= 1.0) {
            std::cout << "FPS: "
                      << static_cast<int>(count / dt.count())
                      << std::endl;

            count = 0;
            t0 = now;
        }
    }

    return 0;
}