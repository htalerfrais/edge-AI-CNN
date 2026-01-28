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

static int last_pred = -1;

int main(int argc, char** argv) {
    signal(SIGINT, sigHandler);
    signal(SIGTERM, sigHandler);

    int inPort = argc > 1 ? std::stoi(argv[1]) : 5000;
    int outPort = argc > 2 ? std::stoi(argv[2]) : 8554;
    int w = argc > 3 ? std::stoi(argv[3]) : 1280;
    int h = argc > 4 ? std::stoi(argv[4]) : 720;

    const char* model_path = "./cnn_model.txt";

    float output[OUTPUT_SIZE];
    //MLPModel* model = load_mlp_model(model_path);
    CNNModel* model = load_cnn_model(model_path);

    if (!model) {
        std::cerr << "Erreur chargement modèle" << std::endl;
        return 1;
    }

    bool digitDetected = false;
    int pred;
    double elapsed;
    float confidence;

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
        cv::findContours(thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        cv::drawContours(roiFrame, contours, -1, cv::Scalar(0,0,255), 2); // red contours

        int bestIdx = -1;
        double bestScore = 0.0;

        for (int i = 0; i < (int)contours.size(); i++) {
            double area = cv::contourArea(contours[i]);
            if (area < 150) continue;

            cv::Rect box = cv::boundingRect(contours[i]);

            // Aspect ratio constraint
            float ratio = (float)box.width / box.height;
            if (ratio < 0.2f || ratio > 2.0f) continue;

            // Fill ratio constraint
            double fill = area / (box.width * box.height);
            if (fill < 0.0) continue;

            double score = area * fill;
            if (score > bestScore) {
                bestScore = score;
                bestIdx = i;
            }
        }

        // DIGIT FOUND
        if (bestIdx >= 0) {
            digitDetected = true;

            cv::Rect box = cv::boundingRect(contours[bestIdx]);

            //cv::rectangle(frame, box, cv::Scalar(0,255,0), 2); // green bounding box

            // Flatten ROI for neural network
            // --- Extract digit mask (white digit on black background) ---
            cv::Mat digitMask = thresh(box);

            // Ensure binary (0 or 255)
            cv::threshold(digitMask, digitMask, 128, 255, cv::THRESH_BINARY);

            // --- Compute barycenter of original digit ---
            double sumX = 0.0, sumY = 0.0, countPix = 0.0;
            for (int y = 0; y < digitMask.rows; y++) {
                const uchar* row = digitMask.ptr<uchar>(y);
                for (int x = 0; x < digitMask.cols; x++) {
                    if (row[x] > 0) { // pixel belongs to digit
                        sumX += x;
                        sumY += y;
                        countPix += 1.0;
                    }
                }
            }

            double cx = sumX / countPix;
            double cy = sumY / countPix;

            // --- Resize digit to fit 20x20 ---
            int origW = digitMask.cols;
            int origH = digitMask.rows;
            int newW, newH;

            if (origW > origH) {
                newW = 20;
                newH = static_cast<int>(origH * (20.0 / origW));
            } else {
                newH = 20;
                newW = static_cast<int>(origW * (20.0 / origH));
            }

            cv::Mat digitResized;
            cv::resize(digitMask, digitResized, cv::Size(newW, newH), 0, 0, cv::INTER_AREA);

            // --- Place resized digit in 20x20 canvas using original barycenter ---
            cv::Mat digit20 = cv::Mat::zeros(20, 20, CV_8UC1);

            // Compute offsets to put the barycenter at (10,10)
            double scaleX = static_cast<double>(newW) / origW;
            double scaleY = static_cast<double>(newH) / origH;

            int offsetX = static_cast<int>(10 - cx * scaleX);
            int offsetY = static_cast<int>(10 - cy * scaleY);

            for (int y = 0; y < digitResized.rows; y++) {
                for (int x = 0; x < digitResized.cols; x++) {
                    int dx = x + offsetX;
                    int dy = y + offsetY;
                    if (dx >= 0 && dx < 20 && dy >= 0 && dy < 20) {
                        digit20.at<uchar>(dy, dx) = digitResized.at<uchar>(y, x);
                    }
                }
            }

            // --- Pad to 28x28 ---
            cv::Mat digit28;
            cv::copyMakeBorder(
                digit20,
                digit28,
                4, 4, 4, 4,
                cv::BORDER_CONSTANT,
                cv::Scalar(0)
            );
            

            // --- Force MNIST-style white strokes ---
            cv::normalize(digit28, digit28, 0, 255, cv::NORM_MINMAX);

            // Hard binarization
            //cv::threshold(digit28, digit28, 140, 255, cv::THRESH_BINARY);

            cv::Mat k = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3,3));
            cv::morphologyEx(digit28, digit28, cv::MORPH_CLOSE, k, cv::Point(-1,-1), 1);

            digit28 = cv::min(digit28 * 1.5, 255); // brighten by 50%

            cv::GaussianBlur(digit28, digit28, cv::Size(3,3), 0.5);

            cv::Mat overlay;
            int scale = 10;
            cv::resize(digit28, overlay, cv::Size(28 * scale, 28 * scale), 0, 0, cv::INTER_NEAREST);
            cv::cvtColor(overlay, overlay, cv::COLOR_GRAY2BGR);

            // Show it in the video
            overlay.copyTo(frame(cv::Rect(10, 10, overlay.cols, overlay.rows)));

            // Convert to float in [0,1]
            cv::Mat digitFloat;
            digit28.convertTo(digitFloat, CV_32F, 1.0 / 255.0);

            // MNIST-style normalization
            const float MNIST_MEAN = 0.1307f;
            const float MNIST_STD  = 0.3081f;
            digitFloat = (digitFloat - MNIST_MEAN) / MNIST_STD;

            float nn_input[784];
            for (int y = 0; y < 28; y++) {
                const float* row = digitFloat.ptr<float>(y);
                for (int x = 0; x < 28; x++) {
                    nn_input[y*28 + x] = row[x];
                }
            }
            
            auto t1 = std::chrono::steady_clock::now();
            forward_pass_cnn(model, nn_input, output);
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