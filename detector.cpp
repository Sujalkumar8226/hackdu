/**
 * Road Pavement Detection — C++ Edge Processor
 * Real-time video capture → preprocessing → ONNX inference → result streaming
 *
 * Build:
 *   mkdir build && cd build
 *   cmake .. -DONNXRUNTIME_DIR=/usr/local/onnxruntime
 *   make -j4
 *
 * Dependencies:
 *   - OpenCV 4.x       (sudo apt install libopencv-dev)
 *   - ONNX Runtime 1.x (https://github.com/microsoft/onnxruntime/releases)
 *   - nlohmann/json    (header-only, included via CMake FetchContent)
 */

#include <opencv2/opencv.hpp>
#include <onnxruntime/core/providers/cpu/cpu_provider_factory.h>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <nlohmann/json.hpp>

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <atomic>
#include <queue>
#include <mutex>
#include <fstream>
#include <sstream>
#include <cmath>

using json = nlohmann::json;
using namespace std::chrono;

// ─────────────────────────────────────────────
//  CONSTANTS
// ─────────────────────────────────────────────
constexpr int   MODEL_INPUT_W  = 640;
constexpr int   MODEL_INPUT_H  = 640;
constexpr float CONF_THRESHOLD = 0.45f;
constexpr float NMS_THRESHOLD  = 0.45f;
constexpr int   NUM_CLASSES    = 7;

const std::vector<std::string> CLASS_NAMES = {
    "longitudinal_crack", "transverse_crack", "alligator_crack",
    "pothole", "rutting", "raveling", "edge_break"
};

// Color palette for bounding boxes (BGR)
const std::vector<cv::Scalar> CLASS_COLORS = {
    {  50, 200, 255}, // orange
    {  50, 255, 200}, // green-yellow
    {  50,  50, 255}, // red
    {   0,  50, 230}, // dark red (pothole — most severe)
    { 200, 100,  50}, // blue
    { 150, 200,  50}, // teal
    { 100,  50, 200}  // purple
};

// ─────────────────────────────────────────────
//  DETECTION RESULT
// ─────────────────────────────────────────────
struct Detection {
    int         classId;
    float       confidence;
    cv::Rect    box;
    std::string className;
};

// ─────────────────────────────────────────────
//  IMAGE PREPROCESSOR
// ─────────────────────────────────────────────
class Preprocessor {
public:
    /**
     * Letterbox resize: pad to square without distorting aspect ratio.
     * Returns the padded image and scale/offset for coordinate remapping.
     */
    static cv::Mat letterbox(const cv::Mat& src,
                             float& scale, float& padLeft, float& padTop)
    {
        int srcW = src.cols, srcH = src.rows;
        scale = std::min(
            static_cast<float>(MODEL_INPUT_W) / srcW,
            static_cast<float>(MODEL_INPUT_H) / srcH
        );

        int newW = static_cast<int>(srcW * scale);
        int newH = static_cast<int>(srcH * scale);
        padLeft  = (MODEL_INPUT_W - newW) / 2.0f;
        padTop   = (MODEL_INPUT_H - newH) / 2.0f;

        cv::Mat resized, padded;
        cv::resize(src, resized, {newW, newH}, 0, 0, cv::INTER_LINEAR);

        int top    = static_cast<int>(padTop);
        int bottom = MODEL_INPUT_H - newH - top;
        int left   = static_cast<int>(padLeft);
        int right  = MODEL_INPUT_W - newW - left;

        cv::copyMakeBorder(resized, padded, top, bottom, left, right,
                           cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
        return padded;
    }

    /**
     * Convert BGR Mat → NCHW float32 tensor (normalized 0–1).
     */
    static std::vector<float> toTensor(const cv::Mat& img)
    {
        cv::Mat rgb;
        cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);

        std::vector<float> tensor(1 * 3 * MODEL_INPUT_H * MODEL_INPUT_W);
        for (int c = 0; c < 3; ++c) {
            for (int h = 0; h < MODEL_INPUT_H; ++h) {
                for (int w = 0; w < MODEL_INPUT_W; ++w) {
                    tensor[c * MODEL_INPUT_H * MODEL_INPUT_W + h * MODEL_INPUT_W + w]
                        = rgb.at<cv::Vec3b>(h, w)[c] / 255.0f;
                }
            }
        }
        return tensor;
    }

    /**
     * Clahe + gentle denoising for better crack visibility.
     */
    static cv::Mat enhance(const cv::Mat& src)
    {
        cv::Mat lab;
        cv::cvtColor(src, lab, cv::COLOR_BGR2Lab);
        std::vector<cv::Mat> channels;
        cv::split(lab, channels);

        auto clahe = cv::createCLAHE(2.0, {8, 8});
        clahe->apply(channels[0], channels[0]);

        cv::Mat enhanced;
        cv::merge(channels, lab);
        cv::cvtColor(lab, enhanced, cv::COLOR_Lab2BGR);

        // Light bilateral filter — preserves edges while removing noise
        cv::Mat denoised;
        cv::bilateralFilter(enhanced, denoised, 5, 50, 50);
        return denoised;
    }
};

// ─────────────────────────────────────────────
//  ONNX INFERENCE ENGINE
// ─────────────────────────────────────────────
class OnnxDetector {
public:
    OnnxDetector(const std::string& modelPath)
        : env_(ORT_LOGGING_LEVEL_WARNING, "RoadDetector")
    {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(4);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        session_ = std::make_unique<Ort::Session>(
            env_, modelPath.c_str(), opts);

        // Cache input/output names
        Ort::AllocatorWithDefaultOptions allocator;
        inputName_  = session_->GetInputNameAllocated(0, allocator).get();
        outputName_ = session_->GetOutputNameAllocated(0, allocator).get();

        std::cout << "[+] ONNX model loaded: " << modelPath << "\n";
        std::cout << "[+] Input:  " << inputName_  << "\n";
        std::cout << "[+] Output: " << outputName_ << "\n";
    }

    std::vector<Detection> detect(const cv::Mat& frame)
    {
        float scale, padL, padT;
        cv::Mat enhanced  = Preprocessor::enhance(frame);
        cv::Mat letterboxed = Preprocessor::letterbox(enhanced, scale, padL, padT);
        auto tensor = Preprocessor::toTensor(letterboxed);

        // Build ONNX input tensor
        std::vector<int64_t> inputShape = {1, 3, MODEL_INPUT_H, MODEL_INPUT_W};
        auto memInfo = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memInfo, tensor.data(), tensor.size(),
            inputShape.data(), inputShape.size());

        const char* inNames[]  = {inputName_.c_str()};
        const char* outNames[] = {outputName_.c_str()};

        auto outputs = session_->Run(
            Ort::RunOptions{nullptr},
            inNames, &inputTensor, 1,
            outNames, 1);

        return parseYoloOutput(outputs[0], scale, padL, padT,
                               frame.cols, frame.rows);
    }

private:
    Ort::Env                        env_;
    std::unique_ptr<Ort::Session>   session_;
    std::string                     inputName_, outputName_;

    /**
     * Parse YOLOv8 output: shape [1, (4+num_classes), num_anchors]
     * Apply confidence filter + non-maximum suppression.
     */
    std::vector<Detection> parseYoloOutput(Ort::Value& output,
                                           float scale, float padL, float padT,
                                           int origW, int origH)
    {
        auto shape    = output.GetTensorTypeAndShapeInfo().GetShape();
        float* data   = output.GetTensorMutableData<float>();
        int numAnchor = static_cast<int>(shape[2]);
        int numAttr   = static_cast<int>(shape[1]); // 4 + num_classes

        std::vector<cv::Rect2f>  boxes;
        std::vector<float>       scores;
        std::vector<int>         classIds;

        for (int i = 0; i < numAnchor; ++i) {
            // Find max class score
            float maxConf = 0.0f;
            int   maxCls  = 0;
            for (int c = 4; c < numAttr; ++c) {
                float conf = data[c * numAnchor + i];
                if (conf > maxConf) { maxConf = conf; maxCls = c - 4; }
            }

            if (maxConf < CONF_THRESHOLD) continue;

            float cx = data[0 * numAnchor + i];
            float cy = data[1 * numAnchor + i];
            float bw = data[2 * numAnchor + i];
            float bh = data[3 * numAnchor + i];

            // Map back to original image coords
            float x1 = ((cx - bw / 2.0f) - padL) / scale;
            float y1 = ((cy - bh / 2.0f) - padT) / scale;
            float w  = bw / scale;
            float h  = bh / scale;

            // Clamp to image bounds
            x1 = std::max(0.0f, std::min(x1, static_cast<float>(origW)));
            y1 = std::max(0.0f, std::min(y1, static_cast<float>(origH)));
            w  = std::min(w,  static_cast<float>(origW)  - x1);
            h  = std::min(h,  static_cast<float>(origH) - y1);

            boxes.push_back({x1, y1, w, h});
            scores.push_back(maxConf);
            classIds.push_back(maxCls);
        }

        // NMS
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, scores, CONF_THRESHOLD, NMS_THRESHOLD, indices);

        std::vector<Detection> results;
        for (int idx : indices) {
            Detection d;
            d.classId    = classIds[idx];
            d.confidence = scores[idx];
            d.box        = cv::Rect(
                static_cast<int>(boxes[idx].x), static_cast<int>(boxes[idx].y),
                static_cast<int>(boxes[idx].width), static_cast<int>(boxes[idx].height));
            d.className  = (d.classId < NUM_CLASSES)
                            ? CLASS_NAMES[d.classId] : "unknown";
            results.push_back(d);
        }
        return results;
    }
};

// ─────────────────────────────────────────────
//  ANNOTATION + HUD RENDERER
// ─────────────────────────────────────────────
class Renderer {
public:
    static void draw(cv::Mat& frame,
                     const std::vector<Detection>& dets,
                     double fps, int frameNum)
    {
        // Bounding boxes
        for (const auto& d : dets) {
            const cv::Scalar& color = (d.classId < NUM_CLASSES)
                ? CLASS_COLORS[d.classId] : cv::Scalar(200,200,200);

            cv::rectangle(frame, d.box, color, 2);

            // Label background
            std::string label = d.className + " " +
                                std::to_string(static_cast<int>(d.confidence * 100)) + "%";
            int baseLine = 0;
            cv::Size sz  = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
                                            0.5, 1, &baseLine);
            cv::Rect labelRect(d.box.x, d.box.y - sz.height - 8,
                               sz.width + 8, sz.height + 8);
            cv::rectangle(frame, labelRect, color, -1);
            cv::putText(frame, label,
                        {d.box.x + 4, d.box.y - 4},
                        cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        cv::Scalar(0, 0, 0), 1);
        }

        // HUD overlay (top-left)
        drawHUD(frame, dets, fps, frameNum);
    }

private:
    static void drawHUD(cv::Mat& frame,
                        const std::vector<Detection>& dets,
                        double fps, int frameNum)
    {
        // Semi-transparent overlay
        cv::Mat overlay = frame.clone();
        cv::rectangle(overlay, {0, 0, 280, 110}, {20, 20, 20}, -1);
        cv::addWeighted(overlay, 0.65, frame, 0.35, 0, frame);

        cv::putText(frame, "ROAD PAVEMENT DETECTOR",
                    {8, 18}, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    {100, 220, 255}, 1);
        cv::putText(frame,
                    "FPS: " + std::to_string(static_cast<int>(fps)) +
                    "  Frame: " + std::to_string(frameNum),
                    {8, 38}, cv::FONT_HERSHEY_SIMPLEX, 0.45,
                    {180, 180, 180}, 1);
        cv::putText(frame,
                    "Defects detected: " + std::to_string(dets.size()),
                    {8, 58}, cv::FONT_HERSHEY_SIMPLEX, 0.45,
                    {100, 255, 150}, 1);

        // PCI bar
        int pciVal = computeSimplePCI(dets);
        cv::Scalar barColor = pciVal >= 70 ? cv::Scalar(50, 220, 50)
                            : pciVal >= 45 ? cv::Scalar(50, 190, 230)
                                           : cv::Scalar(50,  50, 240);
        cv::putText(frame, "PCI: " + std::to_string(pciVal),
                    {8, 80}, cv::FONT_HERSHEY_SIMPLEX, 0.5, barColor, 1);
        cv::rectangle(frame, {8, 88, 260, 10}, {60,60,60}, -1);
        cv::rectangle(frame, {8, 88, static_cast<int>(260 * pciVal / 100.0), 10},
                      barColor, -1);
    }

    static int computeSimplePCI(const std::vector<Detection>& dets)
    {
        if (dets.empty()) return 100;
        float penalty = 0;
        for (const auto& d : dets)
            penalty += d.confidence * 15.0f;
        return std::max(0, static_cast<int>(100 - penalty));
    }
};

// ─────────────────────────────────────────────
//  JSON OUTPUT WRITER
// ─────────────────────────────────────────────
void writeJsonResult(const std::vector<Detection>& dets, int frameNum,
                     const std::string& outPath)
{
    json j;
    j["frame"]      = frameNum;
    j["detections"] = json::array();

    for (const auto& d : dets) {
        j["detections"].push_back({
            {"class",      d.className},
            {"confidence", d.confidence},
            {"bbox", {d.box.x, d.box.y,
                      d.box.x + d.box.width,
                      d.box.y + d.box.height}}
        });
    }

    std::ofstream f(outPath);
    f << j.dump(2);
}

// ─────────────────────────────────────────────
//  MAIN PIPELINE
// ─────────────────────────────────────────────
int main(int argc, char** argv)
{
    std::string modelPath = "road_defect_yolov8.onnx";
    std::string videoSrc  = "0";  // default: webcam

    if (argc >= 2) modelPath = argv[1];
    if (argc >= 3) videoSrc  = argv[2];

    std::cout << "\n=== Road Pavement Detection (C++ Edge) ===\n";
    std::cout << "Model : " << modelPath  << "\n";
    std::cout << "Source: " << videoSrc   << "\n\n";

    // Load detector
    OnnxDetector detector(modelPath);

    // Open video source
    cv::VideoCapture cap;
    if (videoSrc == "0" || videoSrc == "1")
        cap.open(std::stoi(videoSrc));
    else
        cap.open(videoSrc);

    if (!cap.isOpened()) {
        std::cerr << "[!] Cannot open video source: " << videoSrc << "\n";
        return 1;
    }

    int    frameW = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int    frameH = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps    = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) fps = 30.0;

    // Video writer for output recording
    cv::VideoWriter writer("output_annotated.mp4",
                           cv::VideoWriter::fourcc('m','p','4','v'),
                           fps, {frameW, frameH});

    std::cout << "[+] Video: " << frameW << "x" << frameH
              << " @ " << fps << " fps\n";
    std::cout << "[+] Press 'q' to quit, 's' to save screenshot\n\n";

    int    frameNum  = 0;
    double measFps   = fps;
    auto   lastTime  = high_resolution_clock::now();

    cv::Mat frame;
    std::vector<Detection> dets;

    while (true) {
        if (!cap.read(frame) || frame.empty()) break;

        // Inference every 2nd frame for speed on CPU
        if (frameNum % 2 == 0) {
            dets = detector.detect(frame);
        }

        // FPS measurement
        auto now = high_resolution_clock::now();
        double elapsed = duration_cast<milliseconds>(now - lastTime).count() / 1000.0;
        if (elapsed > 0.5) {
            measFps  = 2.0 / elapsed;  // every 2nd frame
            lastTime = now;
        }

        // Annotate
        Renderer::draw(frame, dets, measFps, frameNum);
        writer.write(frame);

        // JSON output every 10 frames (for dashboard consumption)
        if (frameNum % 10 == 0)
            writeJsonResult(dets, frameNum, "latest_result.json");

        cv::imshow("Road Pavement Detector", frame);

        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) break;
        if (key == 's') {
            std::string fn = "screenshot_" + std::to_string(frameNum) + ".jpg";
            cv::imwrite(fn, frame);
            std::cout << "[+] Screenshot saved: " << fn << "\n";
        }

        ++frameNum;
    }

    cap.release();
    writer.release();
    cv::destroyAllWindows();

    std::cout << "\n[+] Done. Processed " << frameNum << " frames.\n";
    std::cout << "[+] Annotated video: output_annotated.mp4\n";
    return 0;
}
