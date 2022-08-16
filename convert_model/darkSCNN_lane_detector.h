/******************************************************************************
 * Copyright 2019 The Apollo Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/

#pragma once
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "opencv2/opencv.hpp"
#include <cuda_runtime_api.h>
#include "modules/perception/base/camera.h"
#include "modules/perception/camera/common/camera_frame.h"
#include "modules/perception/camera/common/data_provider.h"
#include "modules/perception/camera/lib/interface/base_lane_detector.h"
#include "modules/perception/camera/lib/lane/common/proto/darkSCNN.pb.h"
#include "modules/perception/inference/tensorrt/rt_net.h"
#include "modules/perception/lib/registerer/registerer.h"
//#include "cyber/common/log.h"
namespace apollo {
namespace perception {
namespace camera {

#define GPU_CHECK(ans) \
  { GPUAssert((ans), __FILE__, __LINE__); }
inline void GPUAssert(cudaError_t code, const char* file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort) exit(code);
  }
}

// Logger for TensorRT info/warning/errors
class Logger : public nvinfer1::ILogger {
 public:
  explicit Logger(Severity severity = Severity::kWARNING)
      : reportable_severity(severity) {}

  void log(Severity severity, const char* msg) override {
    // suppress messages with severity enum value greater than the reportable
    if (severity > reportable_severity) return;

    switch (severity) {
      case Severity::kINTERNAL_ERROR:
        std::cerr << "INTERNAL_ERROR: ";
        break;
      case Severity::kERROR:
        std::cerr << "ERROR: ";
        break;
      case Severity::kWARNING:
        std::cerr << "WARNING: ";
        break;
      case Severity::kINFO:
        std::cerr << "INFO: ";
        break;
      default:
        std::cerr << "UNKNOWN: ";
        break;
    }
    std::cerr << msg << std::endl;
  }

  Severity reportable_severity;
};

class DarkSCNNLaneDetector : public BaseLaneDetector {
 public:
  DarkSCNNLaneDetector() : BaseLaneDetector() {
    input_height_ = 0;
    input_width_ = 0;
    input_offset_y_ = 0;
    input_offset_x_ = 0;
    crop_height_ = 0;
    crop_width_ = 0;
    resize_height_ = 0;
    resize_width_ = 0;
    image_mean_[0] = 0;
    image_mean_[1] = 0;
    image_mean_[2] = 0;
    confidence_threshold_lane_ = 0;
    lane_output_height_ = 0;
    lane_output_width_ = 0;
    num_lanes_ = 0;
  }

  virtual ~DarkSCNNLaneDetector() {}

  bool Init(const LaneDetectorInitOptions &options = LaneDetectorInitOptions()) override;

  // @brief: detect lane from image.
  // @param [in]: options
  // @param [in/out]: frame
  // detected lanes should be filled, required,
  // 3D information of lane can be filled, optional.
  bool Detect(const LaneDetectorOptions &options, CameraFrame *frame) override;

  float* blobFromImage(cv::Mat& img);
  void doInference(nvinfer1::IExecutionContext& context, float* input, float* output1, float* output2, float* output3, 
      const int output_size, cv::Size input_shape);
  // static void decode_outputs(float* prob, std::vector<Object>& objects, float scale, const int img_w, const int img_h);
  void OnnxToTRTModel(const std::string& model_file, nvinfer1::ICudaEngine** engine_ptr);
  void InitTRT();



  std::string Name() const override;

 private:
  std::shared_ptr<inference::Inference> cnnadapter_lane_ = nullptr;
  std::shared_ptr<base::BaseCameraModel> base_camera_model_ = nullptr;
  darkSCNN::DarkSCNNParam darkscnn_param_;

  // parameters for data provider
  uint16_t input_height_;
  uint16_t input_width_;
  uint16_t input_offset_y_;
  uint16_t input_offset_x_;
  uint16_t crop_height_;
  uint16_t crop_width_;
  uint16_t resize_height_;
  uint16_t resize_width_;
  int image_mean_[3];
  std::vector<float> vpt_mean_;
  std::vector<float> vpt_std_;
  // parameters for network output
  float confidence_threshold_lane_;
  int lane_output_height_;
  int lane_output_width_;
  int num_lanes_;

  int64_t time_1 = 0;
  int64_t time_2 = 0;
  int time_num = 0;
  std::vector<int> compression_params;
  

  DataProvider::ImageOptions data_provider_image_option_;
  base::Image8U image_src_;
  std::vector<std::string> net_inputs_;
  std::vector<std::string> net_outputs_;
  std::shared_ptr<base::Blob<float>> lane_blob_ = nullptr;


  const std::string rpn_onnx_file_ ="modules/perception/camera/lib/lane/detector/darkSCNN/pinet_v2_remove_final.onnx";//

  nvinfer1::ICudaEngine* rpn_engine_;
  nvinfer1::IExecutionContext* rpn_context_;
  int output_size = 32*64;
  cv::Mat img;
  cv::Mat pr_img;
  // float* blob = new float[256*512*3];
  // static float* prob1 = new float[32*64 * 1];
  // static float* prob2 = new float[32*64 * 2];
  // static float* prob3 = new float[32*64 * 4];
  float* blob;
  float* prob1;
  float* prob2;
  float* prob3;
  int pinet_height;
  int pinet_width;
  float threshold_point=0.81;
  float threshold_instance = 0.08;

};

}  // namespace camera
}  // namespace perception
}  // namespace apollo
