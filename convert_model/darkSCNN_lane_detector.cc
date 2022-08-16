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

#include "modules/perception/camera/lib/lane/detector/darkSCNN/darkSCNN_lane_detector.h"

#include <algorithm>
#include <map>
#include "math.h"
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "torch/script.h"
#include "torch/torch.h"
#include "cyber/common/file.h"

#include "modules/perception/camera/common/util.h"
#include "modules/perception/inference/inference_factory.h"
#include "modules/perception/inference/utils/resize.h"

namespace apollo {
namespace perception {
namespace camera {

using apollo::cyber::common::GetAbsolutePath;
using apollo::cyber::common::GetProtoFromFile;

const char* INPUT_BLOB_NAME = "input.1";//"input_0"  
const char* OUTPUT_BLOB_NAME1 = "2602";//1*32*64
const char* OUTPUT_BLOB_NAME2 = "2611";//2*32*64
const char* OUTPUT_BLOB_NAME3 = "2620";//4*32*64
lane_output_height_ = 480;
lane_output_width_= 640;

Logger g_logger_;
static constexpr int kBatchSize = 1;

void DarkSCNNLaneDetector::OnnxToTRTModel(const std::string& model_file, nvinfer1::ICudaEngine** engine_ptr) {

  int verbosity = static_cast<int>(nvinfer1::ILogger::Severity::kWARNING);

  // create the builder
  const auto explicit_batch =
      static_cast<uint32_t>(kBatchSize) << static_cast<uint32_t>(
          nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(g_logger_);
  nvinfer1::INetworkDefinition* network =
      builder->createNetworkV2(explicit_batch);
  AINFO << "0";

  // parse onnx model
  auto parser = nvonnxparser::createParser(*network, g_logger_);
  if (!parser->parseFromFile(model_file.c_str(), verbosity)) {
    std::string msg("failed to parse onnx file");
    g_logger_.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
    exit(EXIT_FAILURE);
  }
  AINFO << "0";

  // Build the engine
  builder->setMaxBatchSize(kBatchSize);
  nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
  AINFO << "0.0";
  config->setMaxWorkspaceSize(1 << 30);
  AINFO << "0.0.0";
  nvinfer1::ICudaEngine* engine =
      builder->buildEngineWithConfig(*network, *config);
  AINFO << "0";
  
  
  std::ofstream p("modules/perception/camera/lib/lane/detector/darkSCNN/pinet.engine", std::ios::binary);
  nvinfer1::IHostMemory* modelStream = engine->serialize(); 
  p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size()); // 写入
	modelStream->destroy(); // 销毁

  *engine_ptr = engine;
  parser->destroy();
  network->destroy();
  config->destroy();
  builder->destroy();
}

float* DarkSCNNLaneDetector::blobFromImage(cv::Mat& img){
  float* blob = new float[PINET_HEIGHT*PINET_WIDTH*3];
  for (int c = 0; c < 3; c++) {
    for (int  h = 0; h < PINET_HEIGHT; h++) {
      for (int w = 0; w < PINET_WIDTH; w++) {
          blob[c * PINET_WIDTH * PINET_HEIGHT + h * PINET_WIDTH + w] =
              (float)img.at<cv::Vec3b>(h, w)[c]/255.0 ;
  }}}
  return blob;
}

void DarkSCNNLaneDetector::doInference(nvinfer1::IExecutionContext& context, float* input, float* output1, float* output2, float* output3, 
      const int output_size, cv::Size input_shape) {
  
  const nvinfer1::ICudaEngine& engine = context.getEngine();

  // Pointers to input and output device buffers to pass to engine.
  // Engine requires exactly IEngine::getNbBindings() number of buffers.
  assert(engine.getNbBindings() == 4);
  void* buffers[4];

  // In order to bind the buffers, we need to know the names of the input and output tensors.
  // Note that indices are guaranteed to be less than IEngine::getNbBindings()
  const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
  assert(engine.getBindingDataType(inputIndex) == nvinfer1::DataType::kFLOAT);

  const int outputIndex1 = engine.getBindingIndex(OUTPUT_BLOB_NAME1);
  const int outputIndex2 = engine.getBindingIndex(OUTPUT_BLOB_NAME2);
  const int outputIndex3 = engine.getBindingIndex(OUTPUT_BLOB_NAME3);
  assert(engine.getBindingDataType(outputIndex1) == nvinfer1::DataType::kFLOAT);
  assert(engine.getBindingDataType(outputIndex2) == nvinfer1::DataType::kFLOAT);
  assert(engine.getBindingDataType(outputIndex3) == nvinfer1::DataType::kFLOAT);
  //int mBatchSize = engine.getMaxBatchSize();
  // Create GPU buffers on device
  GPU_CHECK(cudaMalloc(&buffers[inputIndex], 3*PINET_HEIGHT*PINET_WIDTH*sizeof(float)));
  GPU_CHECK(cudaMalloc(&buffers[outputIndex1], 1 * output_size*sizeof(float)));
  GPU_CHECK(cudaMalloc(&buffers[outputIndex2], 2 * output_size*sizeof(float)));
  GPU_CHECK(cudaMalloc(&buffers[outputIndex3], 4 * output_size*sizeof(float)));

  // Create stream
  cudaStream_t stream;
  GPU_CHECK(cudaStreamCreate(&stream));

  // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
  GPU_CHECK(cudaMemcpyAsync(buffers[inputIndex], input, 3 * input_shape.height * input_shape.width * sizeof(float), cudaMemcpyHostToDevice, stream));
  context.enqueue(1, buffers, stream, nullptr);


  GPU_CHECK(cudaMemcpyAsync(output1, buffers[outputIndex1], 1 * output_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
  GPU_CHECK(cudaMemcpyAsync(output2, buffers[outputIndex2], 2 * output_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
  GPU_CHECK(cudaMemcpyAsync(output3, buffers[outputIndex3], 4 * output_size * sizeof(float), cudaMemcpyDeviceToHost, stream));

  cudaStreamSynchronize(stream);

  // Release stream and buffers
  cudaStreamDestroy(stream);
  GPU_CHECK(cudaFree(buffers[inputIndex]));
  GPU_CHECK(cudaFree(buffers[outputIndex1]));
  GPU_CHECK(cudaFree(buffers[outputIndex2]));
  GPU_CHECK(cudaFree(buffers[outputIndex3]));
}

bool DarkSCNNLaneDetector::Init(const LaneDetectorInitOptions &options) {  

  int pinet_height = 256;
  int pinet_width = 512;
  cv::Mat img(1080, 1920, CV_8UC3,cv::Scalar(0, 0, 0));
  cv::Mat pr_img(PINET_HEIGHT, PINET_WIDTH, CV_8UC3,cv::Scalar(0, 0, 0));
  
  //使用engine加载模型
  char *trtModelStream{nullptr};
  size_t size{0};

  std::ifstream file("modules/perception/camera/lib/lane/detector/darkSCNN/pinet.engine", std::ios::binary);
  if (file.good()) {
      file.seekg(0, file.end);
      size = file.tellg();
      file.seekg(0, file.beg);
      trtModelStream = new char[size];
      assert(trtModelStream);
      file.read(trtModelStream, size);
      file.close();
  }
  nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(g_logger_);
  assert(runtime != nullptr);
  nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
  assert(engine != nullptr); 
  nvinfer1::IExecutionContext* rpn_context_ = engine->createExecutionContext();
  assert(context != nullptr);
  delete[] trtModelStream;
  engine->destroy();
  runtime->destroy();


  // 使用onnx加载模型 and save engine
  // OnnxToTRTModel(rpn_onnx_file_, &rpn_engine_);

  // if (rpn_engine_ == nullptr) {
  //   AERROR << "Failed to load ONNX file.";
  // }
  // // create execution context from the engine
  // nvinfer1::IExecutionContext* rpn_context_ = rpn_engine_->createExecutionContext();

  if (rpn_context_ == nullptr) {
    AERROR << "Failed to create TensorRT Execution Context.";
  }

  // original code
  //--------------------------------------------------------------------------------------//

  std::string proto_path = GetAbsolutePath(options.root_dir, options.conf_file);
  if (!GetProtoFromFile(proto_path, &darkscnn_param_)) {
    AINFO << "load proto param failed, root dir: " << options.root_dir;
    return false;
  }
  std::string param_str;
  google::protobuf::TextFormat::PrintToString(darkscnn_param_, &param_str);
  AINFO << "darkSCNN param: " << param_str;

  const auto model_param = darkscnn_param_.model_param();
  std::string model_root =
      GetAbsolutePath(options.root_dir, model_param.model_name());
  std::string proto_file =
      GetAbsolutePath(model_root, model_param.proto_file());
  std::string weight_file =
      GetAbsolutePath(model_root, model_param.weight_file());
  AINFO << " proto_file: " << proto_file;
  AINFO << " weight_file: " << weight_file;
  AINFO << " model_root: " << model_root;

  base_camera_model_ = options.base_camera_model;
  if (base_camera_model_ == nullptr) {
    AERROR << "options.intrinsic is nullptr!";
    input_height_ = 1080;
    input_width_ = 1920;
  } else {
    input_height_ = static_cast<uint16_t>(base_camera_model_->get_height());
    input_width_ = static_cast<uint16_t>(base_camera_model_->get_width());
  }
  ACHECK(input_width_ > 0) << "input width should be more than 0";
  ACHECK(input_height_ > 0) << "input height should be more than 0";

  AINFO << "input_height: " << input_height_;
  AINFO << "input_width: " << input_width_;

  // compute image provider parameters
  input_offset_y_ = static_cast<uint16_t>(model_param.input_offset_y());
  input_offset_x_ = static_cast<uint16_t>(model_param.input_offset_x());
  resize_height_ = static_cast<uint16_t>(model_param.resize_height());
  resize_width_ = static_cast<uint16_t>(model_param.resize_width());
  crop_height_ = static_cast<uint16_t>(model_param.crop_height());
  crop_width_ = static_cast<uint16_t>(model_param.crop_width());
  confidence_threshold_lane_ = model_param.confidence_threshold();

  CHECK_LE(crop_height_, input_height_)
      << "crop height larger than input height";
  CHECK_LE(crop_width_, input_width_) << "crop width larger than input width";

  if (model_param.is_bgr()) {
    data_provider_image_option_.target_color = base::Color::BGR;
    image_mean_[0] = model_param.mean_b();
    image_mean_[1] = model_param.mean_g();
    image_mean_[2] = model_param.mean_r();
  } else {
    data_provider_image_option_.target_color = base::Color::RGB;
    image_mean_[0] = model_param.mean_r();
    image_mean_[1] = model_param.mean_g();
    image_mean_[2] = model_param.mean_b();
  }
  data_provider_image_option_.do_crop = false; // dafault is true , false if using pinet
  data_provider_image_option_.crop_roi.x = input_offset_x_;
  data_provider_image_option_.crop_roi.y = input_offset_y_;
  data_provider_image_option_.crop_roi.height = crop_height_;
  data_provider_image_option_.crop_roi.width = crop_width_;

  // oringal code but should not de used
  // ----------------------------------------------------//

  // cudaDeviceProp prop;
  // cudaGetDeviceProperties(&prop, options.gpu_id);
  // AINFO << "GPU: " << prop.name;

  // const auto net_param = darkscnn_param_.net_param();
  // net_inputs_.push_back(net_param.input_blob());
  // net_outputs_.push_back(net_param.seg_blob());
  // if (model_param.model_type() == "CaffeNet" && net_param.has_vpt_blob() &&
  //     net_param.vpt_blob().size() > 0) {
  //   net_outputs_.push_back(net_param.vpt_blob());
  // }

  // for (auto name : net_inputs_) {
  //   AINFO << "net input blobs: " << name;
  // }
  // for (auto name : net_outputs_) {
  //   AINFO << "net output blobs: " << name;
  // }

  // // initialize caffe net
  // const auto &model_type = model_param.model_type();
  // AINFO << "model_type: " << model_type;
  // cnnadapter_lane_.reset(
  //     inference::CreateInferenceByName(model_type, proto_file, weight_file,
  //                                      net_outputs_, net_inputs_, model_root));
  // ACHECK(cnnadapter_lane_ != nullptr);

  // cnnadapter_lane_->set_gpu_id(options.gpu_id);
  // ACHECK(resize_width_ > 0) << "resize width should be more than 0";
  // ACHECK(resize_height_ > 0) << "resize height should be more than 0";
  // std::vector<int> shape = {1, 3, resize_height_, resize_width_};
  // std::map<std::string, std::vector<int>> input_reshape{
  //     {net_inputs_[0], shape}};
  // AINFO << "input_reshape: " << input_reshape[net_inputs_[0]][0] << ", "
  //       << input_reshape[net_inputs_[0]][1] << ", "
  //       << input_reshape[net_inputs_[0]][2] << ", "
  //       << input_reshape[net_inputs_[0]][3];
  // if (!cnnadapter_lane_->Init(input_reshape)) {
  //   AINFO << "net init fail.";
  //   return false;
  // }

  // for (auto &input_blob_name : net_inputs_) {
  //   auto input_blob = cnnadapter_lane_->get_blob(input_blob_name);
  //   AINFO << input_blob_name << ": " << input_blob->channels() << " "
  //         << input_blob->height() << " " << input_blob->width();
  // }

  // auto output_blob = cnnadapter_lane_->get_blob(net_outputs_[0]);
  // AINFO << net_outputs_[0] << " : " << output_blob->channels() << " "
  //       << output_blob->height() << " " << output_blob->width();
  // lane_output_height_ = output_blob->height();
  // lane_output_width_ = output_blob->width();
  // num_lanes_ = output_blob->channels();

  // if (net_outputs_.size() > 1) {
  //   vpt_mean_.push_back(model_param.vpt_mean_dx());
  //   vpt_mean_.push_back(model_param.vpt_mean_dy());
  //   vpt_std_.push_back(model_param.vpt_std_dx());
  //   vpt_std_.push_back(model_param.vpt_std_dy());
  // }

  // input_width_=1920;
  // input_height_=1080;
  std::vector<int> lane_shape = {1, 1, lane_output_height_, lane_output_width_};
  lane_blob_.reset(new base::Blob<float>(lane_shape));

  return true;
}

bool DarkSCNNLaneDetector::Detect(const LaneDetectorOptions &options,
                                  CameraFrame *frame) {
  if (frame == nullptr) {
    AINFO << "camera frame is empty.";
    return false;
  }
  // auto start = std::chrono::high_resolution_clock::now();
  auto data_provider = frame->data_provider;
  if (input_width_ != data_provider->src_width()) {
    AERROR << "Input size is not correct: " << input_width_ << " vs "
           << data_provider->src_width();
    return false;
  }
  if (input_height_ != data_provider->src_height()) {
    AERROR << "Input size is not correct: " << input_height_ << " vs "
           << data_provider->src_height();
    return false;
  }

  // use data provider to crop input image
  if (!data_provider->GetImage(data_provider_image_option_, &image_src_)) {
    return false;
  }

  //----------------------------------------------------------------------------//
  
  memcpy(img.data, image_src_.cpu_data(), image_src_.total() * sizeof(uint8_t));
  cv::resize(img,pr_img,cv::Size(PINET_WIDTH,PINET_HEIGHT),0,0,cv::INTER_CUBIC);
  float* blob = blobFromImage(pr_img);//rollaxis,从HWC到CHW


  float* prob1 = new float[32*64 * 1];
  float* prob2 = new float[32*64 * 2];
  float* prob3 = new float[32*64 * 4];
  doInference(*rpn_context_, blob, prob1,prob2,prob3, output_size, pr_img.size());
  delete blob;
  // destroy the engine
  // rpn_context_->destroy();


  //后处理
  std::vector<std::vector<float>> lane_feature;
  std::vector<std::vector<int>> lane_x;
  std::vector<std::vector<int>> lane_y;
  std::vector<int> tmp;
  std::vector<float> tmp_float;
  cv::Mat mask_color(480, 640, CV_32FC1);
  mask_color.setTo(cv::Scalar(0));


  for (int i=0 ;i < 32; i++){
    for (int j=0; j < 64;j++){
      if (prob1[64*i+j] <threshold_point)
        continue;
      int x = (int)((prob2[64*i+j]+j)*10); // 640/64=10，原来512/64=8
      int y = (int)((prob2[32*64+64*i+j]+i)*15);  //480/32=15，原来256/32=8
      if ((x > 640) or (x < 0) or (y > 480) or (y < 0))
        continue;
      tmp_float.push_back(prob3[64*i+j]);
      tmp_float.push_back(prob3[64*i+j + 32*64]);
      tmp_float.push_back(prob3[64*i+j + 32*64*2]);
      tmp_float.push_back(prob3[64*i+j + 32*64*3]);  

      if (lane_feature.size() == 0){
        lane_feature.push_back(tmp_float);

        tmp.push_back((float)x);
        lane_x.push_back(tmp);
        tmp.clear();

        tmp.push_back((float)y);
        lane_y.push_back(tmp);
        tmp.clear();
      }
      else{
        int min_feature_index = -1;
        float min_feature_dis_point = 10000;
        for (unsigned int k=0;k< lane_feature.size(); k++)
        {
          float dis_point = 0;
          for (int p=0;p<4;p++)
            dis_point += pow((tmp_float[p] - lane_feature[k][p]),4);
          dis_point = sqrt(dis_point);
          if (min_feature_dis_point > dis_point){
            min_feature_dis_point = dis_point;
            min_feature_index = k;
          }
        }

        if (min_feature_dis_point <= threshold_instance){
          int size_lane = lane_x[min_feature_index].size();
          for (int p=0;p<4;p++)
          lane_feature[min_feature_index][p] = 
            (lane_feature[min_feature_index][p] * size_lane + tmp_float[p]) / (size_lane+1);
          lane_x[min_feature_index].push_back(x);
          lane_y[min_feature_index].push_back(y);
        }
        else if (lane_feature.size() < 12){
          lane_feature.push_back(tmp_float);

          tmp.push_back(x);
          lane_x.push_back(tmp);
          tmp.clear();

          tmp.push_back(y);
          lane_y.push_back(tmp);
          tmp.clear();
        }
      }
      tmp_float.clear();
    }
  }
  delete prob1;
  delete prob2;
  delete prob3;

  std::vector<float> x_mean;
  std::vector<int> sort;
  for (unsigned int i=0; i<lane_x.size();i++){
    if (lane_x[i].size() < 3) continue;
    float sum=0;
    for (unsigned int j=0; j<lane_x[i].size();j++) sum +=lane_x[i][j];
    sum /= lane_x[i].size();
    x_mean.push_back(sum);
  }

  for (unsigned int i=0;i<x_mean.size();i++){
    int sum =1;
    for (unsigned int j =0;j<x_mean.size();j++){
      if (x_mean[i] > x_mean[j]) sum++;
    }
    sort.push_back(sum);
  }
  
  int k=0;
  for (unsigned int i=0; i<lane_x.size();i++){
    AINFO<<"lane_x[i].size():"<<lane_x[i].size();
    if (lane_x[i].size() < 3) continue;
    for (unsigned int j=0; j<lane_x[i].size();j++){
      mask_color.at<float>(lane_y[i][j],lane_x[i][j]) = (float)(sort[k]);
    }
    k++;
  }
  cv::imwrite("/apollo/result/cmx_result.png",mask_color);


  //-----------------------------------------------------------------------------//

  // //  bottom 0 is data
  // auto input_blob = cnnadapter_lane_->get_blob(net_inputs_[0]);
  // auto blob_channel = input_blob->channels();
  // auto blob_height = input_blob->height();
  // auto blob_width = input_blob->width();
  // ADEBUG << "input_blob: " << blob_channel << " " << blob_height << " "
  //        << blob_width << std::endl;

  // if (blob_height != resize_height_) {
  //   AERROR << "height is not equal" << blob_height << " vs " << resize_height_;
  //   return false;
  // }
  // if (blob_width != resize_width_) {
  //   AERROR << "width is not equal" << blob_width << " vs " << resize_width_;
  //   return false;
  // }
  // ADEBUG << "image_blob: " << image_src_.blob()->shape_string();
  // ADEBUG << "input_blob: " << input_blob->shape_string();
  // // resize the cropped image into network input blob
  // inference::ResizeGPU(
  //     image_src_, input_blob, static_cast<int>(crop_width_), 0,
  //     static_cast<float>(image_mean_[0]), static_cast<float>(image_mean_[1]),
  //     static_cast<float>(image_mean_[2]), false, static_cast<float>(1.0));
  // ADEBUG << "resize gpu finish.";
  // cudaDeviceSynchronize();
  // cnnadapter_lane_->Infer();
  // ADEBUG << "infer finish.";

  // auto elapsed_1 = std::chrono::high_resolution_clock::now() - start;
  // int64_t microseconds_1 =
  //     std::chrono::duration_cast<std::chrono::microseconds>(elapsed_1).count();
  // time_1 += microseconds_1;

  // // convert network output to color map
  // const auto seg_blob = cnnadapter_lane_->get_blob(net_outputs_[0]);
  // ADEBUG << "seg_blob: " << seg_blob->shape_string();
  // std::vector<cv::Mat> masks;
  // for (int i = 0; i < num_lanes_; ++i) {
  //   cv::Mat tmp(lane_output_height_, lane_output_width_, CV_32FC1);
  //   memcpy(tmp.data,
  //          seg_blob->cpu_data() + lane_output_width_ * lane_output_height_ * i,
  //          lane_output_width_ * lane_output_height_ * sizeof(float));
  //   // cv::resize(tmp
  //   // , tmp, cv::Size(lane_output_width_, lane_output_height_), 0,
  //   //            0);
  //   masks.push_back(tmp);
  // }
  // std::vector<int> cnt_pixels(13, 0);
  // cv::Mat mask_color(lane_output_height_, lane_output_width_, CV_32FC1);
  // mask_color.setTo(cv::Scalar(0));
  // for (int c = 0; c < num_lanes_; ++c) {
  //   for (int h = 0; h < masks[c].rows; ++h) {
  //     for (int w = 0; w < masks[c].cols; ++w) {
  //       if (masks[c].at<float>(h, w) >= confidence_threshold_lane_) {
  //         mask_color.at<float>(h, w) = static_cast<float>(c);
  //         cnt_pixels[c]++;
  //       }
  //     }
  //   }
  // }
  // memcpy(lane_blob_->mutable_cpu_data(),
  //        reinterpret_cast<float *>(mask_color.data),
  //        lane_output_width_ * lane_output_height_ * sizeof(float));
  // // Don't use this way to copy data, it will modify data
  // // lane_blob_->set_cpu_data((float*)mask_color.data);
  // frame->lane_detected_blob = lane_blob_;

  // // retrieve vanishing point network output
  // if (net_outputs_.size() > 1) {
  //   const auto vpt_blob = cnnadapter_lane_->get_blob(net_outputs_[1]);
  //   ADEBUG << "vpt_blob: " << vpt_blob->shape_string();
  //   std::vector<float> v_point(2, 0);
  //   std::copy(vpt_blob->cpu_data(), vpt_blob->cpu_data() + 2, v_point.begin());
  //   // compute coordinate in net input image
  //   v_point[0] = v_point[0] * vpt_std_[0] + vpt_mean_[0] +
  //                (static_cast<float>(blob_width) / 2);
  //   v_point[1] = v_point[1] * vpt_std_[1] + vpt_mean_[1] +
  //                (static_cast<float>(blob_height) / 2);
  //   // compute coordinate in original image
  //   v_point[0] = v_point[0] / static_cast<float>(blob_width) *
  //                    static_cast<float>(crop_width_) +
  //                static_cast<float>(input_offset_x_);
  //   v_point[1] = v_point[1] / static_cast<float>(blob_height) *
  //                    static_cast<float>(crop_height_) +
  //                static_cast<float>(input_offset_y_);

  //   ADEBUG << "vanishing point: " << v_point[0] << " " << v_point[1];
  //   if (v_point[0] > 0 && v_point[0] < static_cast<float>(input_width_) &&
  //       v_point[1] > 0 && v_point[0] < static_cast<float>(input_height_)) {
  //     frame->pred_vpt = v_point;
  //   }
  // }

  // auto elapsed_2 = std::chrono::high_resolution_clock::now() - start;
  // int64_t microseconds_2 =
  //     std::chrono::duration_cast<std::chrono::microseconds>(elapsed_2).count();
  // time_2 += microseconds_2 - microseconds_1;

  // time_num += 1;
  // ADEBUG << "Avg detection infer time: " << time_1 / time_num
  //        << " Avg detection merge output time: " << time_2 / time_num;
  // ADEBUG << "Lane detection done!";
  return true;
}

std::string DarkSCNNLaneDetector::Name() const {
  return "DarkSCNNLaneDetector";
}

REGISTER_LANE_DETECTOR(DarkSCNNLaneDetector);

}  // namespace camera
}  // namespace perception
}  // namespace apollo
