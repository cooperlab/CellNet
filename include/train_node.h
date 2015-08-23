#ifndef _TRAIN_NODE_H
#define _TRAIN_NODE_H

#include "node.h"
#include "edge.h"
#include "utils.h"
#include <vector>
#include <iostream>
#include <glib.h>
#include <cv.h>
#include <opencv2/highgui/highgui.hpp> 
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"
#include "caffe/blob.hpp"
#include <iostream>
#include <boost/shared_ptr.hpp>

class TrainNode: public Node{

	public:
		TrainNode(std::string id, int mode, int batch_size, int device_id, std::string model_path, float base_lr, float momentum, float gamma, int iter);
		void *run();
		void init_model();
		void compute_update_value();
		void snapshot();
		int train_step(int first_idx);
		void cross_validate(std::vector<cv::Mat> batch, std::vector<int> batch_labels);

	protected:
		int _batch_size;
		std::string _model_path;
		std::vector<cv::Mat> _data_buffer; 
		std::vector<int> _labels_buffer;
		boost::shared_ptr<caffe::Net<float>> _net;
		float _base_lr;
		float _momentum;
		float _gamma;
		std::vector<boost::shared_ptr<caffe::Blob<float>>> _history;
		std::vector<boost::shared_ptr<caffe::Blob<float>>> _temp;
		boost::shared_ptr<caffe::Blob<float> > _out_layer;
		boost::shared_ptr<caffe::MemoryDataLayer<float>> _data_layer;
		int _iter;
		int _device_id;
};
#endif