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
		TrainNode(std::string id, int mode, int batch_size, std::string model_path, float base_lr);
		void *run();
		void init_model();
		void compute_update_value();

	protected:
		int _batch_size;
		std::string _model_path;
		std::vector<cv::Mat> _data_buffer; 
		std::vector<int> _labels_buffer;
		const boost::shared_ptr<caffe::Net<float>> _net;
		float _base_lr;
		std::vector<boost::shared_ptr<caffe::Blob<float>>> _history;
		std::vector<boost::shared_ptr<caffe::Blob<float>>> _temp;
};
#endif