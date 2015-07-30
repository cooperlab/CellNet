#ifndef _PREDICTION_NODE_H
#define _PREDICTION_NODE_H

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

class PredictionNode: public Node{

	public:
		PredictionNode(std::string id, int mode, int batch_size, std::string model_path, std::string params_file, int device);
		void *run();
		void init_model();
		void compute_accuracy();
		void print_out_labels();
		int step(int first_idx, int batch_size);

	protected:
		int _batch_size;
		std::string _params_file;
		std::vector<cv::Mat> _data_buffer; 
		std::vector<int> _labels_buffer;
		std::vector<int> _predictions;
		const boost::shared_ptr<caffe::Net<float>> _net;
		int _device;
};
#endif