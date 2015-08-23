#ifndef _PREDICTION_PIPE_NODE_H
#define _PREDICTION_PIPE_NODE_H

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
#include <boost/make_shared.hpp>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <fstream>

class PredictionPipeNode: public Node{

	public:
		PredictionPipeNode(std::string id, int mode, int batch_size, std::string model_path, std::string params_file, int device_id, std::string pipe_name, std::string file_out);
		void *run();
		void init_model();
		void compute_accuracy();
		void print_out_labels();
		int read_from_pipe(std::vector<cv::Mat> &outs, std::vector<int> &labels);
		int step(int first_idx, int batch_size);
		void write_to_file();

	protected:
		int _batch_size;
		std::string _params_file;
		std::vector<cv::Mat> _data_buffer; 
		std::vector<int> _labels_buffer;
		std::vector<int> _predictions;
		std::string _test_model_path;
		boost::shared_ptr<caffe::Net<float>> _net;
		int _device_id;
		int _pipe;
		std::string _pipe_name;
		std::vector<int> _stored_labels;
		std::string _file_out;
		boost::shared_ptr<caffe::MemoryDataLayer<float>> _data_layer;
		boost::shared_ptr<caffe::Blob<float> > _out_layer;
};
#endif