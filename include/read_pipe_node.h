#ifndef _READ_PIPE_NODE_H
#define _READ_PIPE_NODE_H

#include "node.h"
#include <vector>
#include <tuple>
#include <iostream>
#include <cv.h>
#include <opencv2/highgui/highgui.hpp> 
#include <opencv2/core/core.hpp>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>

class ReadPipeNode: public Node{

	public:
		ReadPipeNode(std::string id, std::string pipe_name, int mode);
		void *run();

  	private:
  		int read_from_pipe(std::vector<cv::Mat> &outs, std::vector<int> &labels);
  		std::string _pipe_name;
  		std::vector<int> _labels_buffer;
  		std::vector<cv::Mat> _data_buffer;
  		int _counter;
};
#endif