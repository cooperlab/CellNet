#ifndef _WRITE_PIPE_NODE_H
#define _WRITE_PIPE_NODE_H

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

class WritePipeNode: public Node{

	public:
		WritePipeNode(std::string id, std::string pipe_name);
            
		void *run();

  	private:
  		void write_to_pipe();
  		void send_done_to_pipe();
  		std::string _pipe_name;
  		std::vector<int> _labels_buffer;
  		std::vector<cv::Mat> _data_buffer;
  		int _counter;
		int _pipe;
};
#endif
