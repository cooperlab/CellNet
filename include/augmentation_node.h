#ifndef _AUGMENTATION_NODE_H
#define _AUGMENTATION_NODE_H

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

class AugmentationNode: public Node{

	public:
		AugmentationNode(std::string id, int mode, int aug_factor);
    void augment_images(std::vector<cv::Mat> imgs, std::vector<int> labels);
		void *run();

  	private:
  		std::vector<int> _labels_buffer;
  		std::vector<cv::Mat> _data_buffer;
  		int _counter;
      int _aug_factor;
};
#endif
