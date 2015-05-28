#include "debug_node.h"
#include "edge.h"

DebugNode::DebugNode(){
}

void DebugNode::set_target(cv::Mat target){
	_output.insert(target);
}