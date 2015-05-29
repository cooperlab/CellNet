#include "debug_node.h"
#include "edge.h"
#include <iostream>

DebugNode::DebugNode(std::string id): Node(id){
}

void DebugNode::run(){

	// Debugger
	_output.push_back(_target);
	std::cout << "Generated images " << std::to_string(_output.size()) << std::endl;
}

void DebugNode::set_target(cv::Mat target){
	_target = target;
}