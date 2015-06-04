#include "node.h"

Node::Node(std::string id): _is_ready(false), _is_valid(false), _id(id), _in_edges(), _out_edges(){}
std::string Node::get_id(){return _id;}
void Node::insert_in_edge(Edge *edge_ptr){_in_edges.push_back(edge_ptr);}
void Node::insert_out_edge(Edge *edge_ptr){_out_edges.push_back(edge_ptr);}

void Node::copy_to_buffer(std::vector<cv::Mat> out){

	for(std::vector<int>::size_type i=0; i < _out_edges.size(); i++){

		//std::cout << "To Buffer" << std::endl; 
		// Lock access to buffer
		//std::cout << "Node: " << _id << " trying to lock buffer " << _out_edges.at(i)->_id << std::endl; 
		boost::mutex::scoped_lock lk(_out_edges.at(i)->_mutex);

		//std::cout << "Node: " << _id << " locked buffer " << _out_edges.at(i)->_id << std::endl;

		/******* Restricted Access ********/
		// Get current buffer
		std::vector<cv::Mat> *curr_buffer = _out_edges.at(i)->get_buffer();
		
		// Concatenate buffers
		std::vector<cv::Mat> new_buffer;
		new_buffer.reserve(curr_buffer->size() + out.size());
		new_buffer.insert( new_buffer.end(), curr_buffer->begin(), curr_buffer->end());
		new_buffer.insert( new_buffer.end(), out.begin(), out.end());

		// Set new buffer
		_out_edges.at(i)->set_buffer(new_buffer);
		/******* Restricted Access ********/

		//std::cout << "Node: " << _id << " unlocking buffer " << _out_edges.at(i)->_id << std::endl; 
		_out_edges.at(i)->_mutex.unlock();
	}
}

void Node::copy_from_buffer(cv::Mat &out){
	
	// Lock access to buffer
	//std::cout << "From Buffer" << std::endl; 
	//std::cout << "Node: " << _id << " trying to lock buffer " << _in_edges.at(0)->_id << std::endl; 
	boost::mutex::scoped_lock lk(_in_edges.at(0)->_mutex);

	//std::cout << "Node: " << _id << " locked buffer " << _in_edges.at(0)->_id << std::endl;
	/******* Restricted Access ********/
	// Get buffer
	std::vector<cv::Mat> *_buffer = _in_edges.at(0)->get_buffer();

	// Remove first element from buffer
	if(!_buffer->empty()){

		out = _buffer->at(0);//.clone();

		//std::cout << "Buffer size:" << std::to_string(_buffer->size()) << std::endl; 
		_buffer->erase(_buffer->begin());	
		//std::cout << "Buffer size:" << std::to_string(_buffer->size()) << std::endl; 

		// Set new buffer and return first element
		//_in_edges.at(0)->set_buffer(*_buffer);
	}
	/******* Restricted Access ********/
	//std::cout << "Node: " << _id << " unlocking	 buffer " << _in_edges.at(0)->_id << std::endl; 
	_in_edges.at(0)->_mutex.unlock();
}