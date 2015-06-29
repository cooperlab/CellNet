#include "node.h"

Node::Node(std::string id, int mode): _id(id), _mode(mode), _in_edges(), _out_edges(), _counter(0), runtime_total_first(0), _counter_threads(0), ctrl(0){}
std::string Node::get_id(){return _id;}
void Node::insert_in_edge(Edge *edge_ptr){_in_edges.push_back(edge_ptr);}
void Node::insert_out_edge(Edge *edge_ptr){_out_edges.push_back(edge_ptr);}

void Node::copy_to_buffer(std::vector<cv::Mat> out){

	if(_mode == 0){	
		for(std::vector<int>::size_type i=0; i < _out_edges.size(); i++){

			//std::cout << "To Buffer" << std::endl; 
			// Lock access to buffer
			boost::mutex::scoped_lock lk(_out_edges.at(i)->_mutex);

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
		}
	}
	else if(_mode == 1){

			// This code considers only one thread
			int i = ctrl;
			boost::mutex::scoped_lock lk(_out_edges.at(i)->_mutex);
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

			// Update control
			if(++ctrl >= _out_edges.size()){
				ctrl = 0;
			}
			/******* Restricted Access ********/
	}
	else if(_mode == 2){

		// This code considers only one thread
		int block_size = out.size()/_out_edges.size();
		for(std::vector<int>::size_type i=0; i < _out_edges.size(); i++){

			//std::cout << "To Buffer" << std::endl; 
			// Lock access to buffer
			boost::mutex::scoped_lock lk(_out_edges.at(i)->_mutex);

			/******* Restricted Access ********/
			// Get current buffer
			std::vector<cv::Mat> *curr_buffer = _out_edges.at(i)->get_buffer();
			
			// Concatenate buffers
			std::vector<cv::Mat> new_buffer;

			if(i < _out_edges.size()-1){

				if(block_size > 0){

					new_buffer.reserve(curr_buffer->size() + block_size);
					new_buffer.insert( new_buffer.end(), curr_buffer->begin(), curr_buffer->end());
					new_buffer.insert( new_buffer.end(), out.begin() +  i * block_size, out.begin() + (i+1) * block_size - 1);
				}
			}
			else{

				new_buffer.reserve(curr_buffer->size() + out.size() - (_out_edges.size()-1)*block_size);
				new_buffer.insert( new_buffer.end(), curr_buffer->begin(), curr_buffer->end());
				new_buffer.insert( new_buffer.end(), out.begin() + (_out_edges.size()-1)*block_size, out.end());
			}

			// Set new buffer
			_out_edges.at(i)->set_buffer(new_buffer);
			/******* Restricted Access ********/
		}
	}
}

void Node::copy_from_buffer(cv::Mat &out){
	
	// Lock access to buffer
	boost::mutex::scoped_lock lk(_in_edges.at(0)->_mutex);

	/******* Restricted Access ********/
	// Get buffer
	std::vector<cv::Mat> *_buffer = _in_edges.at(0)->get_buffer();

	// Remove first element from buffer
	if(!_buffer->empty()){

		out = _buffer->at(0);
		_buffer->erase(_buffer->begin());	
	}
	/******* Restricted Access ********/
	//std::cout << "Node: " << _id << " unlocking	 buffer " << _in_edges.at(0)->_id << std::endl; 
}

void Node::copy_chunk_from_buffer(std::vector<cv::Mat> &out){
	
	// Lock access to buffer
	boost::mutex::scoped_lock lk(_in_edges.at(0)->_mutex);

	/******* Restricted Access ********/
	// Get buffer
	std::vector<cv::Mat> *_buffer = _in_edges.at(0)->get_buffer();

	// Remove first element from buffer
	if(!_buffer->empty()){

		std::vector<cv::Mat> new_block;
		new_block.reserve(_buffer->size());
		new_block.insert( new_block.end(), _buffer->begin(), _buffer->end());
		out = new_block;
		_buffer->clear();	
	}
	/******* Restricted Access ********/
	//std::cout << "Node: " << _id << " unlocking	 buffer " << _in_edges.at(0)->_id << std::endl; 
}

void Node::increment_counter(){
	
	boost::mutex::scoped_lock lk(_mutex_counter);
	_counter++;
}

void Node::increment_threads(){

	boost::mutex::scoped_lock lk(_mutex_ctrl);
	_counter_threads++;
}

bool Node::check_finished(){

	boost::mutex::scoped_lock lk(_mutex_ctrl);
	_counter_threads--;
	if(_counter_threads == 0){
		return true;
	}
	else{
		return false;
	}
}