#include "train_node.h"

TrainNode::TrainNode(std::string id, int mode, int batch_size, std::string model_path, float base_lr): Node(id, mode), _batch_size(batch_size), _model_path(model_path), _data_buffer(), _labels_buffer(), _net(new caffe::Net<float>(model_path.c_str(), caffe::TRAIN)), _base_lr(base_lr), _history(), _temp(){
	runtime_total_first = utils::get_time();
	_data_buffer.clear();
	init_model();
}

void TrainNode::init_model(){

    caffe::Caffe::SetDevice(0);
    caffe::Caffe::set_mode(caffe::Caffe::GPU);

    // Initialize the history
  	const std::vector<boost::shared_ptr<caffe::Blob<float> > >& net_params = this->_net->params();
  	_history.clear();
  	_temp.clear();
  	for (int i = 0; i < net_params.size(); ++i) {
    	const std::vector<int>& shape = net_params[i]->shape();
    	_history.push_back(boost::shared_ptr<caffe::Blob<float> >(new caffe::Blob<float>(shape)));
    	_temp.push_back(boost::shared_ptr<caffe::Blob<float> >(new caffe::Blob<float>(shape)));
 	}
}


void *TrainNode::run(){

	increment_threads();
	while(true){

		copy_chunk_from_buffer(_data_buffer, _labels_buffer);
		if(_data_buffer.size() >= _batch_size){

			// Track of number of elements
			for(std::vector<cv::Mat>::size_type i=0; i < _data_buffer.size(); i++){
				increment_counter();
			}
			
			// For each epoch feed the model with a mini-batch of samples
			int epochs = _data_buffer.size()/_batch_size;
			for(int i = 0; i < epochs; i++){	

				// Split batch
				std::vector<cv::Mat> batch;
				std::vector<int> batch_labels;

				// Reseve space
				batch.reserve(_batch_size);
				batch.insert(batch.end(), _data_buffer.begin(), _data_buffer.begin() + _batch_size);
				batch_labels.reserve(_batch_size);
				batch_labels.insert(batch_labels.end(), _labels_buffer.begin(), _labels_buffer.begin() + _batch_size);

				// Remove batch from buffer
				_data_buffer.erase(_data_buffer.begin(), _data_buffer.begin() + _batch_size);
				_labels_buffer.erase(_labels_buffer.begin(), _labels_buffer.begin() + _batch_size);

				// Get memory layer from net
				const boost::shared_ptr<caffe::MemoryDataLayer<float>> data_layer = boost::static_pointer_cast<caffe::MemoryDataLayer<float>>(_net->layer_by_name("data"));
				
				// Add matrices
				data_layer->AddMatVector(batch, batch_labels);
				std::cout << "# of Inputed Images: " << std::to_string(batch.size()) << std::endl;

				// Foward
				float loss;
				_net->ForwardPrefilled(&loss);
				std::cout << "Loss: " << std::to_string(loss) << std::endl;

				// Backward
				_net->Backward();

				// Update
				compute_update_value();
				_net->Update();
			}
		}
		else{

			// Check if all input nodes have already finished
			bool is_all_done = true;
			for(std::vector<int>::size_type i=0; i < _in_edges.size(); i++){

				if(!_in_edges.at(i)->is_in_node_done()){
					is_all_done = false;
				}
			}

			// All input nodes have finished
			if(is_all_done){
				break;
			}
		}
	}

	if(check_finished() == true){

		std::cout << "******************" << std::endl << "TrainNode" << std::endl << "Total_time_first: " << std::to_string(utils::get_time() - runtime_total_first) << std::endl << "# of elements: " << std::to_string(_counter) << std::endl << "******************" << std::endl;
		// Notify it has finished
		for(std::vector<int>::size_type i=0; i < _out_edges.size(); i++){
			_out_edges.at(i)->set_in_node_done();
		}
	}

	return NULL;
}

void TrainNode::compute_update_value(){

	const std::vector<boost::shared_ptr<caffe::Blob<float>>>& net_params = _net->params();
	const std::vector<float>& net_params_lr = _net->params_lr();
 	const std::vector<float>& net_params_weight_decay = _net->params_weight_decay();

  	// Get the learning rate
  	float rate = _base_lr;
  	
  	//ClipGradients();
  	float momentum = 1.0;
  	float weight_decay = 1.0;
  	std::string regularization_type = "L2";
	for (int param_id = 0; param_id < net_params.size(); ++param_id) {

      	// Compute the value to history, and then copy them to the blob's diff.
      	float local_rate = rate * net_params_lr[param_id];
      	float local_decay = weight_decay * net_params_weight_decay[param_id];

      	if (local_decay) {
        
	        if (regularization_type == "L2") {
	          	
	          	// add weight decay
		        caffe::caffe_gpu_axpy(net_params[param_id]->count(), local_decay, net_params[param_id]->gpu_data(), net_params[param_id]->mutable_gpu_diff());
		    } else if (regularization_type == "L1") {
		          
		        caffe::caffe_gpu_sign(net_params[param_id]->count(), net_params[param_id]->gpu_data(), _temp[param_id]->mutable_gpu_data());
		        caffe::caffe_gpu_axpy(net_params[param_id]->count(), local_decay, _temp[param_id]->gpu_data(), net_params[param_id]->mutable_gpu_diff());
		    } else {

	          LOG(FATAL) << "Unknown regularization type: " << regularization_type;
	        }
     	}
      	caffe::caffe_gpu_axpby(net_params[param_id]->count(), local_rate, net_params[param_id]->gpu_diff(), momentum, _history[param_id]->mutable_gpu_data());
      	// copy
      	caffe::caffe_copy(net_params[param_id]->count(), _history[param_id]->gpu_data(), net_params[param_id]->mutable_gpu_diff());
    }
}