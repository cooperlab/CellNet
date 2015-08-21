#include "train_node.h"

TrainNode::TrainNode(std::string id, int mode, int batch_size, int device_id, std::string model_path, float base_lr, float momentum, float gamma, int iter): Node(id, mode), _batch_size(batch_size), _model_path(model_path), _data_buffer(), _labels_buffer(), _net(), _base_lr(base_lr), _momentum(momentum), _gamma(gamma), _history(), _temp(), _iter(iter), _device_id(_device_id){
	
	runtime_total_first = utils::get_time();
	_data_buffer.clear();
	init_model();
}

void TrainNode::init_model(){

	// Setup GPU
	caffe::Caffe::SetDevice(_device_id);	
	caffe::Caffe::set_mode(caffe::Caffe::CPU);
	caffe::Caffe::DeviceQuery();

	// Initialize Net
	_net.reset(new caffe::Net<float>(_model_path.c_str(), caffe::TRAIN));

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
	int first_idx = 0;

	while(true){

		copy_chunk_from_buffer(_data_buffer, _labels_buffer);
		if(first_idx + _batch_size <= _data_buffer.size()){
			
			// For each epoch feed the model with a mini-batch of samples
			first_idx = train_step(first_idx);
			std::cout << first_idx << std::endl;
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

		// Goes thru the data over again
		for(int k=0; k < _iter-1; k++){
			
			std::cout << "iter: " << k << std::endl; 
			train_step(0);
		}

		std::cout << "******************" << std::endl << "TrainNode" << std::endl << "Total_time_first: " << std::to_string(utils::get_time() - runtime_total_first) << std::endl << "# of elements: " << std::to_string(_labels_buffer.size()) << std::endl << "******************" << std::endl;

		// Save model
		snapshot();

		// Notify it has finished
		for(std::vector<int>::size_type i=0; i < _out_edges.size(); i++){
			_out_edges.at(i)->set_in_node_done();
		}
	}

	return NULL;
}

int TrainNode::train_step(int first_idx){

	int epochs = (_data_buffer.size() - first_idx)/_batch_size;
	for(int i = 0; i < epochs; i++){	

		// Split batch
		std::vector<cv::Mat> batch;
		std::vector<int> batch_labels;

		// Reseve space
		batch.reserve(_batch_size);
		batch.insert(batch.end(), _data_buffer.begin() + first_idx, _data_buffer.begin() + first_idx + _batch_size);
		batch_labels.reserve(_batch_size);
		batch_labels.insert(batch_labels.end(), _labels_buffer.begin() + first_idx, _labels_buffer.begin() + first_idx + _batch_size);

		// Predict next batch
		cross_validate(batch, batch_labels);

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
		first_idx += _batch_size;
	}

	return first_idx;
}

void TrainNode::cross_validate(std::vector<cv::Mat> batch, std::vector<int> batch_labels){

	// Get memory layer from net
	const boost::shared_ptr<caffe::MemoryDataLayer<float>> data_layer = boost::static_pointer_cast<caffe::MemoryDataLayer<float>>(_net->layer_by_name("data"));

	// Add matrices
	data_layer->AddMatVector(batch, batch_labels);

	// Foward
	float loss;
	_net->ForwardPrefilled(&loss);
	const boost::shared_ptr<caffe::Blob<float> >& out_layer = _net->blob_by_name("ip2");

	const float* results = out_layer->cpu_data();

	// Append outputs
	std::vector<int> predictions;
	for(int j=0; j < out_layer->shape(0); j++){
		
		int idx_max = 0;
		float max = results[j*out_layer->shape(1) + 0];

		// Argmax
		for(int k=0; k < out_layer->shape(1); k++){

			if(results[j*out_layer->shape(1) + k] > max){

				max = results[j*out_layer->shape(1) + k];
				idx_max = k;
			}
		}

		predictions.push_back(idx_max);
	}

	// Compare labels
	int hit = 0;
	for(int k = 0; k < predictions.size(); k++){

		if(predictions[k] == batch_labels[k]){

			hit++;
		}
	}

	float acc = (float)hit/predictions.size();
	std::cout << "Batch accuracy: " << acc << std::endl;
}

void TrainNode::snapshot(){
	caffe::NetParameter net_param;

	_net->ToProto(&net_param, true);
	std::string filename("cell_net");
	std::string model_filename;

	// Add one to iter_ to get the number of iterations that have completed.
	model_filename = filename + ".caffemodel";

	std::cout << "Snapshotting to " << model_filename << std::endl;
	caffe::WriteProtoToBinaryFile(net_param, model_filename.c_str());
}

void TrainNode::compute_update_value(){

	const std::vector<boost::shared_ptr<caffe::Blob<float> > >& net_params = _net->params();
	const std::vector<float>& net_params_lr = _net->params_lr();
	const std::vector<float>& net_params_weight_decay = _net->params_weight_decay();

  	// get parameters
	float rate = _base_lr;
	float momentum = _momentum;
	float weight_decay = _gamma;
	std::string regularization_type = "L2";

	switch (caffe::Caffe::mode()) {
		case caffe::Caffe::CPU:
		for (int param_id = 0; param_id < net_params.size(); ++param_id) {

      		// Compute the value to history, and then copy them to the blob's diff.
			float local_rate = rate * net_params_lr[param_id];
			float local_decay = weight_decay * net_params_weight_decay[param_id];

			if (local_decay) {
				if (regularization_type == "L2") {

          			// add weight decay
					caffe::caffe_axpy(net_params[param_id]->count(), local_decay, net_params[param_id]->cpu_data(), net_params[param_id]->mutable_cpu_diff());
				} else if (regularization_type == "L1") {

					caffe::caffe_cpu_sign(net_params[param_id]->count(), net_params[param_id]->cpu_data(), _temp[param_id]->mutable_cpu_data());
					caffe::caffe_axpy(net_params[param_id]->count(), local_decay, _temp[param_id]->cpu_data(), net_params[param_id]->mutable_cpu_diff());
				} else {
					LOG(FATAL) << "Unknown regularization type: " << regularization_type;
				}
			}

			caffe::caffe_cpu_axpby(net_params[param_id]->count(), local_rate, net_params[param_id]->cpu_diff(), momentum, _history[param_id]->mutable_cpu_data());

      		// copy
			caffe::caffe_copy(net_params[param_id]->count(), _history[param_id]->cpu_data(), net_params[param_id]->mutable_cpu_diff());
		}
		break;
		case caffe::Caffe::GPU:
#ifndef CPU_ONLY

		for (int param_id = 0; param_id < net_params.size(); ++param_id) {

      		// Compute the value to history, and then copy them to the blob's diff.
			float local_rate = rate * net_params_lr[param_id];
			float local_decay = weight_decay * net_params_weight_decay[param_id];

			if (local_decay) {
				if (regularization_type == "L2") {

          			// add weight decay
					caffe::caffe_gpu_axpy(net_params[param_id]->count(), local_decay, net_params[param_id]->gpu_data(), net_params[param_id]->mutable_gpu_diff());
				} else if (regularization_type == "L1") {

					caffe::caffe_gpu_sign(net_params[param_id]->count(), net_params[param_id]->gpu_data(),_temp[param_id]->mutable_gpu_data());
					caffe::caffe_gpu_axpy(net_params[param_id]->count(), local_decay, _temp[param_id]->gpu_data(), net_params[param_id]->mutable_gpu_diff());
				} else {

					LOG(FATAL) << "Unknown regularization type: " << regularization_type;
				}
			}

			caffe::caffe_gpu_axpby(net_params[param_id]->count(), local_rate, net_params[param_id]->gpu_diff(), momentum, _history[param_id]->mutable_gpu_data());

      		// copy
			caffe::caffe_copy(net_params[param_id]->count(),_history[param_id]->gpu_data(),net_params[param_id]->mutable_gpu_diff());
		}
#else
		NO_GPU;
#endif
		break;
		default:
		LOG(FATAL) << "Unknown caffe mode: " << caffe::Caffe::mode();
	}
}

