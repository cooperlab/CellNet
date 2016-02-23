//
//	Copyright (c) 2015, Emory University
//	All rights reserved.
//
//	Redistribution and use in source and binary forms, with or without modification, are
//	permitted provided that the following conditions are met:
//
//	1. Redistributions of source code must retain the above copyright notice, this list of
//	conditions and the following disclaimer.
//
//	2. Redistributions in binary form must reproduce the above copyright notice, this list
// 	of conditions and the following disclaimer in the documentation and/or other materials
//	provided with the distribution.
//
//	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
//	EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
//	OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
//	SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//	INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
//	TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
//	BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
//	CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
//	WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
//	DAMAGE.
//
//
#include "train_node.h"



TrainNode::TrainNode(string id, int mode, int batch_size, int device_id, string model_path, 
					 float base_lr, float momentum, float gamma, int iter, string outFilename): 
Node(id, mode), 
_batch_size(batch_size), 
_model_path(model_path), 
_data_buffer(), 
_labels_buffer(), 
_net(), 
_base_lr(base_lr), 
_momentum(momentum), 
_gamma(gamma), 
_history(), 
_temp(), 
_iter(iter), 
_device_id(device_id), 
_out_layer(), _data_layer(),
_outFilename(outFilename)
{
	_counter = 0;	
	_data_buffer.clear();
}





void TrainNode::init_model()
{
#if !defined(CPU_ONLY)
	// Setup GPU - Make sure this is done in the thread the node will 
	// be running under. 
	//
	caffe::Caffe::SetDevice(_device_id);	
	caffe::Caffe::set_mode(caffe::Caffe::GPU);
	caffe::Caffe::DeviceQuery();
#else
	caffe::Caffe::set_mode(caffe::Caffe::CPU);
#endif

	// Initialize Net
	_net.reset(new caffe::Net<float>(_model_path.c_str(), caffe::TRAIN));

    // Initialize the history
	const std::vector<boost::shared_ptr<caffe::Blob<float> > >& net_params = this->_net->params();
	_history.clear();
	_temp.clear();

	for(int i = 0; i < net_params.size(); ++i) {

		const vector<int>& shape = net_params[i]->shape();
		_history.push_back(boost::shared_ptr<caffe::Blob<float> >(new caffe::Blob<float>(shape)));
		_temp.push_back(boost::shared_ptr<caffe::Blob<float> >(new caffe::Blob<float>(shape)));
	}

	_data_layer = 
			boost::static_pointer_cast<caffe::MemoryDataLayer<float>>(_net->layer_by_name("data"));
	_out_layer = _net->blob_by_name("prob");
}





void *TrainNode::run()
{
	int 	first_idx = 0;

	increment_threads();
	init_model();

	_runtimeStart = utils::get_time();


	while( true ) {

		copy_chunk_from_buffer(_data_buffer, _labels_buffer);

		if( first_idx + _batch_size <= _data_buffer.size() ) {
			cout << "Training " << 	_data_buffer.size() - first_idx << " samples" << endl;		
			// For each epoch feed the model with a mini-batch of samples
			first_idx = train_step(first_idx);

		} else {

			// Check if all input nodes have already finished
			bool is_all_done = true;
			for(vector<int>::size_type i=0; i < _in_edges.size(); i++) {

				if(!_in_edges.at(i)->is_in_node_done()) {
					is_all_done = false;
					break;
				}
			}

			// All input nodes have finished
			if( is_all_done ) {
				break;
			}
		}
	}


	if( check_finished() == true ) {
		
		for(int k=0; k < _iter-1; k++) {
			
			train_step(0);
		}

		cout << "******************" << endl 
			 << "TrainNode" << endl 
			 << "Run time: " << to_string(utils::get_time() - _runtimeStart) << endl
			 << "# of elements: " << to_string(_labels_buffer.size()) << endl 
			 << "******************" << endl;

		// Save model
		snapshot();

		// Notify it has finished
		for(vector<int>::size_type i=0; i < _out_edges.size(); i++) {
			_out_edges.at(i)->set_in_node_done();
		}
	}
	return NULL;
}





int TrainNode::train_step(int first_idx)
{

	int epochs = (_data_buffer.size() - first_idx)/_batch_size;
	cout << "Running " << epochs << " epochs for training" << endl;
	double start = utils::get_time();;

	for(int i = 0; i < epochs; i++) {

		// Split batch
		vector<cv::Mat> batch;
		vector<int> batch_labels;

		// Reseve space
		batch.reserve(_batch_size);
		batch.insert(batch.end(), _data_buffer.begin() + first_idx, 
					 _data_buffer.begin() + first_idx + _batch_size);
		batch_labels.reserve(_batch_size);
		batch_labels.insert(batch_labels.end(), _labels_buffer.begin() + first_idx, 
							_labels_buffer.begin() + first_idx + _batch_size);

		// Predict next batch
		cross_validate(batch, batch_labels);

		// Add matrices
		_data_layer->AddMatVector(batch, batch_labels);

		// Foward
		float loss;
		_net->ForwardPrefilled(&loss);

		// Backward
		_net->Backward();

		// Update
		compute_update_value();
		_net->Update();
		first_idx += _batch_size;
	}

	cout << "Traing took " << utils::get_time() - start << " for " 
		<< _data_buffer.size() << " elements" << endl;

	return first_idx;
}





void TrainNode::cross_validate(vector<cv::Mat> batch, vector<int> batch_labels)
{

	// Add matrices
	_data_layer->AddMatVector(batch, batch_labels);

	// Foward
	float loss;
	_net->ForwardPrefilled(&loss);

	const float* results = _out_layer->cpu_data();

	// Append outputs
	vector<int> predictions;
	for(int j=0; j < _out_layer->shape(0); j++) {
		
		int idx_max = 0;
		float max = results[j*_out_layer->shape(1) + 0];

		// Argmax
		for(int k=0; k < _out_layer->shape(1); k++) {

			if( results[j*_out_layer->shape(1) + k] > max ) {

				max = results[j*_out_layer->shape(1) + k];
				idx_max = k;
			}
		}
		predictions.push_back(idx_max);
	}

	// Compare labels
	int hit = 0;
	for(int k = 0; k < predictions.size(); k++) {

		if( predictions[k] == batch_labels[k] ){
			hit++;
		}
	}
	float acc = (float)hit/predictions.size();
}





void TrainNode::snapshot()
{
	caffe::NetParameter net_param;

	_net->ToProto(&net_param, true);

	std::cout << "Snapshotting to " << _outFilename << endl;
	caffe::WriteProtoToBinaryFile(net_param, _outFilename.c_str());
}





void TrainNode::compute_update_value()
{

	const vector<boost::shared_ptr<caffe::Blob<float> > >& net_params = _net->params();
	const vector<float>& net_params_lr = _net->params_lr();
	const vector<float>& net_params_weight_decay = _net->params_weight_decay();

  	// get parameters
	float rate = _base_lr;
	float momentum = _momentum;
	float weight_decay = _gamma;
	string regularization_type = "L2";

#if defined(CPU_ONLY)
	for (int param_id = 0; param_id < net_params.size(); ++param_id) {

		// Compute the value to history, and then copy them to the blob's diff.
		float local_rate = rate * net_params_lr[param_id];
		float local_decay = weight_decay * net_params_weight_decay[param_id];

		if( local_decay ) {
			if( regularization_type == "L2" ) {

				// add weight decay
				caffe::caffe_axpy(net_params[param_id]->count(), 
								  local_decay, 
								  net_params[param_id]->cpu_data(), 
								  net_params[param_id]->mutable_cpu_diff());
			} else if( regularization_type == "L1" ) {

				caffe::caffe_cpu_sign(net_params[param_id]->count(), 
									  net_params[param_id]->cpu_data(), 
									  _temp[param_id]->mutable_cpu_data());
				caffe::caffe_axpy(net_params[param_id]->count(), 
								  local_decay, 
								  _temp[param_id]->cpu_data(), 
								  net_params[param_id]->mutable_cpu_diff());
			} else {
				LOG(FATAL) << "Unknown regularization type: " << regularization_type;
			}
		}

		caffe::caffe_cpu_axpby(net_params[param_id]->count(), 
							   local_rate, 
							   net_params[param_id]->cpu_diff(), 
							   momentum, 
							   _history[param_id]->mutable_cpu_data());

		// copy
		caffe::caffe_copy(net_params[param_id]->count(), 
						  _history[param_id]->cpu_data(), 
						  net_params[param_id]->mutable_cpu_diff());
	}

#else

	for(int param_id = 0; param_id < net_params.size(); ++param_id) {

		// Compute the value to history, and then copy them to the blob's diff.
		float local_rate = rate * net_params_lr[param_id];
		float local_decay = weight_decay * net_params_weight_decay[param_id];

		if( local_decay ) {
			if( regularization_type == "L2" ) {

				// add weight decay
				caffe::caffe_gpu_axpy(net_params[param_id]->count(), 
									  local_decay, 
									  net_params[param_id]->gpu_data(), 
									  net_params[param_id]->mutable_gpu_diff());
			} else if( regularization_type == "L1" ) {

				caffe::caffe_gpu_sign(net_params[param_id]->count(), 
									  net_params[param_id]->gpu_data(),
									  _temp[param_id]->mutable_gpu_data());
				caffe::caffe_gpu_axpy(net_params[param_id]->count(), 
									  local_decay, 
									  _temp[param_id]->gpu_data(), 
									  net_params[param_id]->mutable_gpu_diff());
			} else {
				LOG(FATAL) << "Unknown regularization type: " << regularization_type;
			}
		}

		caffe::caffe_gpu_axpby(net_params[param_id]->count(), 
							   local_rate, 
							   net_params[param_id]->gpu_diff(), 
							   momentum, 
							   _history[param_id]->mutable_gpu_data());

		// copy
		caffe::caffe_copy(net_params[param_id]->count(),
						  _history[param_id]->gpu_data(),
						  net_params[param_id]->mutable_gpu_diff());
	}
#endif
}

