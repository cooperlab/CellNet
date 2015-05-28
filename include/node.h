#ifndef _NODE_H
#define _NODE_H

#include <vector>
#include <string>

class Node {
	
	public:
		Node(std::string name, std::string id);
		virtual void run(){};
		virtual void set_target(cv::Mat target){};
		virtual std::vector<cv::Mat> get_output(){return null};
		std::string get_id(){return _id};
		void insert_in_edge(int id){_in_edges_ids.insert(id)};
		void insert_out_edge(int id){_out_edges_ids.insert(id)};
		std::vector<int> get_in_edges_ids(){return _in_edges_ids};
		std::vector<int> get_out_edges_ids(){return _out_edges_ids};
		
  	protected:
		bool _is_ready;
		bool _is_valid;
		std::string _id;
		std::string _name;
		std::vector<int> _in_edges_ids;
		std::vector<int> _out_edges_ids;
};
#endif