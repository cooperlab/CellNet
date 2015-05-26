#ifndef _NODE_H
#define _NODE_H

#include <vector>
#include <string>
#include <boost/uuid/uuid.hpp>

class Node {
	
	public:
		Node(std::string name, boost::uuids::uuid id);
		virtual void run(){};

  	protected:
		bool _is_ready;
		bool _is_valid;
		boost::uuids::uuid _id;
		std::string _name;
		std::vector<int> _in_edges_ids;
		std::vector<int> _out_edges_ids;
};
#endif