#include "node.h"

Node::Node(std::string id): _is_ready(false), _is_valid(false), _id(id), _in_edges_ids(), _out_edges_ids(){}
std::string Node::get_id(){return _id;}
void Node::insert_in_edge(int id){_in_edges_ids.push_back(id);}
void Node::insert_out_edge(int id){_out_edges_ids.push_back(id);}
std::vector<int> Node::get_in_edges_ids(){return _in_edges_ids;}
std::vector<int> Node::get_out_edges_ids(){return _out_edges_ids;}