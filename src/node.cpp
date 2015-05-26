#include "node.h"

Node::Node(std::string name, boost::uuids::uuid id): _name(name), _id(id), _in_edges_ids(), _out_edges_ids(), _is_ready(false), _is_valid(false){
}