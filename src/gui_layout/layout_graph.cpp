//
// Graph data structure for force-directed layout
//

#include "gui_layout/layout_graph.h"

#include <algorithm>
#include <stdexcept>
#include <limits>

namespace coast::layout {

node_id_t layout_graph::add_node(float x, float y, float radius) {
    node_t n;
    n.id = next_id_++;
    n.x = x;
    n.y = y;
    n.radius = radius;

    id_to_index_[n.id] = nodes_.size();
    nodes_.push_back(std::move(n));

    return nodes_.back().id;
}

node_id_t layout_graph::add_node(const node_t& node) {
    node_t n = node;
    n.id = next_id_++;

    id_to_index_[n.id] = nodes_.size();
    nodes_.push_back(std::move(n));

    return nodes_.back().id;
}

void layout_graph::add_edge(node_id_t from, node_id_t to, float strength, float ideal_length) {
    if (!has_node(from) || !has_node(to)) {
        return;
    }

    edge_t e;
    e.from = from;
    e.to = to;
    e.strength = strength;
    e.ideal_length = ideal_length;

    edges_.push_back(e);
}

void layout_graph::remove_node(node_id_t id) {
    auto it = id_to_index_.find(id);
    if (it == id_to_index_.end()) {
        return;
    }

    std::size_t idx = it->second;

    // Remove all edges connected to this node
    std::erase_if(edges_, [id](const edge_t& e) {
        return e.from == id || e.to == id;
    });

    // Remove the node
    nodes_.erase(nodes_.begin() + static_cast<std::ptrdiff_t>(idx));
    id_to_index_.erase(it);

    // Update index map for nodes after the removed one
    for (std::size_t i = idx; i < nodes_.size(); ++i) {
        id_to_index_[nodes_[i].id] = i;
    }
}

void layout_graph::remove_edge(node_id_t from, node_id_t to) {
    std::erase_if(edges_, [from, to](const edge_t& e) {
        return e.from == from && e.to == to;
    });
}

void layout_graph::clear() {
    nodes_.clear();
    edges_.clear();
    id_to_index_.clear();
    next_id_ = 0;
}

node_t& layout_graph::node(node_id_t id) {
    auto it = id_to_index_.find(id);
    if (it == id_to_index_.end()) {
        throw std::out_of_range("Node ID not found");
    }
    return nodes_[it->second];
}

const node_t& layout_graph::node(node_id_t id) const {
    auto it = id_to_index_.find(id);
    if (it == id_to_index_.end()) {
        throw std::out_of_range("Node ID not found");
    }
    return nodes_[it->second];
}

bool layout_graph::has_node(node_id_t id) const {
    return id_to_index_.contains(id);
}

std::vector<const edge_t*> layout_graph::edges_from(node_id_t id) const {
    std::vector<const edge_t*> result;
    for (const auto& e : edges_) {
        if (e.from == id) {
            result.push_back(&e);
        }
    }
    return result;
}

std::vector<const edge_t*> layout_graph::edges_to(node_id_t id) const {
    std::vector<const edge_t*> result;
    for (const auto& e : edges_) {
        if (e.to == id) {
            result.push_back(&e);
        }
    }
    return result;
}

void layout_graph::set_position(node_id_t id, float x, float y) {
    auto it = id_to_index_.find(id);
    if (it != id_to_index_.end()) {
        nodes_[it->second].x = x;
        nodes_[it->second].y = y;
    }
}

void layout_graph::set_fixed(node_id_t id, bool fixed) {
    auto it = id_to_index_.find(id);
    if (it != id_to_index_.end()) {
        nodes_[it->second].fixed = fixed;
    }
}

void layout_graph::set_user_data(node_id_t id, std::any data) {
    auto it = id_to_index_.find(id);
    if (it != id_to_index_.end()) {
        nodes_[it->second].user_data = std::move(data);
    }
}

std::array<float, 4> layout_graph::bounds() const {
    if (nodes_.empty()) {
        return {0.0f, 0.0f, 0.0f, 0.0f};
    }

    float min_x = std::numeric_limits<float>::max();
    float min_y = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float max_y = std::numeric_limits<float>::lowest();

    for (const auto& n : nodes_) {
        min_x = std::min(min_x, n.x - n.radius);
        min_y = std::min(min_y, n.y - n.radius);
        max_x = std::max(max_x, n.x + n.radius);
        max_y = std::max(max_y, n.y + n.radius);
    }

    return {min_x, min_y, max_x, max_y};
}

std::pair<float, float> layout_graph::center() const {
    if (nodes_.empty()) {
        return {0.0f, 0.0f};
    }

    float sum_x = 0.0f;
    float sum_y = 0.0f;

    for (const auto& n : nodes_) {
        sum_x += n.x;
        sum_y += n.y;
    }

    float inv_count = 1.0f / static_cast<float>(nodes_.size());
    return {sum_x * inv_count, sum_y * inv_count};
}

} // namespace coast::layout
