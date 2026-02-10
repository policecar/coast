//
// Graph data structure for force-directed layout
//

#ifndef COAST_LAYOUT_GRAPH_H
#define COAST_LAYOUT_GRAPH_H

#include <cstdint>
#include <vector>
#include <span>
#include <any>
#include <array>
#include <limits>
#include <unordered_map>

namespace coast::layout {

using node_id_t = std::uint32_t;

/// Invalid node ID sentinel value
inline constexpr node_id_t invalid_node_id = std::numeric_limits<node_id_t>::max();

/// Node in the layout graph
struct node_t {
    node_id_t id = invalid_node_id;
    float x = 0.0f;              // Current position X
    float y = 0.0f;              // Current position Y
    float vx = 0.0f;             // Velocity X (for simulation)
    float vy = 0.0f;             // Velocity Y (for simulation)
    float radius = 20.0f;        // Collision/visual radius
    float mass = 1.0f;           // Mass for force calculations
    bool fixed = false;          // If true, position is locked
    std::any user_data;          // Arbitrary payload for renderer
};

/// Edge in the layout graph
struct edge_t {
    node_id_t from = invalid_node_id;
    node_id_t to = invalid_node_id;
    float strength = 1.0f;       // Affects attractive force
    float ideal_length = 100.0f; // Preferred edge length
};

/// Graph data structure for layout algorithms
class layout_graph {
public:
    layout_graph() = default;

    /// Add a node at the specified position
    /// @return The ID of the newly created node
    node_id_t add_node(float x, float y, float radius = 20.0f);

    /// Add a node with full configuration
    node_id_t add_node(const node_t& node);

    /// Add a directed edge from -> to
    void add_edge(node_id_t from, node_id_t to, float strength = 1.0f, float ideal_length = 100.0f);

    /// Remove a node and all its connected edges
    void remove_node(node_id_t id);

    /// Remove an edge
    void remove_edge(node_id_t from, node_id_t to);

    /// Clear all nodes and edges
    void clear();

    /// Get mutable reference to a node
    node_t& node(node_id_t id);

    /// Get const reference to a node
    [[nodiscard]] const node_t& node(node_id_t id) const;

    /// Check if a node exists
    [[nodiscard]] bool has_node(node_id_t id) const;

    /// Get all nodes
    [[nodiscard]] std::span<const node_t> nodes() const { return nodes_; }

    /// Get mutable access to all nodes
    std::span<node_t> nodes() { return nodes_; }

    /// Get all edges
    [[nodiscard]] std::span<const edge_t> edges() const { return edges_; }

    /// Get edges originating from a specific node
    [[nodiscard]] std::vector<const edge_t*> edges_from(node_id_t id) const;

    /// Get edges terminating at a specific node
    [[nodiscard]] std::vector<const edge_t*> edges_to(node_id_t id) const;

    /// Set node position
    void set_position(node_id_t id, float x, float y);

    /// Set node fixed state
    void set_fixed(node_id_t id, bool fixed);

    /// Set node user data
    void set_user_data(node_id_t id, std::any data);

    /// Get number of nodes
    [[nodiscard]] std::size_t node_count() const { return nodes_.size(); }

    /// Get number of edges
    [[nodiscard]] std::size_t edge_count() const { return edges_.size(); }

    /// Get the bounding box of all nodes
    /// @return {min_x, min_y, max_x, max_y}
    [[nodiscard]] std::array<float, 4> bounds() const;

    /// Get the center of all nodes
    [[nodiscard]] std::pair<float, float> center() const;

private:
    std::vector<node_t> nodes_;
    std::vector<edge_t> edges_;
    std::unordered_map<node_id_t, std::size_t> id_to_index_;
    node_id_t next_id_ = 0;
};

} // namespace coast::layout

#endif // COAST_LAYOUT_GRAPH_H
