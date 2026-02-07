//
// Force-directed layout algorithm with Barnes-Hut optimization
//

#ifndef COAST_FORCE_LAYOUT_H
#define COAST_FORCE_LAYOUT_H

#include "layout_graph.h"

#include <memory>

namespace coast::layout {

/// Parameters for force-directed layout simulation
struct force_params {
    float repulsion = 5000.0f;       // Node-node repulsion strength
    float attraction = 0.01f;        // Edge attraction strength
    float damping = 0.85f;           // Velocity damping per step
    float min_distance = 1.0f;       // Minimum distance for force calc (avoid division by zero)
    float theta = 0.8f;              // Barnes-Hut approximation threshold (0 = exact, 1 = fast approx)
    float max_velocity = 50.0f;      // Maximum velocity magnitude
    float gravity = 0.0f;            // Pull towards center
    float gravity_center_x = 0.0f;   // Gravity center X
    float gravity_center_y = 0.0f;   // Gravity center Y
    int iterations_per_frame = 1;    // Steps per update call
};

/// Barnes-Hut quadtree node for O(n log n) force approximation
struct quad_node {
    float center_x = 0.0f;           // Bounding box center X
    float center_y = 0.0f;           // Bounding box center Y
    float half_size = 0.0f;          // Half the width/height of the bounding box
    float mass_center_x = 0.0f;      // Center of mass X
    float mass_center_y = 0.0f;      // Center of mass Y
    float total_mass = 0.0f;         // Total mass in this region
    int node_count = 0;              // Number of nodes in this region
    node_id_t single_node = invalid_node_id;  // If node_count == 1, this is the node
    std::array<std::unique_ptr<quad_node>, 4> children;  // NW, NE, SW, SE
};

/// Force-directed layout algorithm
class force_layout {
public:
    explicit force_layout(layout_graph& graph);
    ~force_layout();

    force_layout(const force_layout&) = delete;
    force_layout& operator=(const force_layout&) = delete;
    force_layout(force_layout&&) noexcept;
    force_layout& operator=(force_layout&&) noexcept;

    /// Set layout parameters
    void set_params(const force_params& params);

    /// Get current parameters
    [[nodiscard]] const force_params& params() const { return params_; }

    /// Run one simulation step
    void step();

    /// Run until energy drops below threshold or max iterations reached
    /// @return Number of iterations performed
    int settle(float energy_threshold = 0.1f, int max_iterations = 500);

    /// Get current kinetic energy (for convergence detection)
    [[nodiscard]] float kinetic_energy() const;

    /// Center graph around a point
    void center_at(float x, float y);

    /// Center graph in a region
    void center_in_bounds(float width, float height);

    /// Reset all velocities to zero
    void reset_velocities();

    /// Check if simulation has settled (low energy)
    [[nodiscard]] bool is_settled(float threshold = 0.1f) const;

private:
    layout_graph& graph_;
    force_params params_;

    // Barnes-Hut quadtree root
    std::unique_ptr<quad_node> tree_root_;

    // Internal methods
    void rebuild_quadtree();
    void insert_into_tree(quad_node* node, node_id_t id, float x, float y, float mass);
    void compute_mass_centers(quad_node* node);
    void compute_repulsion();
    void compute_repulsion_for_node(node_t& n, const quad_node* tree_node);
    void compute_attraction();
    void compute_gravity();
    void apply_velocities();
    void enforce_velocity_limits();
};

} // namespace coast::layout

#endif // COAST_FORCE_LAYOUT_H
