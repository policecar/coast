//
// Force-directed layout algorithm with Barnes-Hut optimization
//

#include "gui_layout/force_layout.h"

#include <cmath>
#include <algorithm>

namespace coast::layout {

force_layout::force_layout(layout_graph& graph)
    : graph_(graph) {
}

force_layout::~force_layout() = default;

force_layout::force_layout(force_layout&& other) noexcept
    : graph_(other.graph_)
    , params_(other.params_)
    , tree_root_(std::move(other.tree_root_)) {
}

force_layout& force_layout::operator=(force_layout&& other) noexcept {
    if (this != &other) {
        // Cannot reassign reference, just copy params and tree
        params_ = other.params_;
        tree_root_ = std::move(other.tree_root_);
    }
    return *this;
}

void force_layout::set_params(const force_params& params) {
    params_ = params;
}

void force_layout::step() {
    for (int i = 0; i < params_.iterations_per_frame; ++i) {
        rebuild_quadtree();
        compute_repulsion();
        compute_attraction();
        compute_gravity();
        enforce_velocity_limits();
        apply_velocities();
    }
}

int force_layout::settle(float energy_threshold, int max_iterations) {
    int iterations = 0;
    while (iterations < max_iterations) {
        step();
        ++iterations;
        if (kinetic_energy() < energy_threshold) {
            break;
        }
    }
    return iterations;
}

float force_layout::kinetic_energy() const {
    float energy = 0.0f;
    for (const auto& n : graph_.nodes()) {
        if (!n.fixed) {
            energy += n.mass * (n.vx * n.vx + n.vy * n.vy);
        }
    }
    return energy * 0.5f;
}

void force_layout::center_at(float x, float y) {
    auto [cx, cy] = graph_.center();
    float dx = x - cx;
    float dy = y - cy;

    for (auto& n : graph_.nodes()) {
        n.x += dx;
        n.y += dy;
    }
}

void force_layout::center_in_bounds(float width, float height) {
    center_at(width * 0.5f, height * 0.5f);
}

void force_layout::reset_velocities() {
    for (auto& n : graph_.nodes()) {
        n.vx = 0.0f;
        n.vy = 0.0f;
    }
}

bool force_layout::is_settled(float threshold) const {
    return kinetic_energy() < threshold;
}

void force_layout::rebuild_quadtree() {
    tree_root_.reset();

    if (graph_.node_count() == 0) {
        return;
    }

    // Find bounds of all nodes
    auto [min_x, min_y, max_x, max_y] = graph_.bounds();

    // Add some padding
    float padding = 100.0f;
    min_x -= padding;
    min_y -= padding;
    max_x += padding;
    max_y += padding;

    // Make it square
    float width = max_x - min_x;
    float height = max_y - min_y;
    float size = std::max(width, height);

    // Create root node
    tree_root_ = std::make_unique<quad_node>();
    tree_root_->center_x = min_x + size * 0.5f;
    tree_root_->center_y = min_y + size * 0.5f;
    tree_root_->half_size = size * 0.5f;

    // Insert all nodes
    for (const auto& n : graph_.nodes()) {
        insert_into_tree(tree_root_.get(), n.id, n.x, n.y, n.mass);
    }

    // Compute mass centers
    compute_mass_centers(tree_root_.get());
}

void force_layout::insert_into_tree(quad_node* qn, node_id_t id, float x, float y, float mass) {
    // If this quad node is empty, just store the node here
    if (qn->node_count == 0) {
        qn->single_node = id;
        qn->mass_center_x = x;
        qn->mass_center_y = y;
        qn->total_mass = mass;
        qn->node_count = 1;
        return;
    }

    // If we have exactly one node, we need to subdivide and move it
    if (qn->node_count == 1 && qn->single_node != invalid_node_id) {
        // Move existing node to appropriate child
        node_id_t existing_id = qn->single_node;
        float existing_x = qn->mass_center_x;
        float existing_y = qn->mass_center_y;
        float existing_mass = qn->total_mass;

        qn->single_node = invalid_node_id;

        // Determine which quadrant
        int existing_quad = (existing_x > qn->center_x ? 1 : 0) + (existing_y > qn->center_y ? 2 : 0);

        // Create child if needed
        if (!qn->children[existing_quad]) {
            qn->children[existing_quad] = std::make_unique<quad_node>();
            float new_half = qn->half_size * 0.5f;
            float dx = (existing_quad & 1) ? new_half : -new_half;
            float dy = (existing_quad & 2) ? new_half : -new_half;
            qn->children[existing_quad]->center_x = qn->center_x + dx;
            qn->children[existing_quad]->center_y = qn->center_y + dy;
            qn->children[existing_quad]->half_size = new_half;
        }

        insert_into_tree(qn->children[existing_quad].get(), existing_id, existing_x, existing_y, existing_mass);
    }

    // Determine which quadrant for new node
    int quad = (x > qn->center_x ? 1 : 0) + (y > qn->center_y ? 2 : 0);

    // Create child if needed
    if (!qn->children[quad]) {
        qn->children[quad] = std::make_unique<quad_node>();
        float new_half = qn->half_size * 0.5f;
        float dx = (quad & 1) ? new_half : -new_half;
        float dy = (quad & 2) ? new_half : -new_half;
        qn->children[quad]->center_x = qn->center_x + dx;
        qn->children[quad]->center_y = qn->center_y + dy;
        qn->children[quad]->half_size = new_half;
    }

    insert_into_tree(qn->children[quad].get(), id, x, y, mass);
    ++qn->node_count;
}

void force_layout::compute_mass_centers(quad_node* qn) {
    if (!qn || qn->node_count == 0) {
        return;
    }

    // Leaf node with single body
    if (qn->node_count == 1 && qn->single_node != invalid_node_id) {
        return; // Mass center already set during insertion
    }

    // Internal node - compute from children
    float total_mass = 0.0f;
    float weighted_x = 0.0f;
    float weighted_y = 0.0f;

    for (auto& child : qn->children) {
        if (child) {
            compute_mass_centers(child.get());
            total_mass += child->total_mass;
            weighted_x += child->total_mass * child->mass_center_x;
            weighted_y += child->total_mass * child->mass_center_y;
        }
    }

    if (total_mass > 0.0f) {
        float inv_mass = 1.0f / total_mass;
        qn->total_mass = total_mass;
        qn->mass_center_x = weighted_x * inv_mass;
        qn->mass_center_y = weighted_y * inv_mass;
    }
}

void force_layout::compute_repulsion() {
    if (!tree_root_) {
        return;
    }

    for (auto& n : graph_.nodes()) {
        if (!n.fixed) {
            compute_repulsion_for_node(n, tree_root_.get());
        }
    }
}

void force_layout::compute_repulsion_for_node(node_t& n, const quad_node* qn) {
    if (!qn || qn->node_count == 0) {
        return;
    }

    float dx = qn->mass_center_x - n.x;
    float dy = qn->mass_center_y - n.y;
    float dist_sq = dx * dx + dy * dy;
    float dist = std::sqrt(std::max(dist_sq, params_.min_distance * params_.min_distance));

    // Check if we should use Barnes-Hut approximation
    float size = qn->half_size * 2.0f;
    float ratio = size / dist;

    if (qn->node_count == 1 || ratio < params_.theta) {
        // Use approximation or this is a single node
        if (qn->single_node != n.id) {
            // Apply repulsive force (inverse square law)
            float force = -params_.repulsion * n.mass * qn->total_mass / (dist_sq + params_.min_distance);
            float inv_dist = 1.0f / dist;
            n.vx += force * dx * inv_dist;
            n.vy += force * dy * inv_dist;
        }
    } else {
        // Recurse into children
        for (const auto& child : qn->children) {
            if (child) {
                compute_repulsion_for_node(n, child.get());
            }
        }
    }
}

void force_layout::compute_attraction() {
    for (const auto& edge : graph_.edges()) {
        if (!graph_.has_node(edge.from) || !graph_.has_node(edge.to)) {
            continue;
        }

        auto& n1 = graph_.node(edge.from);
        auto& n2 = graph_.node(edge.to);

        float dx = n2.x - n1.x;
        float dy = n2.y - n1.y;
        float dist = std::sqrt(dx * dx + dy * dy);

        if (dist < params_.min_distance) {
            continue;
        }

        // Spring force: pull towards ideal length
        float delta = dist - edge.ideal_length;
        float force = params_.attraction * edge.strength * delta;
        float inv_dist = 1.0f / dist;
        float fx = force * dx * inv_dist;
        float fy = force * dy * inv_dist;

        if (!n1.fixed) {
            n1.vx += fx;
            n1.vy += fy;
        }

        if (!n2.fixed) {
            n2.vx -= fx;
            n2.vy -= fy;
        }
    }
}

void force_layout::compute_gravity() {
    if (params_.gravity <= 0.0f) {
        return;
    }

    for (auto& n : graph_.nodes()) {
        if (!n.fixed) {
            float dx = params_.gravity_center_x - n.x;
            float dy = params_.gravity_center_y - n.y;
            n.vx += params_.gravity * dx;
            n.vy += params_.gravity * dy;
        }
    }
}

void force_layout::enforce_velocity_limits() {
    float max_vel_sq = params_.max_velocity * params_.max_velocity;

    for (auto& n : graph_.nodes()) {
        if (!n.fixed) {
            float vel_sq = n.vx * n.vx + n.vy * n.vy;
            if (vel_sq > max_vel_sq) {
                float scale = params_.max_velocity / std::sqrt(vel_sq);
                n.vx *= scale;
                n.vy *= scale;
            }
        }
    }
}

void force_layout::apply_velocities() {
    for (auto& n : graph_.nodes()) {
        if (!n.fixed) {
            n.x += n.vx;
            n.y += n.vy;
            n.vx *= params_.damping;
            n.vy *= params_.damping;
        }
    }
}

} // namespace coast::layout
