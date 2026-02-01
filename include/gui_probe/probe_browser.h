//
// Created for the COAST probe system
//

#ifndef COAST_PROBE_BROWSER_H
#define COAST_PROBE_BROWSER_H

#include <string>

namespace sim {
class env;
struct io_entity;
}

namespace core {
struct structure_entry;
}

namespace probe {
struct probe_instance;
class entity_probe_set;
}

namespace gui_probe {

/// Tree view browser for discovering and activating probes.
class probe_browser {
    sim::env* env_{nullptr};
    bool visible_{false};

    // Selected probe for activation controls
    probe::probe_instance* selected_probe_{nullptr};

    // Activation settings
    int decimation_{1};

public:
    probe_browser() = default;
    explicit probe_browser(sim::env* env);

    void set_env(sim::env* env) { env_ = env; }

    void render();

    void show() { visible_ = true; }
    void hide() { visible_ = false; }
    void toggle() { visible_ = !visible_; }
    [[nodiscard]] bool is_visible() const { return visible_; }

private:
    void render_entity(sim::io_entity* entity);
    void render_structure_tree(core::structure_entry* entry, probe::entity_probe_set& probes);
    void render_probe_point(core::structure_entry* entry, const std::string& probe_name,
                            probe::entity_probe_set& probes);
    void render_activation_popup();
};

} // namespace gui_probe

#endif // COAST_PROBE_BROWSER_H
