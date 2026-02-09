//
// Created for the COAST probe system
//

#ifndef COAST_OBSERVER_PANEL_H
#define COAST_OBSERVER_PANEL_H

namespace sim {
class env;
}

namespace gui_probe {

/// Panel that displays all active probe observers.
class observer_panel {
    sim::env* env_{nullptr};
    bool visible_{false};

public:
    observer_panel() = default;
    explicit observer_panel(sim::env* env);

    void set_env(sim::env* env) { env_ = env; }

    /// Render all active probe observers.
    void render();

    /// Update observers by draining probe values (call once per frame).
    void update();

    void show() { visible_ = true; }
    void hide() { visible_ = false; }
    void toggle() { visible_ = !visible_; }
    [[nodiscard]] bool is_visible() const { return visible_; }
};

} // namespace gui_probe

#endif // COAST_OBSERVER_PANEL_H
