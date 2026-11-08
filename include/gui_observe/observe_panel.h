//
// Created for the COAST observe system
//

#ifndef COAST_OBSERVE_PANEL_H
#define COAST_OBSERVE_PANEL_H

namespace gui_observe {

/// Panel that lists OBSERVE groups, allows editing of their step ranges and
/// combine mode, and exposes manual START / STOP / SINGLE_CAPTURE triggers.
///
/// Mirrors the shape of `gui_probe::observer_panel`. The panel pulls all of
/// its state directly from `observe::observe_registry::instance()` and routes
/// every mutation through the registry's `with_group_locked` helper so that
/// concurrent map operations remain safe.
class observe_panel {
    bool visible_{false};
    char new_group_buf_[128]{};

public:
    observe_panel() = default;

    /// Render the panel (no-op when not visible).
    void render();

    void show()    { visible_ = true; }
    void hide()    { visible_ = false; }
    void toggle()  { visible_ = !visible_; }
    [[nodiscard]] bool is_visible() const { return visible_; }
};

} // namespace gui_observe

#endif // COAST_OBSERVE_PANEL_H
