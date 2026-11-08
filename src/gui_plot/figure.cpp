#include "gui_plot/figure.h"

#include "gui_plot/plot_detail.h"  // detail:: export sizing + default names
#include "gui_plot/plot_manager.h" // the singleton: registration + save queues + uid
#include "gui_plot/plot_tools.h"   // theme_scope, theme_names, request_theme_editor + ImGui

#include <algorithm>
#include <cmath>
#include <cstdio>

// The figure base: common state (title/theme/visibility), the window chrome
// (render + the shared Save row), the cross-thread painter seam (body_painter),
// and registration with the GUI render loop (show). Concrete kinds fill
// paint_body / paint_controls / fill_snapshot and live in their own .cpp.

namespace coast::plot {

figure::figure(std::string title)
    : title_(std::move(title)), uid_(plot_manager::instance().next_uid())
{
}
figure::~figure() = default;

void figure::set_title(std::string t)
{
    std::scoped_lock lk(m_);
    title_ = std::move(t);
}
void figure::set_theme(std::string name)
{
    std::scoped_lock lk(m_);
    theme_ = std::move(name);
}
void figure::show()
{
    open_.store(true, std::memory_order_relaxed);
    // Register once. shared_from_this() is valid here because the caller holds a
    // shared_ptr (make_shared<...>()->show() keeps the temporary alive across the
    // call, and the registry then takes over ownership).
    if (!registered_.exchange(true)) plot_manager::instance().register_figure(shared_from_this());
}
void figure::close() { open_.store(false, std::memory_order_relaxed); }

void figure::save_png(std::string path, int width, int height)
{
    int   w = width, h = height;
    float sc = 1.0f;
    {
        std::scoped_lock lk(m_);
        if (w <= 0) w = last_w_ > 0.0f ? static_cast<int>(last_w_) : 800;
        if (h <= 0) h = last_h_ > 0.0f ? static_cast<int>(last_h_) : 600;
        sc = detail::export_scale(w, h, last_w_, last_h_);
    }
    plot_manager::instance().enqueue_capture(weak_from_this(), std::move(path), std::max(1, w), std::max(1, h), sc);
}
void figure::save_h5(std::string path)
{
    if (!is_saveable()) return;
    float rw = 0.0f, rh = 0.0f;
    {
        std::scoped_lock lk(m_);
        rw = last_w_;
        rh = last_h_;
    }
    plot_manager::instance().enqueue_h5(weak_from_this(), std::move(path), rw, rh);
}

bool figure::snapshot_h5(h5_save_data& out)
{
    std::scoped_lock lk(m_);
    out.title = title_;
    out.theme = theme_;
    return fill_snapshot(out);
}

painter figure::body_painter()
{
    auto self = shared_from_this();
    return [self](region r) {
        std::scoped_lock lk(self->m_);
        theme_scope      ts(self->theme_);
        self->paint_body(r);
    };
}

// Theme picker + "edit…" shortcut (GUI thread, m_ held). Shared by the plot
// kinds' paint_controls; image/canvas omit it. Ends with a SameLine so the
// caller's next widget (or the Save row) continues on the same line.
void figure::draw_theme_row()
{
    ImGui::SetNextItemWidth(130.0f);
    if (ImGui::BeginCombo("theme", theme_.c_str())) {
        for (const auto& n : theme_names()) {
            const bool sel = (theme_ == n);
            if (ImGui::Selectable(n.c_str(), sel)) theme_ = n;
            if (sel) ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }
    ImGui::SameLine();
    if (ImGui::SmallButton("edit…")) request_theme_editor();
    ImGui::SameLine();
}

// Save PNG… (all kinds) + Save H5… (saveable kinds) buttons + popups. GUI thread,
// NOT under m_ (the popups' Save routes through save_h5()/enqueue_capture, which
// lock as needed). Continues the controls row.
void figure::draw_save_row(const std::string& title)
{
    if (ImGui::SmallButton("Save PNG…")) {
        save_w_ = last_w_ > 0.0f ? static_cast<int>(last_w_) : 800;
        save_h_ = last_h_ > 0.0f ? static_cast<int>(last_h_) : 600;
        if (save_path_[0] == '\0')
            std::snprintf(save_path_, sizeof save_path_, "%s", detail::default_png_name(title).c_str());
        ImGui::OpenPopup("save_png");
    }
    if (ImGui::BeginPopup("save_png")) {
        const float aspect = (last_h_ > 0.0f) ? (last_w_ / last_h_) : 1.0f;
        ImGui::InputText("file", save_path_, sizeof save_path_);
        if (ImGui::InputInt("width", &save_w_) && save_keep_aspect_)
            save_h_ = std::max(1, static_cast<int>(std::lround(save_w_ / aspect)));
        if (ImGui::InputInt("height", &save_h_) && save_keep_aspect_)
            save_w_ = std::max(1, static_cast<int>(std::lround(save_h_ * aspect)));
        ImGui::Checkbox("keep aspect ratio", &save_keep_aspect_);
        if (ImGui::Button("Save")) {
            plot_manager::instance().enqueue_capture(weak_from_this(), save_path_, std::max(1, save_w_),
                                    std::max(1, save_h_),
                                    detail::export_scale(save_w_, save_h_, last_w_, last_h_));
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel")) ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
    }

    if (!is_saveable()) return;
    ImGui::SameLine();
    if (ImGui::SmallButton("Save H5…")) {
        if (save_h5_path_[0] == '\0')
            std::snprintf(save_h5_path_, sizeof save_h5_path_, "%s", detail::default_h5_name(title).c_str());
        ImGui::OpenPopup("save_h5");
    }
    if (ImGui::BeginPopup("save_h5")) {
        ImGui::InputText("file", save_h5_path_, sizeof save_h5_path_);
        if (ImGui::Button("Save")) {
            save_h5(save_h5_path_); // virtual: figure -> plot h5, canvas -> canvas h5
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel")) ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
    }
}

void figure::render()
{
    std::string title;
    {
        std::scoped_lock lk(m_);
        title = title_;
    }
    char id[128];
    std::snprintf(id, sizeof id, "%s###coastfig%llu", title.c_str(),
                  static_cast<unsigned long long>(uid_));

    bool win_open = true;
    ImGui::SetNextWindowSize(ImVec2(480, 320), ImGuiCond_FirstUseEver);
    if (ImGui::Begin(id, &win_open)) {
        // Interactive row: kind-specific widgets (under the lock; they mutate kind
        // state) then the common Save buttons (not under the lock).
        {
            std::scoped_lock lk(m_);
            paint_controls();
        }
        draw_save_row(title); // title snapshot taken under m_ above

        const ImVec2 avail = ImGui::GetContentRegionAvail();
        {
            std::scoped_lock lk(m_);
            last_w_ = avail.x;
            last_h_ = avail.y;
        }
        // Body: the painter self-locks + themes; run it OUTSIDE our lock.
        body_painter()(region{avail.x, avail.y});
    }
    ImGui::End();
    if (!win_open) open_.store(false, std::memory_order_relaxed);
}

} // namespace coast::plot
