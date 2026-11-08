#include "gui_plot/image.h"

#include "gui_plot/plot_manager.h"      // post_gui_task (GUI-thread disposal)
#include "gui_vis/core/gl_texture.h"    // full gl_texture (+ color_t)
#include "imgui.h"
#include "stb_image.h"

#include <algorithm>
#include <memory>
#include <string>

namespace coast::plot {

image::image(std::string title) : figure(std::move(title)) {}
image::~image()
{
    // The last image ref may be dropped on the console thread; freeing tex_ inline
    // would call glDeleteTextures off the GUI thread. Hand it to the GUI-thread
    // task hook — the closure carries the texture and frees it (its destructor)
    // when run on the GUI thread. Null if never rendered => nothing to post.
    // (free_gl() already resets tex_ for figures torn down via the registry.)
    if (!tex_) return;
    // A destructor must not throw: post_gui_task allocates (mutex + push_back), so
    // guard the enqueue — a bad_alloc escaping here would terminate the process
    // (cf. the ~fd_capture fix in session.cpp). On failure the texture is freed as
    // the closure unwinds; off-thread GL under OOM is the lesser evil than a crash.
    try {
        plot_manager::instance().post_gui_task([tex = std::shared_ptr<gl_texture>(std::move(tex_))] {});
    } catch (...) { // NOLINT(bugprone-empty-catch)
    }
}

void image::set_png(std::string path)
{
    std::scoped_lock lk(m_);
    src_   = source::png;
    path_  = std::move(path);
    dirty_ = true;
}
void image::set_rgba(std::span<const std::uint8_t> pixels, int width, int height)
{
    std::scoped_lock lk(m_);
    src_   = source::rgba;
    pixels_.assign(pixels.begin(), pixels.end());
    w_     = width;
    h_     = height;
    dirty_ = true;
}
void image::free_gl()
{
    std::scoped_lock lk(m_);
    tex_.reset();
}

void image::reload_locked()
{
    dirty_ = false;
    error_.clear();
    if (src_ == source::png) {
        int            w = 0, h = 0, ch = 0;
        unsigned char* data = stbi_load(path_.c_str(), &w, &h, &ch, 4); // force RGBA
        if (data == nullptr) {
            const char* why = stbi_failure_reason();
            error_          = std::string("cannot load '") + path_ + "': " + (why ? why : "?");
            return;
        }
        tex_ = std::make_unique<gl_texture>(w, h, reinterpret_cast<const color_t*>(data));
        stbi_image_free(data);
    } else if (src_ == source::rgba) {
        if (w_ > 0 && h_ > 0 && pixels_.size() >= static_cast<std::size_t>(w_) * h_ * 4) {
            tex_ = std::make_unique<gl_texture>(w_, h_,
                                                reinterpret_cast<const color_t*>(pixels_.data()));
        } else {
            error_ = "set_rgba: pixel buffer too small for width*height*4";
        }
    }
}

void image::paint_body(region rg)
{
    if (dirty_) reload_locked();
    if (!error_.empty()) {
        ImGui::TextColored(ImVec4(0.95f, 0.4f, 0.4f, 1.0f), "%s", error_.c_str());
    } else if (tex_ && tex_->is_valid()) {
        const float tw    = static_cast<float>(tex_->get_width());
        const float th    = static_cast<float>(tex_->get_height());
        float       scale = std::min(rg.w / tw, rg.h / th);
        if (!(scale > 0.0f)) scale = 1.0f;
        ImGui::Image(tex_->as_imgui_texture(), ImVec2(tw * scale, th * scale));
    } else {
        ImGui::TextDisabled("no image — call set_png() or set_rgba()");
    }
}

} // namespace coast::plot
