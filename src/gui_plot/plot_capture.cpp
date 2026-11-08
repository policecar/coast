#include "gui_plot/plot_manager.h" // the class whose capture methods this file defines
#include "gui_plot/plot_detail.h"  // detail:: export sizing helpers (defined here)
#include "gui_plot/plot_h5.h"      // write_plot_h5 / write_canvas_h5 (worker-thread writes)

#include "gui_plot/canvas.h" // dynamic_cast<canvas*> + snapshot_canvas in flush
#include "imgui.h"
#include "implot.h"

#include "backends/imgui_impl_opengl3.h"
#include <SDL.h>
#include <SDL_opengl.h>

#include "hdf5/executor.h"

#include "diag/diag.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <algorithm>
#include <cctype>
#include <cfloat> // FLT_MAX (parking the cursor for off-screen captures)
#include <cmath>
#include <filesystem>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

// Off-screen PNG export + the plot_manager capture methods. save_png()/save_h5()
// (on any thread) only enqueue a request onto plot_manager; flush_captures()
// services them on the GUI thread, rendering each figure into an FBO of the
// requested size (so exports can exceed the on-screen window). HDF5 writes are
// handed to the executor worker thread as plain-data payloads.

namespace coast::plot {

namespace {

std::string slug(const std::string& title, const char* ext)
{
    std::string s;
    for (char c : title)
        s += (std::isalnum(static_cast<unsigned char>(c)) != 0) ? c : '_';
    if (s.empty()) s = "plot";
    return s + ext;
}

// -- GL 3.0 framebuffer entry points, loaded lazily via SDL -----------------
PFNGLGENFRAMEBUFFERSPROC        p_glGenFramebuffers        = nullptr;
PFNGLBINDFRAMEBUFFERPROC        p_glBindFramebuffer        = nullptr;
PFNGLFRAMEBUFFERTEXTURE2DPROC   p_glFramebufferTexture2D   = nullptr;
PFNGLDELETEFRAMEBUFFERSPROC     p_glDeleteFramebuffers     = nullptr;
PFNGLCHECKFRAMEBUFFERSTATUSPROC p_glCheckFramebufferStatus = nullptr;

bool load_fbo_fns()
{
    if (p_glGenFramebuffers != nullptr) return true;
    p_glGenFramebuffers = reinterpret_cast<PFNGLGENFRAMEBUFFERSPROC>(
        SDL_GL_GetProcAddress("glGenFramebuffers"));
    p_glBindFramebuffer = reinterpret_cast<PFNGLBINDFRAMEBUFFERPROC>(
        SDL_GL_GetProcAddress("glBindFramebuffer"));
    p_glFramebufferTexture2D = reinterpret_cast<PFNGLFRAMEBUFFERTEXTURE2DPROC>(
        SDL_GL_GetProcAddress("glFramebufferTexture2D"));
    p_glDeleteFramebuffers = reinterpret_cast<PFNGLDELETEFRAMEBUFFERSPROC>(
        SDL_GL_GetProcAddress("glDeleteFramebuffers"));
    p_glCheckFramebufferStatus = reinterpret_cast<PFNGLCHECKFRAMEBUFFERSTATUSPROC>(
        SDL_GL_GetProcAddress("glCheckFramebufferStatus"));
    return p_glGenFramebuffers && p_glBindFramebuffer && p_glFramebufferTexture2D &&
           p_glDeleteFramebuffers && p_glCheckFramebufferStatus;
}

} // namespace

// Scale the pixel-size fields of an ImPlotStyle by f. Declared in plot_detail.h;
// also used by canvas's zoom, so it lives outside the anonymous namespace.
void detail::scale_implot_style(ImPlotStyle& s, float f)
{
    auto sv = [f](ImVec2& v) {
        v.x *= f;
        v.y *= f;
    };
    s.PlotBorderSize *= f;
    sv(s.MajorTickLen);
    sv(s.MinorTickLen);
    sv(s.MajorTickSize);
    sv(s.MinorTickSize);
    sv(s.MajorGridSize);
    sv(s.MinorGridSize);
    sv(s.PlotPadding);
    sv(s.LabelPadding);
    sv(s.LegendPadding);
    sv(s.LegendInnerPadding);
    sv(s.LegendSpacing);
    sv(s.MousePosPadding);
    sv(s.AnnotationPadding);
    sv(s.PlotDefaultSize);
    sv(s.PlotMinSize);
    s.DigitalPadding *= f;
    s.DigitalSpacing *= f;
}

namespace {

// Render one figure's body into an off-screen FBO of size W×H and return the
// RGBA8 pixels (top-down, opaque), or empty on failure. Runs its OWN ImGui frame
// (called OUTSIDE the main NewFrame…Render span — see flush_captures). GUI thread.
std::vector<unsigned char> capture_figure_rgba(figure& fig, int W, int H, float scale)
{
    if (!load_fbo_fns()) {
        COAST_ERROR("plot", "capture: framebuffer objects unavailable");
        return {};
    }

    ImGuiIO&     io         = ImGui::GetIO();
    const ImVec2 saved_disp = io.DisplaySize;

    // An off-screen capture must never appear hovered or clicked: it builds a real
    // frame, and io.MousePos still holds the live cursor. The interactive=false
    // region below is the primary gate for view-state mutation (io parking cannot
    // cover an in-progress drag — the buttons stay down); parking the cursor and
    // wheel additionally keeps hover-dependent chrome (ImPlot tooltips, legend
    // highlights) out of the exported image.
    const ImVec2 saved_mouse = io.MousePos;
    const float  saved_wheel = io.MouseWheel;
    io.MousePos              = ImVec2(-FLT_MAX, -FLT_MAX);
    io.MouseWheel            = 0.0f;

    ImGuiStyle&       gstyle  = ImGui::GetStyle();
    ImPlotStyle&      pstyle  = ImPlot::GetStyle();
    const ImGuiStyle  saved_g = gstyle;
    const ImPlotStyle saved_p = pstyle;
    if (scale != 1.0f) {
        gstyle.ScaleAllSizes(scale);
        detail::scale_implot_style(pstyle, scale);
        // Multiplicative, matching canvas::paint_body: an export inherits any
        // global font scale instead of silently dropping it. Restored by the
        // wholesale gstyle = saved_g below (FontScaleMain is style state).
        gstyle.FontScaleMain *= scale; // ScaleAllSizes is geometry-only
    }

    constexpr ImGuiWindowFlags flags =
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse |
        ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoSavedSettings |
        ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNav;

    // Build one self-contained off-screen frame of the figure body (NewFrame …
    // Render). The caller decides whether to rasterize the resulting draw data.
    auto build_frame = [&] {
        ImGui_ImplOpenGL3_NewFrame();
        io.DisplaySize = ImVec2(static_cast<float>(W), static_cast<float>(H));
        io.DeltaTime   = 1.0f / 60.0f;
        ImGui::NewFrame();
        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ImVec2(static_cast<float>(W), static_cast<float>(H)));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        if (ImGui::Begin("##coast_export", nullptr, flags))
            fig.body_painter()(
                region{static_cast<float>(W), static_cast<float>(H), /*interactive=*/false});
        ImGui::End();
        ImGui::PopStyleVar(3);
        ImGui::Render();
    };

    // ImPlot applies an AutoFit axis range the frame AFTER the data is submitted;
    // the items in the current frame still render with the previous/default range.
    // This off-screen capture is one-shot, so a single frame would rasterize the
    // pre-fit (default 0..1) plot — the "first save has the wrong scale, second
    // save is fine" bug. Build one throwaway priming frame so the fit settles into
    // ImPlot's per-plot state, then build + rasterize the second, fitted frame.
    build_frame(); // prime: draw data discarded
    build_frame(); // capture: rasterized below via ImGui::GetDrawData()

    GLuint tex = 0, fbo = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    p_glGenFramebuffers(1, &fbo);
    p_glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    p_glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);

    std::vector<unsigned char> flipped;
    if (p_glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE) {
        glViewport(0, 0, W, H);
        const ImVec4 bg = ImGui::GetStyle().Colors[ImGuiCol_WindowBg];
        glClearColor(bg.x, bg.y, bg.z, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        const int                  stride = W * 4;
        std::vector<unsigned char> px(static_cast<std::size_t>(stride) * H);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadPixels(0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
        glPixelStorei(GL_PACK_ALIGNMENT, 4);

        for (std::size_t i = 3; i < px.size(); i += 4) px[i] = 255;

        flipped.resize(px.size());
        for (int r = 0; r < H; ++r)
            std::copy_n(px.data() + static_cast<std::size_t>(H - 1 - r) * stride, stride,
                        flipped.data() + static_cast<std::size_t>(r) * stride);
    } else {
        COAST_ERROR("plot", "capture: framebuffer incomplete ({}x{})", W, H);
    }

    p_glBindFramebuffer(GL_FRAMEBUFFER, 0);
    p_glDeleteFramebuffers(1, &fbo);
    glBindTexture(GL_TEXTURE_2D, 0);
    glDeleteTextures(1, &tex);

    io.DisplaySize = saved_disp;
    gstyle         = saved_g;
    pstyle         = saved_p;
    io.MousePos    = saved_mouse;
    io.MouseWheel  = saved_wheel;
    return flipped;
}

void write_figure_png(figure& fig, const std::string& path, int W, int H, float scale)
{
    std::vector<unsigned char> px = capture_figure_rgba(fig, W, H, scale);
    if (px.empty()) return;
    if (stbi_write_png(path.c_str(), W, H, 4, px.data(), W * 4) == 0)
        COAST_ERROR("plot", "save_png: write failed: {}", path);
}

std::vector<unsigned char> encode_figure_png(figure& fig, int W, int H, float scale)
{
    std::vector<unsigned char> px = capture_figure_rgba(fig, W, H, scale);
    if (px.empty()) return {};
    std::vector<unsigned char> out;
    stbi_write_png_to_func(
        [](void* ctx, void* data, int len) {
            auto* v = static_cast<std::vector<unsigned char>*>(ctx);
            auto* b = static_cast<unsigned char*>(data);
            v->insert(v->end(), b, b + len);
        },
        &out, W, H, 4, px.data(), W * 4);
    return out;
}

} // namespace

// -- stateless export-sizing helpers (declared in plot_detail.h) ------------
namespace detail {

// Uniform DPI scale so an enlarged export looks like the on-screen plot rather
// than px-native (tiny) text. Smaller of the two ratios so nothing over-scales
// past the frame; clamp to a sane band. rw/rh = on-screen body size.
float export_scale(int w, int h, float rw, float rh)
{
    if (rw <= 0.0f || rh <= 0.0f) return 1.0f;
    return std::clamp(std::min(static_cast<float>(w) / rw, static_cast<float>(h) / rh), 0.5f, 8.0f);
}
// Startup-set default export directory (see set_export_dir). "." keeps the
// historical CWD behavior; the dialogs seed <export_dir>/<slug> and let the user
// edit from there.
std::string& export_dir_storage() { static std::string d = "."; return d; }

std::string with_export_dir(const std::string& name)
{
    const std::string& d = export_dir_storage();
    if (d.empty() || d == ".") return name;   // CWD default: bare filename
    return (std::filesystem::path(d) / name).string();
}

std::string default_png_name(const std::string& title) { return with_export_dir(slug(title, ".png")); }
std::string default_h5_name(const std::string& title) { return with_export_dir(slug(title, ".h5")); }

} // namespace detail

// Public startup setter (declared in plot_tools.h alongside set_theme_file). The
// storage lives in detail (this TU); a non-empty override wins.
void set_export_dir(std::string dir)
{
    if (!dir.empty()) detail::export_dir_storage() = std::move(dir);
}

// -- plot_manager capture methods -------------------------------------------

void plot_manager::enqueue_capture(std::weak_ptr<figure> f, std::string path, int w, int h, float scale)
{
    std::scoped_lock lk(cap_m_);
    caps_.push_back({std::move(f), std::move(path), w, h, scale});
}
void plot_manager::enqueue_h5(std::weak_ptr<figure> f, std::string path, float ref_w, float ref_h)
{
    std::scoped_lock lk(h5_m_);
    h5s_.push_back({std::move(f), std::move(path), ref_w, ref_h});
}
void plot_manager::enqueue_canvas_h5(std::weak_ptr<figure> f, std::string path, float rw, float rh)
{
    std::scoped_lock lk(cvh5_m_);
    cvh5s_.push_back({std::move(f), std::move(path), rw, rh});
}

void plot_manager::flush_captures(hdf5::executor& exec)
{
    std::vector<capture_req> png_pend;
    {
        std::scoped_lock lk(cap_m_);
        png_pend.swap(caps_);
    }
    std::vector<h5_req> h5_pend;
    {
        std::scoped_lock lk(h5_m_);
        h5_pend.swap(h5s_);
    }
    std::vector<h5_req> canvas_pend;
    {
        std::scoped_lock lk(cvh5_m_);
        canvas_pend.swap(cvh5s_);
    }
    if (png_pend.empty() && h5_pend.empty() && canvas_pend.empty()) return;

    // Thumbnail dimensions: longest side ~256, matching the on-screen aspect.
    auto thumb_dims = [](float ref_w, float ref_h) {
        constexpr int tmax   = 256;
        const float   aspect = (ref_h > 0.0f) ? (ref_w / ref_h) : 1.0f;
        int           tw = tmax, th = tmax;
        if (aspect >= 1.0f)
            th = std::max(1, static_cast<int>(std::lround(tmax / aspect)));
        else
            tw = std::max(1, static_cast<int>(std::lround(tmax * aspect)));
        return std::pair{tw, th};
    };

    for (const auto& req : png_pend) {
        if (auto sp = req.f.lock()) write_figure_png(*sp, req.path, req.w, req.h, req.scale);
    }

    for (const auto& req : h5_pend) {
        auto sp = req.f.lock();
        if (!sp) continue;

        h5_save_data data;
        if (!sp->snapshot_h5(data)) continue; // nothing to save
        data.path = req.path;

        auto [tw, th]  = thumb_dims(req.ref_w, req.ref_h);
        data.thumb_w   = tw;
        data.thumb_h   = th;
        data.thumb_png = encode_figure_png(*sp, tw, th, detail::export_scale(tw, th, req.ref_w, req.ref_h));

        exec.submit([payload = std::move(data)] { write_plot_h5(payload); });
    }

    for (const auto& req : canvas_pend) {
        auto sp = req.f.lock();
        if (!sp) continue;
        auto* cv = dynamic_cast<canvas*>(sp.get());
        if (cv == nullptr) continue;

        canvas_save_data data;
        cv->snapshot_canvas(data); // gathers each saveable cell's DTO (GUI thread)
        data.path = req.path;

        auto [tw, th]  = thumb_dims(req.ref_w, req.ref_h);
        data.thumb_w   = tw;
        data.thumb_h   = th;
        data.thumb_png = encode_figure_png(*sp, tw, th, detail::export_scale(tw, th, req.ref_w, req.ref_h));

        exec.submit([payload = std::move(data)] { write_canvas_h5(payload); });
    }
}

} // namespace coast::plot
