#ifndef COAST_GUI_PLOT_IMAGE_H
#define COAST_GUI_PLOT_IMAGE_H

#include "gui_plot/figure.h"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace coast {
class gl_texture; // forward-declared: pulls in GL/ImGui, kept out of this header
}

namespace coast::plot {

// A generic image window: a channel for anything that produces a picture. Supply
// a PNG path or raw RGBA pixels; the GUI thread decodes/uploads to a GL texture
// and displays it. Not persisted to HDF5.
class image final : public figure {
public:
    explicit image(std::string title);
    ~image() override; // out-of-line: holds a unique_ptr<gl_texture> (incomplete)

    void set_png(std::string path);                                   // (re)load a PNG file
    void set_rgba(std::span<const std::uint8_t> pixels, int width, int height); // raw RGBA8

    void save_h5(std::string) override {} // images are not saveable
    void free_gl() override;

protected:
    void paint_body(region r) override;

private:
    enum class source : std::uint8_t { none, png, rgba };
    void reload_locked(); // GUI thread, m_ held

    source                      src_ = source::none;
    std::string                 path_;
    std::vector<std::uint8_t>   pixels_;
    int                         w_ = 0;
    int                         h_ = 0;
    bool                        dirty_ = false;
    std::string                 error_;
    std::unique_ptr<gl_texture> tex_; // GUI thread only
};

} // namespace coast::plot

#endif // COAST_GUI_PLOT_IMAGE_H
