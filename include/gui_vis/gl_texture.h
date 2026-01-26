#ifndef COAST_GL_TEXTURE_H
#define COAST_GL_TEXTURE_H

#include <cstdint>
#include <vector>

#include "color_types.h"
#include "imgui.h"

// Forward declare OpenGL types to avoid including heavy headers
using GLuint = unsigned int;

namespace coast {

/**
 * RAII wrapper for OpenGL textures.
 * Provides simple interface for creating and updating GPU textures
 * compatible with ImGui::Image().
 */
class gl_texture {
public:
    gl_texture() = default;

    /**
     * Create a texture with specified dimensions.
     * @param w Width in pixels
     * @param h Height in pixels
     * @param data Optional initial pixel data (RGBA format)
     */
    gl_texture(int w, int h, const color_t* data = nullptr);

    ~gl_texture();

    // Move-only semantics (textures should not be copied)
    gl_texture(gl_texture&& other) noexcept;
    gl_texture& operator=(gl_texture&& other) noexcept;
    gl_texture(const gl_texture&) = delete;
    gl_texture& operator=(const gl_texture&) = delete;

    /**
     * Update texture contents from pixel data.
     * @param data Pixel data in RGBA format (must match texture dimensions)
     */
    void update(const color_t* data);

    /**
     * Update texture contents from vector.
     * @param data Vector of pixels (size must match width * height)
     */
    void update(const std::vector<color_t>& data);

    /**
     * Check if texture is valid (has been created on GPU).
     */
    [[nodiscard]] bool is_valid() const { return texture_id != 0; }

    /**
     * Get OpenGL texture ID.
     */
    [[nodiscard]] GLuint get_id() const { return texture_id; }

    /**
     * Get texture width in pixels.
     */
    [[nodiscard]] int get_width() const { return width; }

    /**
     * Get texture height in pixels.
     */
    [[nodiscard]] int get_height() const { return height; }

    /**
     * Convert to ImTextureID for use with ImGui::Image().
     */
    [[nodiscard]] ImTextureID as_imgui_texture() const;

    /**
     * Release GPU resources. Called automatically by destructor.
     */
    void release();

private:
    GLuint texture_id = 0;
    int width = 0;
    int height = 0;
};

} // namespace coast

#endif // COAST_GL_TEXTURE_H
