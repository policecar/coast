#include "core/gl_texture.h"

#include <SDL_opengl.h>
#include <cassert>
#include <utility>

namespace coast {

gl_texture::gl_texture(int w, int h, const color_t* data)
    : width(w), height(h)
{
    glGenTextures(1, &texture_id);
    glBindTexture(GL_TEXTURE_2D, texture_id);

    // Set texture parameters for pixel-perfect rendering
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // Allocate texture storage with initial data (or null for empty texture)
    glTexImage2D(
        GL_TEXTURE_2D,
        0,                  // mipmap level
        GL_RGBA8,           // internal format
        width,
        height,
        0,                  // border (must be 0)
        GL_RGBA,            // pixel data format
        GL_UNSIGNED_BYTE,   // pixel data type
        data                // pixel data (can be nullptr)
    );

    glBindTexture(GL_TEXTURE_2D, 0);
}

gl_texture::~gl_texture()
{
    release();
}

gl_texture::gl_texture(gl_texture&& other) noexcept
    : texture_id(other.texture_id)
    , width(other.width)
    , height(other.height)
{
    other.texture_id = 0;
    other.width = 0;
    other.height = 0;
}

gl_texture& gl_texture::operator=(gl_texture&& other) noexcept
{
    if (this != &other) {
        release();
        texture_id = other.texture_id;
        width = other.width;
        height = other.height;
        other.texture_id = 0;
        other.width = 0;
        other.height = 0;
    }
    return *this;
}

void gl_texture::update(const color_t* data)
{
    if (!is_valid() || data == nullptr) return;

    glBindTexture(GL_TEXTURE_2D, texture_id);
    glTexSubImage2D(
        GL_TEXTURE_2D,
        0,                  // mipmap level
        0, 0,               // x, y offset
        width, height,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        data
    );
    glBindTexture(GL_TEXTURE_2D, 0);
}

void gl_texture::update(const std::vector<color_t>& data)
{
    assert(static_cast<int>(data.size()) >= width * height);
    update(data.data());
}

ImTextureID gl_texture::as_imgui_texture() const
{
    // ImGui expects texture ID as ImU64 (unsigned 64-bit)
    return static_cast<ImTextureID>(texture_id);
}

void gl_texture::release()
{
    if (texture_id != 0) {
        glDeleteTextures(1, &texture_id);
        texture_id = 0;
    }
    width = 0;
    height = 0;
}

} // namespace coast
