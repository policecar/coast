#ifndef COAST_COLOR_TYPES_H
#define COAST_COLOR_TYPES_H

#include <cstdint>

namespace coast {

/**
 * Renderer-agnostic color type representing RGBA values.
 * Compatible with OpenGL texture formats (GL_RGBA, GL_UNSIGNED_BYTE).
 */
struct color_t {
    uint8_t r;
    uint8_t g;
    uint8_t b;
    uint8_t a;

    constexpr color_t() : r(0), g(0), b(0), a(255) {}

    constexpr color_t(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255)
        : r(r), g(g), b(b), a(a) {}

    constexpr bool operator==(const color_t& other) const {
        return r == other.r && g == other.g && b == other.b && a == other.a;
    }

    constexpr bool operator!=(const color_t& other) const {
        return !(*this == other);
    }
};

// Common color constants (matching raylib colors for easy migration)
inline constexpr color_t COLOR_WHITE      {255, 255, 255, 255};
inline constexpr color_t COLOR_BLACK      {0,   0,   0,   255};
inline constexpr color_t COLOR_RAYWHITE   {245, 245, 245, 255};
inline constexpr color_t COLOR_BLUE       {0,   121, 241, 255};
inline constexpr color_t COLOR_ORANGE     {255, 161, 0,   255};
inline constexpr color_t COLOR_RED        {230, 41,  55,  255};
inline constexpr color_t COLOR_GREEN      {0,   228, 48,  255};
inline constexpr color_t COLOR_YELLOW     {253, 249, 0,   255};
inline constexpr color_t COLOR_DARKGREEN  {0,   117, 44,  255};
inline constexpr color_t COLOR_GRAY       {130, 130, 130, 255};
inline constexpr color_t COLOR_LIGHTGRAY  {200, 200, 200, 255};
inline constexpr color_t COLOR_DARKGRAY   {80,  80,  80,  255};

} // namespace coast

#endif // COAST_COLOR_TYPES_H
