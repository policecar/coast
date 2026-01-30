#ifndef IMGUI_SCALING_H
#define IMGUI_SCALING_H

#include "imgui.h"

namespace coast {

// Get scaled width for input widgets (accounts for HiDPI font scaling)
// Base font size is assumed to be 16px
inline float scaled(float base_value)
{
    return base_value * (ImGui::GetFontSize() / 16.0f);
}

} // namespace coast

#endif // IMGUI_SCALING_H
