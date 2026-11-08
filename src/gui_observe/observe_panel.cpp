//
// Created for the COAST observe system
//

#include "gui_observe/observe_panel.h"

#include "observe/observe_registry.h"
#include "observe/observe_group.h"
#include "observe/observe_trigger.h"

#include <imgui.h>

#include <cstdint>
#include <cstring>
#include <string>

namespace gui_observe {

namespace {

bool is_active(const observe::observe_group_t* g) {
    // Reflect the per-step commit decision — the user-visible "is this group
    // capturing right now". Recomputed by the observation pass each step.
    return g->active_this_step.load(std::memory_order_relaxed);
}

const char* state_label(const observe::observe_group_t* g) {
    return is_active(g) ? "ACTIVE" : "IDLE";
}

ImVec4 state_color(const observe::observe_group_t* g) {
    return is_active(g) ? ImVec4(0.30f, 0.85f, 0.30f, 1.0f)
                        : ImVec4(0.55f, 0.55f, 0.55f, 1.0f);
}

} // namespace

void observe_panel::render() {
    if (!visible_) return;

    auto& reg = observe::observe_registry::instance();

    ImGui::SetNextWindowSize(ImVec2(640, 460), ImGuiCond_FirstUseEver);

    bool open = visible_;
    if (!ImGui::Begin("Observer Capture", &open)) {
        ImGui::End();
        visible_ = open;
        return;
    }

    auto groups = reg.all_groups();

    if (groups.empty()) {
        ImGui::TextDisabled("No OBSERVE groups configured.");
        ImGui::TextDisabled("Add a group below or load a config containing an \"observe\" section.");
    }

    for (auto* g : groups) {
        if (g == nullptr) continue;

        ImGui::PushID(g);

        // Header line: group id + state badge
        ImGui::Text("Group: %s", g->group_id.c_str());
        ImGui::SameLine();
        ImGui::TextColored(state_color(g), "[%s]", state_label(g));

        // Site count via the group's lock-protected getter (never races a
        // concurrent sim-thread insert).
        ImGui::TextDisabled("Sites: %zu", g->site_count());

        if (!g->h5_path.empty()) {
            ImGui::TextDisabled("Output: %s", g->h5_path.string().c_str());
        }

        // Combine mode combo. Routed through with_group_locked so we don't race
        // with on_post_process or other registry mutations.
        {
            const char* items[] = { "OR", "AND" };
            int current = (g->combine == observe::combine_mode_t::AND_) ? 1 : 0;
            ImGui::SetNextItemWidth(120.0f);
            if (ImGui::Combo("Combine", &current, items, IM_ARRAYSIZE(items))) {
                const auto new_mode = (current == 1) ? observe::combine_mode_t::AND_
                                                     : observe::combine_mode_t::OR_;
                reg.with_group_locked(g, [new_mode](observe::observe_group_t& gr) {
                    gr.combine = new_mode;
                });
            }
        }

        // Ranges. Editable list of (start, stop) inclusive step ranges.
        ImGui::Text("Ranges:");
        ImGui::Indent();

        // Take a copy so we can iterate stably while planning edits.
        std::vector<observe::step_range_t> ranges_copy = g->ranges;
        int remove_idx = -1;
        bool any_edit  = false;

        for (std::size_t i = 0; i < ranges_copy.size(); ++i) {
            ImGui::PushID(static_cast<int>(i));

            std::uint64_t start_v = ranges_copy[i].start_step;
            std::uint64_t stop_v  = ranges_copy[i].stop_step;

            ImGui::SetNextItemWidth(140.0f);
            if (ImGui::InputScalar("##start", ImGuiDataType_U64, &start_v)) {
                ranges_copy[i].start_step = start_v;
                any_edit = true;
            }
            ImGui::SameLine();
            ImGui::TextUnformatted("..");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(140.0f);
            if (ImGui::InputScalar("##stop", ImGuiDataType_U64, &stop_v)) {
                ranges_copy[i].stop_step = stop_v;
                any_edit = true;
            }
            ImGui::SameLine();
            if (ImGui::SmallButton("X")) {
                remove_idx = static_cast<int>(i);
            }

            ImGui::PopID();
        }

        if (ImGui::SmallButton("+ range")) {
            ranges_copy.push_back(observe::step_range_t{0, 0});
            any_edit = true;
        }

        if (remove_idx >= 0) {
            ranges_copy.erase(ranges_copy.begin() + remove_idx);
            any_edit = true;
        }

        if (any_edit) {
            reg.with_group_locked(g, [&ranges_copy](observe::observe_group_t& gr) {
                gr.ranges = ranges_copy;
            });
        }

        ImGui::Unindent();

        // Trigger buttons
        if (ImGui::Button("START")) {
            reg.trigger(g->group_id, observe::trigger_mode_t::START);
        }
        ImGui::SameLine();
        if (ImGui::Button("STOP")) {
            reg.trigger(g->group_id, observe::trigger_mode_t::STOP);
        }
        ImGui::SameLine();
        if (ImGui::Button("SINGLE_CAPTURE")) {
            reg.trigger(g->group_id, observe::trigger_mode_t::SINGLE_CAPTURE);
        }

        ImGui::Separator();
        ImGui::PopID();
    }

    // Add-group input
    ImGui::Spacing();
    ImGui::SetNextItemWidth(220.0f);
    ImGui::InputText("##new_group", new_group_buf_, sizeof(new_group_buf_));
    ImGui::SameLine();
    if (ImGui::Button("Add Group")) {
        if (new_group_buf_[0] != '\0') {
            reg.get_or_create_group(new_group_buf_);
            std::memset(new_group_buf_, 0, sizeof(new_group_buf_));
        }
    }

    ImGui::End();
    visible_ = open;
}

} // namespace gui_observe
