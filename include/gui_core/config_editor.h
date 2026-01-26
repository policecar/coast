#ifndef COAST_CONFIG_EDITOR_H
#define COAST_CONFIG_EDITOR_H

#include "json_loader.h"

namespace coast {

/**
 * Configuration editor dialog for editing global_cfg_t.
 * Provides UI for editing dataset configs, neural model configs,
 * and simulation graph nodes.
 */
class config_editor {
public:
    explicit config_editor(loader::global_cfg_t& config);

    /**
     * Render the configuration editor window.
     * @param p_open Pointer to visibility flag (window can be closed via X button)
     */
    void render(bool* p_open);

private:
    loader::global_cfg_t& config;

    // Render sections
    void render_dataset_configs();
    void render_ngm2_configs();
    void render_simulation_graph();
};

} // namespace coast

#endif // COAST_CONFIG_EDITOR_H
