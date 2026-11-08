#ifndef COAST_DATASET_BROWSER_H
#define COAST_DATASET_BROWSER_H

#include <vector>

#include "gui_core/dataset_access.h"
#include "gui_core/thumbnail_grid.h"

namespace coast {

class sim_instance;

/**
 * Browser window for visually inspecting dataset samples.
 *
 * A thin view over the shared dataset_access (enumeration) and thumbnail_grid
 * (virtualized rendering) helpers: a dataset combo plus a view-only thumbnail grid.
 */
class dataset_browser {
public:
    explicit dataset_browser(sim_instance& sim);
    ~dataset_browser();

    void render();

    /** Access visibility state for menu toggle. */
    bool& visible() { return is_visible; }

    /** Free GPU/OpenGL resources. */
    void free_resources();

private:
    sim_instance& sim;
    bool is_visible = false;

    std::vector<dataset_handle> datasets;
    int selected_dataset_idx = -1;

    thumbnail_grid grid;

    void rebuild_dataset_list();
};

} // namespace coast

#endif // COAST_DATASET_BROWSER_H
