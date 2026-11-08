#ifndef COAST_DATASET_LOADER_H
#define COAST_DATASET_LOADER_H

#include <memory>
#include <string>

namespace hdf5 { class executor; }
namespace loader { class json_loader; }
namespace sim { class env; class sim_builder; }

namespace coast {

/**
 * Loads ONLY the dataset (io_entity) objects described by a JSON config, without
 * standing up the rest of a simulation — no neuron groups, no sim thread, no
 * logging, and no GPU context.
 *
 * It reuses the normal build pipeline (json_loader -> sim_builder ->
 * register_entity_handlers -> build_local) but feeds build_local a config filtered
 * down to CPU dataset nodes only. Neuron-group nodes are dropped (so no ngm config
 * is required), GPU dataset nodes are dropped (so no gpu_context is needed), and
 * node_sets that reference any dropped node are dropped (so build_local cannot
 * throw on a missing template node).
 *
 * The constructed entities live in this object's own sim::env for its entire
 * lifetime, so dataset_handle accessors enumerated from env()/builder() (via
 * enumerate_datasets) remain valid exactly as long as the dataset_loader does.
 *
 * Intended owner: the Metric Explorer, which keeps a list of these — one per
 * imported config — fully decoupled from whatever sim_instance (if any) is loaded.
 */
class dataset_loader {
public:
    /**
     * @param config_path  Path to a JSON configuration file.
     * @param h5_exec      HDF5 executor, required by HDF5-backed datasets (dSprites).
     * @throws std::runtime_error / std::exception on parse or build failure.
     */
    dataset_loader(const std::string& config_path, hdf5::executor& h5_exec);
    ~dataset_loader();

    // Non-copyable, non-movable: handle accessors capture pointers into the owned
    // env, so the object's address must be stable.
    dataset_loader(const dataset_loader&) = delete;
    dataset_loader& operator=(const dataset_loader&) = delete;
    dataset_loader(dataset_loader&&) = delete;
    dataset_loader& operator=(dataset_loader&&) = delete;

    [[nodiscard]] const std::string& config_path() const { return config_path_; }
    [[nodiscard]] sim::env& env() { return *env_; }
    [[nodiscard]] const sim::sim_builder& builder() const { return *builder_; }

private:
    std::string config_path_;
    std::unique_ptr<loader::json_loader> json_loader_;
    std::unique_ptr<sim::env> env_;
    std::unique_ptr<sim::sim_builder> builder_;
};

} // namespace coast

#endif // COAST_DATASET_LOADER_H
