//
// CIFAR-10 dataset IO entity: five 10000-sample training batches (or one test
// batch), 1 label byte per sample. Everything else lives in cifar_io_base.
//

#include <filesystem>
#include <string>

#include "io/cifar10_io.h"

namespace cifar10 {

namespace {
constexpr std::size_t label_bytes       = 1;
constexpr std::size_t samples_per_batch = 10000;
constexpr std::size_t train_batches     = 5;
}

cifar10_io::cifar10_io(const cifar10_param_t& params, sim::output_id_t output_id) :
    dataset::cifar_io_base(output_id, params, "CIFAR-10")
{
    namespace fs = std::filesystem;

    const std::size_t count = params.use_test_set ? samples_per_batch
                                                  : train_batches * samples_per_batch;
    images.resize(count * image_pixels);

    if (params.use_test_set) {
        load_batch_file((fs::path(params.data_dir) / "test_batch.bin").string(),
                        label_bytes, 0, samples_per_batch);
    } else {
        for (std::size_t batch = 0; batch < train_batches; ++batch) {
            const std::string name = "data_batch_" + std::to_string(batch + 1) + ".bin";
            load_batch_file((fs::path(params.data_dir) / name).string(),
                            label_bytes, batch * samples_per_batch, samples_per_batch);
        }
    }

    rebuild_sequencer(count);
}

} // namespace cifar10
