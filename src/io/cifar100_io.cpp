//
// CIFAR-100 dataset IO entity: one 50000-sample training file (or one 10000-
// sample test file), 2 label bytes (coarse + fine) per sample. Everything else
// lives in cifar_io_base.
//

#include <filesystem>

#include "io/cifar100_io.h"

namespace cifar100 {

namespace {
constexpr std::size_t label_bytes   = 2;
constexpr std::size_t train_samples = 50000;
constexpr std::size_t test_samples  = 10000;
}

cifar100_io::cifar100_io(const cifar100_param_t& params, sim::output_id_t output_id) :
    dataset::cifar_io_base(output_id, params, "CIFAR-100")
{
    namespace fs = std::filesystem;

    const std::size_t count = params.use_test_set ? test_samples : train_samples;
    images.resize(count * image_pixels);

    const char* file = params.use_test_set ? "test.bin" : "train.bin";
    load_batch_file((fs::path(params.data_dir) / file).string(), label_bytes, 0, count);

    rebuild_sequencer(count);
}

} // namespace cifar100
