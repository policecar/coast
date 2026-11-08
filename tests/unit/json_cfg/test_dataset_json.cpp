#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <nlohmann/json.hpp>

#include <sstream>
#include <string>
#include <string_view>

#include "diag/diag.h"
#include "json_cfg/dataset_json.h"

using ordered_json = nlohmann::ordered_json;

namespace {

void check_sequence_fields(const dataset::sequence_params_t& actual,
                           const dataset::sequence_params_t& expected)
{
    CHECK(actual.change_interval == expected.change_interval);
    CHECK(actual.noise_range.first == doctest::Approx(expected.noise_range.first));
    CHECK(actual.noise_range.second == doctest::Approx(expected.noise_range.second));
    CHECK(actual.random_seed == expected.random_seed);
    CHECK(actual.shuffle == expected.shuffle);
}

void check_equal(const mnist::mnist_param_t& actual, const mnist::mnist_param_t& expected)
{
    check_sequence_fields(actual, expected);
    CHECK(actual.image_file == expected.image_file);
    CHECK(actual.label_file == expected.label_file);
}

void check_equal(const emnist::emnist_param_t& actual, const emnist::emnist_param_t& expected)
{
    check_sequence_fields(actual, expected);
    CHECK(actual.image_file == expected.image_file);
    CHECK(actual.label_file == expected.label_file);
}

void check_equal(const kmnist::kmnist_param_t& actual, const kmnist::kmnist_param_t& expected)
{
    check_sequence_fields(actual, expected);
    CHECK(actual.image_file == expected.image_file);
    CHECK(actual.label_file == expected.label_file);
}

void check_equal(const color_mnist::color_mnist_param_t& actual,
                 const color_mnist::color_mnist_param_t& expected)
{
    check_sequence_fields(actual, expected);
    CHECK(actual.image_file == expected.image_file);
    CHECK(actual.label_file == expected.label_file);
}

void check_equal(const dsprites::dsprites_param_t& actual,
                 const dsprites::dsprites_param_t& expected)
{
    check_sequence_fields(actual, expected);
    CHECK(actual.hdf5_file == expected.hdf5_file);
}

void check_equal(const cifar10::cifar10_param_t& actual,
                 const cifar10::cifar10_param_t& expected)
{
    check_sequence_fields(actual, expected);
    CHECK(actual.data_dir == expected.data_dir);
    CHECK(actual.use_test_set == expected.use_test_set);
}

void check_equal(const cifar100::cifar100_param_t& actual,
                 const cifar100::cifar100_param_t& expected)
{
    check_sequence_fields(actual, expected);
    CHECK(actual.data_dir == expected.data_dir);
    CHECK(actual.use_test_set == expected.use_test_set);
    CHECK(actual.use_coarse_labels == expected.use_coarse_labels);
}

template<typename T>
void check_round_trip(const T& expected)
{
    const ordered_json j = expected;
    const auto actual = j.get<T>();
    check_equal(actual, expected);
}

template<typename T>
void check_null_defaults()
{
    const ordered_json j = nullptr;
    const auto actual = j.get<T>();
    check_equal(actual, T{});
}

// The dsprites deprecation warning now goes through coast::diag (COAST_WARN),
// whose default sink writes via fprintf(stderr) — NOT std::cerr's streambuf — so
// capture it by installing a diag sink rather than swapping cerr's rdbuf.
class scoped_diag_capture {
public:
    scoped_diag_capture()
    {
        coast::diag::set_level(coast::diag::level::warn);
        coast::diag::set_sink(
            [this](coast::diag::level, std::string_view, std::string_view m) {
                captured_ += std::string(m);
            });
    }

    scoped_diag_capture(const scoped_diag_capture&) = delete;
    scoped_diag_capture& operator=(const scoped_diag_capture&) = delete;

    ~scoped_diag_capture() { coast::diag::set_sink({}); }  // restore default stderr sink

    [[nodiscard]] std::string str() const { return captured_; }

private:
    std::string captured_;
};

} // namespace

TEST_CASE("dataset_json: default params round-trip by value")
{
    check_round_trip(mnist::mnist_param_t{});
    check_round_trip(emnist::emnist_param_t{});
    check_round_trip(kmnist::kmnist_param_t{});
    check_round_trip(color_mnist::color_mnist_param_t{});
    check_round_trip(dsprites::dsprites_param_t{});
    check_round_trip(cifar10::cifar10_param_t{});
    check_round_trip(cifar100::cifar100_param_t{});
}

TEST_CASE("dataset_json: populated params round-trip by value")
{
    mnist::mnist_param_t mnist;
    mnist.image_file = "/data/mnist/images";
    mnist.label_file = "/data/mnist/labels";
    mnist.change_interval = 42;
    mnist.noise_range = {0.1, 0.9};
    mnist.random_seed = seed_t{ 99 };
    mnist.shuffle = false;
    check_round_trip(mnist);

    emnist::emnist_param_t emnist;
    emnist.image_file = "/data/emnist/images";
    emnist.label_file = "/data/emnist/labels";
    emnist.change_interval = 43;
    emnist.noise_range = {0.2, 0.8};
    emnist.random_seed = seed_t{ 98 };
    emnist.shuffle = false;
    check_round_trip(emnist);

    kmnist::kmnist_param_t kmnist;
    kmnist.image_file = "/data/kmnist/images";
    kmnist.label_file = "/data/kmnist/labels";
    kmnist.change_interval = 44;
    kmnist.noise_range = {0.3, 0.7};
    kmnist.random_seed = seed_t{ 97 };
    kmnist.shuffle = false;
    check_round_trip(kmnist);

    color_mnist::color_mnist_param_t color_mnist;
    color_mnist.image_file = "/data/color_mnist/images";
    color_mnist.label_file = "/data/color_mnist/labels";
    color_mnist.change_interval = 45;
    color_mnist.noise_range = {0.4, 0.6};
    color_mnist.random_seed = seed_t{ 96 };
    color_mnist.shuffle = false;
    check_round_trip(color_mnist);

    dsprites::dsprites_param_t dsprites;
    dsprites.hdf5_file = "/data/dsprites/data.h5";
    dsprites.change_interval = 46;
    dsprites.noise_range = {0.05, 0.15};
    dsprites.random_seed = seed_t{ 95 };
    dsprites.shuffle = false;
    check_round_trip(dsprites);

    cifar10::cifar10_param_t cifar10;
    cifar10.data_dir = "/data/cifar10";
    cifar10.change_interval = 47;
    cifar10.noise_range = {0.11, 0.22};
    cifar10.random_seed = seed_t{ 94 };
    cifar10.shuffle = false;
    cifar10.use_test_set = true;
    check_round_trip(cifar10);

    cifar100::cifar100_param_t cifar100;
    cifar100.data_dir = "/data/cifar100";
    cifar100.change_interval = 48;
    cifar100.noise_range = {0.12, 0.34};
    cifar100.random_seed = seed_t{ 93 };
    cifar100.shuffle = false;
    cifar100.use_test_set = true;
    cifar100.use_coarse_labels = true;
    check_round_trip(cifar100);
}

TEST_CASE("dataset_json: null params default instead of throwing")
{
    check_null_defaults<mnist::mnist_param_t>();
    check_null_defaults<emnist::emnist_param_t>();
    check_null_defaults<kmnist::kmnist_param_t>();
    check_null_defaults<color_mnist::color_mnist_param_t>();
    check_null_defaults<dsprites::dsprites_param_t>();
    check_null_defaults<cifar10::cifar10_param_t>();
    check_null_defaults<cifar100::cifar100_param_t>();
}

TEST_CASE("dataset_json: dsprites randomize_readout remains accepted with warning")
{
    const auto j = ordered_json::parse(R"({
        "hdf5_file": "/data/dsprites/data.h5",
        "change_interval": 42,
        "noise_range": [0.1, 0.9],
        "random_seed": 99,
        "shuffle": false,
        "randomize_readout": true
    })");

    scoped_diag_capture captured;
    const auto actual = j.get<dsprites::dsprites_param_t>();

    dsprites::dsprites_param_t expected;
    expected.hdf5_file = "/data/dsprites/data.h5";
    expected.change_interval = 42;
    expected.noise_range = {0.1, 0.9};
    expected.random_seed = seed_t{ 99 };
    expected.shuffle = false;
    check_equal(actual, expected);
    CHECK(captured.str().find("randomize_readout") != std::string::npos);
}
