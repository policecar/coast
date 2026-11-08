#ifndef COAST_TEST_BLOB_WRITER_H
#define COAST_TEST_BLOB_WRITER_H

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string_view>

#include <nlohmann/json_fwd.hpp>

namespace coast::test {

// Writer for golden-test JSON blobs.
//
// Design (see IDD §5, §6):
//   - Each blob is a single JSON file at a chosen path.
//   - Leaves are scalars (double or int64) or 1D numeric arrays.
//   - Inline arrays live directly in the JSON.
//   - "Large" arrays can be written to a binary sidecar next to the JSON,
//     leaving a {"binary": <name>, "dtype": "f64"|"i64", "shape": [N]} ref.
//   - Names may contain '/' to nest into sub-objects, e.g. "g0/dend_3/act".
//   - Duplicate writes to the same path are an error: a test should never
//     silently overwrite its own data.
//   - close() is idempotent; the destructor calls it if the user did not.
//
// Thread safety: not thread-safe. A blob is owned by one test thread.
class blob_writer {
public:
    explicit blob_writer(std::filesystem::path path);
    ~blob_writer();

    blob_writer(const blob_writer&)            = delete;
    blob_writer& operator=(const blob_writer&) = delete;
    blob_writer(blob_writer&&) noexcept;
    blob_writer& operator=(blob_writer&&) noexcept;

    void scalar(std::string_view name, double            v);
    void scalar(std::string_view name, std::int64_t      v);

    void array(std::string_view name, std::span<const double>       v);
    void array(std::string_view name, std::span<const std::int64_t> v);

    // sidecar_name is the file name (no directory) of the binary file written
    // next to the blob JSON. Raw native-endian dump, dtype-tagged in the JSON
    // ref. The matching reader (coast_compare) trusts the dtype/shape.
    void array_binary(std::string_view name,
                      std::span<const double>       v,
                      std::string_view sidecar_name);
    void array_binary(std::string_view name,
                      std::span<const std::int64_t> v,
                      std::string_view sidecar_name);

    // Flush JSON to disk. Idempotent. Throws on filesystem failure.
    void close();

    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

private:
    void write_sidecar_(std::string_view sidecar_name,
                        const void*      data,
                        std::size_t      byte_count) const;

    std::filesystem::path     path_;
    std::unique_ptr<nlohmann::json> root_;   // pimpl-style; root is a json::object()
    bool                      closed_ = false;
};

} // namespace coast::test

#endif // COAST_TEST_BLOB_WRITER_H
