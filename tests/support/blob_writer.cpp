#include "blob_writer.h"

#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace coast::test {

namespace {

// Split "a/b/c" → ["a", "b", "c"]. Throws on empty segments (e.g. "a//b",
// "/x", "x/"), which would yield meaningless JSON keys.
std::vector<std::string> split_path(std::string_view name) {
    if (name.empty()) {
        throw std::runtime_error("blob_writer: empty field name");
    }
    std::vector<std::string> out;
    std::size_t start = 0;
    for (std::size_t i = 0; i <= name.size(); ++i) {
        if (i == name.size() || name[i] == '/') {
            if (i == start) {
                throw std::runtime_error(
                    "blob_writer: empty path segment in '" + std::string(name) + "'");
            }
            out.emplace_back(name.substr(start, i - start));
            start = i + 1;
        }
    }
    return out;
}

// Walk root_ down the path, creating intermediate objects as needed. Throws
// if any intermediate already exists with a non-object type, or if the leaf
// already exists.
nlohmann::json& walk_to_new_leaf(nlohmann::json& root, std::string_view name) {
    auto segs = split_path(name);
    nlohmann::json* cur = &root;
    for (std::size_t i = 0; i + 1 < segs.size(); ++i) {
        const auto& key = segs[i];
        if (cur->contains(key)) {
            if (!(*cur)[key].is_object()) {
                throw std::runtime_error(
                    "blob_writer: path '" + std::string(name) +
                    "' collides with a non-object at segment '" + key + "'");
            }
        } else {
            (*cur)[key] = nlohmann::json::object();
        }
        cur = &(*cur)[key];
    }
    const auto& leaf_key = segs.back();
    if (cur->contains(leaf_key)) {
        throw std::runtime_error(
            "blob_writer: duplicate field '" + std::string(name) + "'");
    }
    return (*cur)[leaf_key];   // freshly inserted null, ready to assign
}

} // namespace

blob_writer::blob_writer(std::filesystem::path path)
    : path_(std::move(path)),
      root_(std::make_unique<nlohmann::json>(nlohmann::json::object())) {}

blob_writer::~blob_writer() {
    if (!closed_) {
        try { close(); } catch (...) { /* dtor swallows */ }
    }
}

blob_writer::blob_writer(blob_writer&&) noexcept            = default;
blob_writer& blob_writer::operator=(blob_writer&&) noexcept = default;

void blob_writer::scalar(std::string_view name, double v) {
    if (closed_) throw std::runtime_error("blob_writer: write after close()");
    walk_to_new_leaf(*root_, name) = v;
}

void blob_writer::scalar(std::string_view name, std::int64_t v) {
    if (closed_) throw std::runtime_error("blob_writer: write after close()");
    walk_to_new_leaf(*root_, name) = v;
}

void blob_writer::array(std::string_view name, std::span<const double> v) {
    if (closed_) throw std::runtime_error("blob_writer: write after close()");
    walk_to_new_leaf(*root_, name) =
        nlohmann::json(std::vector<double>(v.begin(), v.end()));
}

void blob_writer::array(std::string_view name, std::span<const std::int64_t> v) {
    if (closed_) throw std::runtime_error("blob_writer: write after close()");
    walk_to_new_leaf(*root_, name) =
        nlohmann::json(std::vector<std::int64_t>(v.begin(), v.end()));
}

void blob_writer::array_binary(std::string_view name,
                               std::span<const double> v,
                               std::string_view sidecar_name) {
    if (closed_) throw std::runtime_error("blob_writer: write after close()");
    write_sidecar_(sidecar_name, v.data(), v.size_bytes());
    auto& leaf = walk_to_new_leaf(*root_, name);
    leaf = nlohmann::json::object();
    leaf["binary"] = std::string(sidecar_name);
    leaf["dtype"]  = "f64";
    leaf["shape"]  = nlohmann::json::array({ v.size() });
}

void blob_writer::array_binary(std::string_view name,
                               std::span<const std::int64_t> v,
                               std::string_view sidecar_name) {
    if (closed_) throw std::runtime_error("blob_writer: write after close()");
    write_sidecar_(sidecar_name, v.data(), v.size_bytes());
    auto& leaf = walk_to_new_leaf(*root_, name);
    leaf = nlohmann::json::object();
    leaf["binary"] = std::string(sidecar_name);
    leaf["dtype"]  = "i64";
    leaf["shape"]  = nlohmann::json::array({ v.size() });
}

void blob_writer::write_sidecar_(std::string_view sidecar_name,
                                 const void*      data,
                                 std::size_t      byte_count) const {
    if (sidecar_name.empty()) {
        throw std::runtime_error("blob_writer: empty sidecar name");
    }
    if (sidecar_name.find('/') != std::string_view::npos ||
        sidecar_name.find('\\') != std::string_view::npos) {
        throw std::runtime_error(
            "blob_writer: sidecar name must not contain path separators: '" +
            std::string(sidecar_name) + "'");
    }
    const auto sidecar_path = path_.parent_path() / std::string(sidecar_name);
    std::ofstream out(sidecar_path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("blob_writer: failed to open sidecar '" +
                                 sidecar_path.string() + "' for writing");
    }
    out.write(static_cast<const char*>(data),
              static_cast<std::streamsize>(byte_count));
    if (!out) {
        throw std::runtime_error("blob_writer: failed to write sidecar '" +
                                 sidecar_path.string() + "'");
    }
}

void blob_writer::close() {
    if (closed_) return;
    if (auto parent = path_.parent_path(); !parent.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(parent, ec);   // ok if it already exists
    }
    std::ofstream out(path_);
    if (!out) {
        throw std::runtime_error("blob_writer: failed to open blob '" +
                                 path_.string() + "' for writing");
    }
    out << root_->dump(2) << '\n';
    if (!out) {
        throw std::runtime_error("blob_writer: failed to write blob '" +
                                 path_.string() + "'");
    }
    closed_ = true;
}

} // namespace coast::test
