//
// Created by jk on 25.01.26.
//

#ifndef DATASET_CFG_H
#define DATASET_CFG_H

#include <concepts>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>

#include "io/mnist_cfg.h"
#include "io/dsprites_cfg.h"
#include "io/emnist_cfg.h"
#include "io/kmnist_cfg.h"
#include "io/cifar10_cfg.h"
#include "io/cifar100_cfg.h"
#include "io/chairs3d_cfg.h"
#include "io/color_mnist_cfg.h"

namespace dataset {

struct cfg_store {

    std::unordered_map<std::string,mnist::mnist_param_t>             mnist;
    std::unordered_map<std::string,dsprites::dsprites_param_t>       dsprites;
    std::unordered_map<std::string,emnist::emnist_param_t>           emnist;
    std::unordered_map<std::string,kmnist::kmnist_param_t>           kmnist;
    std::unordered_map<std::string,cifar10::cifar10_param_t>         cifar10;
    std::unordered_map<std::string,cifar100::cifar100_param_t>       cifar100;
    std::unordered_map<std::string,chairs3d::chairs3d_param_t>       chairs3d;
    std::unordered_map<std::string,color_mnist::color_mnist_param_t> color_mnist;

    cfg_store();
};

// Calls f(family, map) for every dataset family of `s` (const or not). The family
// keys are sim::entity_types' dataset_family names — a unit test holds the two
// in step (tests/unit/tools/test_entity_types.cpp).
template <class Store, class F>
    requires std::same_as<std::remove_const_t<Store>, cfg_store>
void for_each_family(Store& s, F&& f)
{
    f(std::string_view{ "mnist" },       s.mnist);
    f(std::string_view{ "dsprites" },    s.dsprites);
    f(std::string_view{ "emnist" },      s.emnist);
    f(std::string_view{ "kmnist" },      s.kmnist);
    f(std::string_view{ "cifar10" },     s.cifar10);
    f(std::string_view{ "cifar100" },    s.cifar100);
    f(std::string_view{ "chairs3d" },    s.chairs3d);
    f(std::string_view{ "color_mnist" }, s.color_mnist);
}

inline cfg_store::cfg_store()
{
    for_each_family(*this, [](std::string_view, auto& map) { map["default"]; });
}


}

#endif //DATASET_CFG_H
