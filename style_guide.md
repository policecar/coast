# COAST C++ Style Guide

This document captures coding conventions and practices for the COAST project.

## Attributes

### Unused Members and Parameters Reserved for Future Use

When a class member or function parameter is intentionally included for future integration but not yet used, mark it with `[[maybe_unused]]` to suppress compiler warnings:

```cpp
// Unused member variable
private:
    [[maybe_unused]] sim_instance& sim;  // Reserved for future visualization creation

// Unused function parameter
void process([[maybe_unused]] const options& opts) {
    // opts will be used when feature X is implemented
}
```

This makes the intent explicit and avoids warnings from compilers like Clang that flag unused private fields or parameters.

## Naming Conventions

- Classes and structs: `snake_case` (e.g., `sim_instance`, `entity_browser`)
- Member variables: `snake_case` (e.g., `is_visible`, `config_path`)
- Methods: `snake_case` (e.g., `render_dialogs`, `get_sim_env`)
- Type aliases: `snake_case_t` suffix (e.g., `neuron_group_t`, `global_cfg_t`)
- Header guards: `COAST_FILENAME_H` (e.g., `COAST_GUI_MANAGER_H`)

## File Organization

- Headers in `include/<module>/` directories
- Implementation in `src/<module>/` directories
- One class per file, filename matches class name

## Includes

- Group includes: standard library, third-party, project headers
- Use forward declarations in headers where possible to reduce compile times

## RAII and Resource Management

- Use `std::unique_ptr` for owned resources
- Provide `free_resources()` methods for GPU/OpenGL resources that need explicit cleanup before context destruction
- Mark classes as non-copyable when they manage unique resources
