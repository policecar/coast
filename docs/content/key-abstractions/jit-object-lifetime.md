+++
title = "JIT Object Lifetime"
weight = 40
+++

# JIT Object Lifetime

The h5-browser console runs user C++ through an embedded clang-repl interpreter. Objects that code creates can be handed to host storage and outlive the interpreter that built them. Their **code** does not automatically outlive it — and that mismatch is a whole class of crashes.

COAST closes the class structurally, with a **retaining JIT code arena** in `src/console/repl.cpp`. You do not need to remember anything per type or per teardown site. But you do need to know the arena exists before you "clean up" what looks like a memory leak.

## The Problem

Consider a console macro that builds a plot:

```cpp
auto h = std::make_shared<coast::plot::histogram>("act");
h->show();   // plot_manager now keeps a shared_ptr — the figure outlives the macro
```

`std::make_shared<T>` instantiates `std::_Sp_counted_ptr_inplace<T, ...>` **at the point of use** — inside the interpreter's JIT module. That control block's `_M_dispose()` and `_M_destroy()` are **virtual**, so the object carries a vptr into a vtable that lives in JIT memory.

Tearing the interpreter down (Reset Console, switching files, or destroying the session) unmaps that module. The figure is still registered, and its vptr now points at nothing. The next release dispatches through it and the process dies:

```
#0  std::_Sp_counted_base<...>::_M_release_last_use_cold()   <-- host code, virtual call
#1  coast::plot::plot_manager::free_resources()              <-- figs_.clear()
#2  run_gui shutdown lambda
```

The fingerprint is characteristic: **frame #0 is host code in libstdc++ performing a virtual call**. The code doing the dispatch is fine; the vtable it dispatches through is gone.

## Which Objects Are Affected

The rule is not "shared_ptr" and not "plots". It is:

> An object is hazardous iff it stores a pointer to JIT-emitted code that host code will later call.

Type erasure and virtual dispatch bind to the JIT's copy. Concrete types do not.

| Hazardous | Why |
|---|---|
| `std::shared_ptr<T>` — **any `T`, including host types** | control-block vtable |
| Any macro-defined polymorphic type, including overrides of host interfaces | its own vtable |
| `std::function`, `std::move_only_function` | stored invoker + manager pointers |
| `std::any` | stored manager pointer |
| `std::packaged_task` / `promise` / `future` | shared state has a virtual destructor |
| `unique_ptr<HostBase>` holding a macro-derived object | virtual destructor |
| Raw function pointers into JIT code held host-side | direct call into the module |

| Safe | Why |
|---|---|
| `std::vector<double>`, `std::string`, PODs, aggregates of host types | the host destroys them using the **host's own** instantiation; no stored code pointer is consulted (their heap buffers are fine too — the JIT resolves `operator new` to the process's) |

Two consequences are worth internalising, because both are counter-intuitive and both were **measured**, not assumed:

- **The JIT emits its own vtable even when the host image already contains an identical instantiation.** Comparing control-block vptrs via `dladdr`, a host `make_shared<T>` resolves into the executable while a macro's resolves into anonymous JIT memory. No type is safe by accident.
- **A host-side factory cannot fix the general case.** Moving `make_shared` into `coast_lib` behind a private constructor does relocate that one vtable to the host — but the host can never pre-instantiate a type that does not exist until the user types it into the console.

## The Solution: A Retaining Code Arena

Since the hazard cannot be closed per type, close it at the memory layer: **make the emitted code outlive the interpreter.**

Keeping whole interpreters alive would work but is far too expensive — a retired `clang::Interpreter` costs **~145 MB**, and an idle one that compiled nothing still costs **~141 MB**, because ~97% of it is clang frontend (AST, Sema, preprocessor) that nothing needs afterwards.

So COAST retains only the *code*. clang-repl accepts a builder chain that is entirely public API — no patched LLVM:

```
Interpreter::create(CI, IncrementalExecutorBuilder)
  └─ IEB.JITBuilder  (llvm::orc::LLJITBuilder)
       └─ setObjectLinkingLayerCreator(...)
            └─ ObjectLinkingLayer(ES, retaining_memory_manager)
                 └─ deallocate(): FinalizedAlloc::release(), then never unmap
```

`FinalizedAlloc::release()` hands back the address and resets the allocation, satisfying the assertion in its destructor. The arena then simply never forwards it to the inner `InProcessMemoryManager`, so the pages stay mapped and executable. The interpreter dies and its 141 MB of frontend is reclaimed; its code survives.

The arena is a process-wide singleton that is **never destroyed** — `ObjectLinkingLayer` holds it *by reference*, and outliving every interpreter is the entire point.

### This leaks by design

That is the honest trade, and it is bounded and measured: **~50–160 kB per interpreter**, scaling with *code emitted* rather than with reset count. An interpreter is retired on Reset Console **and on every file switch**, so retention tracks console activity, not just resets. At these sizes a long session costs single-digit MB.

{{% callout warning %}}
The arena never frees JIT code. This is deliberate — it is what makes console-created objects safe. Do not "fix" it. Removing it reintroduces a crash class that no per-type discipline can close.
{{% /callout %}}

### Exception handling comes along for free

JITLink pairs `__register_frame` (an *allocation* action) with `__deregister_frame` (a *deallocation* action). Because the arena never deallocates, the deregistration never runs and EH frames stay registered — so unwinding out of retained code keeps working, in both directions across the JIT/host boundary. This is verified, not inferred.

## Consequences for the GUI

Plot figures created in the console now **survive Reset Console** and session teardown. Reset no longer clears the figure registry, which also means host-loaded plots — which were never at risk — are no longer destroyed as collateral. A user who wants a plot gone closes it.

## Testing

The arena is disabled by the environment variable `COAST_REPL_RETAIN_JIT=0`. It is on whenever the variable is unset or holds any other value.

That switch exists for A/B testing, and it is what makes the regression tests meaningful: with the arena off, `unit.console.session`'s plot tests die with SIGSEGV; with it on they pass. A test that passes both ways would prove nothing. Use the same trick when validating by hand in the GUI — create a plot, Reset Console, quit, with and without the variable.

## See Also

- `src/console/repl.cpp` — the arena and its wiring
- [Threading]({{% relref "/threading" %}}) — the console runs its interpreter on its own thread and tears it down there
