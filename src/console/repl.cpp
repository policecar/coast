#include "console/repl.h"

#include "h5browse/console_api.h"

#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/Utils.h"
#include "clang/Interpreter/IncrementalExecutor.h"
#include "clang/Interpreter/Interpreter.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/Preprocessor.h"

#include "llvm/ExecutionEngine/JITLink/JITLinkMemoryManager.h"
#include "llvm/ExecutionEngine/Orc/EHFrameRegistrationPlugin.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/TargetSelect.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string_view>

// COAST_INCLUDE_DIR and CLANG_RESOURCE_DIR are injected by CMake. The include
// dir lets macros/preamble resolve "h5browse/console_api.h"; the resource dir
// lets the embedded driver find Clang's builtin headers (it cannot self-locate
// them — argv[0] is not the clang binary).
#ifndef COAST_INCLUDE_DIR
#define COAST_INCLUDE_DIR ""
#endif
#ifndef CLANG_RESOURCE_DIR
#define CLANG_RESOURCE_DIR ""
#endif
// macOS-only: injected by CMake so the embedded interpreter finds the SDK
// sysroot (C headers) and libc++ (the <cstdio> C++ wrappers). Empty on other
// platforms, where the embedded driver locates the standard headers itself.
#ifndef COAST_MACOS_SYSROOT
#define COAST_MACOS_SYSROOT ""
#endif
#ifndef COAST_LIBCXX_INCLUDE_DIR
#define COAST_LIBCXX_INCLUDE_DIR ""
#endif

namespace console {

namespace {

// ===== retaining JIT code arena =====
//
// An object a macro hands to host storage outlives the interpreter that built it.
// Its code does not: `std::make_shared<T>` instantiates _Sp_counted_ptr_inplace<T>
// IN THE JIT MODULE, and that control block's _M_dispose/_M_destroy are virtual, so
// the object's control block points at a vtable in JIT memory. interp.reset() unmaps
// it; the next release dispatches through a dangling vptr and segfaults. The same
// goes for std::function (invoker/manager pointers), std::any, and any type a macro
// derives from a host interface. Measured: the JIT emits its own vtable even when
// the host image already has an identical instantiation, so no type is safe by luck
// and no host-side factory can fix the general case.
//
// The fix is to make the emitted CODE outlive the interpreter. Keeping whole
// interpreters alive costs ~145 MB each (measured), of which ~97% is clang frontend
// we do not need. So instead give ORC a memory manager we own that declines to unmap
// finalized allocations: the interpreter dies and its frontend is reclaimed, while
// its code pages survive in this process-wide arena. Measured cost: ~50-160 kB per
// interpreter, scaling with code emitted rather than with resets.
//
// This LEAKS BY DESIGN -- that is the price of the guarantee, and it is bounded and
// small. Set COAST_REPL_RETAIN_JIT=0 to disable it and observe the crash it prevents.
class retaining_memory_manager final : public llvm::jitlink::JITLinkMemoryManager {
public:
    explicit retaining_memory_manager(
        std::unique_ptr<llvm::jitlink::InProcessMemoryManager> inner)
        : inner_(std::move(inner))
    {
    }

    void allocate(const llvm::jitlink::JITLinkDylib* jd, llvm::jitlink::LinkGraph& g,
                  OnAllocatedFunction on_allocated) override
    {
        inner_->allocate(jd, g, std::move(on_allocated));
    }

    void deallocate(std::vector<FinalizedAlloc> allocs,
                    OnDeallocatedFunction        on_deallocated) override
    {
        // RETAIN. release() hands back the address and resets the FinalizedAlloc,
        // which is what its destructor asserts on; we then simply never forward it
        // to the inner manager, so the pages stay mapped and executable.
        for (auto& a : allocs)
            if (a) {
                (void)a.release();
                retained_.fetch_add(1, std::memory_order_relaxed);
            }
        on_deallocated(llvm::Error::success());
    }

    static std::size_t retained() { return retained_.load(std::memory_order_relaxed); }

private:
    std::unique_ptr<llvm::jitlink::InProcessMemoryManager> inner_;
    static inline std::atomic<std::size_t>                 retained_{0};
};

// Process-wide and intentionally never destroyed: ObjectLinkingLayer holds this
// BY REFERENCE and must not outlive it, and the whole point is that the memory
// survives every interpreter.
llvm::jitlink::JITLinkMemoryManager* jit_arena()
{
    static retaining_memory_manager* arena = [] {
        auto ip = llvm::jitlink::InProcessMemoryManager::Create();
        if (!ip) {
            llvm::consumeError(ip.takeError());
            return static_cast<retaining_memory_manager*>(nullptr);
        }
        return new retaining_memory_manager(std::move(*ip));
    }();
    return arena;
}

// ON by default: correctness depends on it. COAST_REPL_RETAIN_JIT=0 disables it
// for A/B testing only -- with the arena off, ANY shared_ptr or polymorphic object
// a macro hands to host storage crashes once its interpreter is torn down.
bool retain_jit_enabled()
{
    const char* e = std::getenv("COAST_REPL_RETAIN_JIT");
    return e == nullptr || std::strcmp(e, "0") != 0;
}
// ===== end retaining JIT code arena =====

void init_llvm_once()
{
    static std::once_flag flag;
    std::call_once(flag, [] {
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();
        llvm::InitializeNativeTargetAsmParser();
    });
}

// True if `line` is a `#include`/`#import` preprocessor directive (leading
// whitespace and whitespace between `#` and the keyword allowed). Text-based:
// the caller must first establish that the line lies wholly outside a block
// comment (see scan_comment_state / split_into_units).
bool is_include_directive(std::string_view line)
{
    std::size_t i = line.find_first_not_of(" \t");
    if (i == std::string_view::npos || line[i] != '#') return false;
    i = line.find_first_not_of(" \t", i + 1); // skip '#' and any ws after it
    if (i == std::string_view::npos) return false;
    const std::string_view rest = line.substr(i);
    return rest.starts_with("include") || rest.starts_with("import");
}

// Advance `in_comment` (a /* ... */ block-comment flag) across one line.
//
// Deliberately NOT a full lexer: it does not model string or raw-string
// literals, so a `/*` inside a string literal is read as opening a comment.
// That asymmetry is safe BY CONSTRUCTION, because the only thing this gates is
// whether an #include gets ISOLATED (see split_into_units): a false "we are in a
// comment" merely skips isolation, degrading to the old whole-block behaviour.
// Note the reach: `in_comment` persists across lines, so an unmatched `/*` in a
// string literal skips isolation for every later #include in the submission,
// not just that line — still only a degradation, never a split-apart comment
// turning working code into a parse error. `//` is honoured so a trailing
// `// /*` does not open a phantom comment.
void scan_comment_state(std::string_view line, bool& in_comment)
{
    for (std::size_t i = 0; i < line.size();) {
        if (in_comment) {
            if (line.compare(i, 2, "*/") == 0) {
                in_comment = false;
                i += 2;
            } else {
                ++i;
            }
        } else if (line.compare(i, 2, "/*") == 0) {
            in_comment = true;
            i += 2;
        } else if (line.compare(i, 2, "//") == 0) {
            break; // rest of the line is a line comment
        } else {
            ++i;
        }
    }
}

// Split a submitted block into ordered execution units so that a fresh
// `#include` file-entry never shares a clang Parse() unit with code that may
// fail to parse. RATIONALE (measured, LLVM 22): when one Parse() unit both
// enters a new source file via #include AND then hits a parse/sema error, the
// failed-TU cleanup corrupts the preprocessor's source-file state and poisons
// every later Parse() — the classic symptom being a follow-up cell reporting
// "use of undeclared identifier 'std'" as the whole preamble vanishes. Upstream
// clang-repl dodges this by consuming one line at a time; we consume blocks, so
// we isolate each #include ourselves. Code-only units recover cleanly from a
// parse error (verified), so contiguous non-include lines stay a single unit,
// preserving multi-line statements and trailing-expression semantics.
//
// A line is only isolated when it lies WHOLLY outside a block comment — it must
// both start and end outside one. That rules out two shapes that would otherwise
// be cut in half into an unterminated comment (a hard parse error on code clang
// accepts): an #include commented out inside a /* ... */ block, and a directive
// with a comment opened on its own line (`#include <x>  /* why ...`). Anything
// not provably outside a comment simply stays in the current run: isolation is
// an optimisation, so declining it is always safe.
std::vector<std::string> split_into_units(const std::string& code)
{
    std::vector<std::string> units;
    std::string              run; // accumulates contiguous non-include lines
    std::size_t              pos        = 0;
    bool                     in_comment = false;
    while (pos < code.size()) {
        std::size_t nl   = code.find('\n', pos);
        const bool  last = (nl == std::string::npos);
        const std::size_t end  = last ? code.size() : nl + 1; // keep newline
        const std::string_view line(code.data() + pos, end - pos);

        const bool started_outside = !in_comment;
        scan_comment_state(line, in_comment);
        const bool wholly_outside = started_outside && !in_comment;

        if (wholly_outside && is_include_directive(line)) {
            if (!run.empty()) {
                units.push_back(std::move(run));
                run.clear();
            }
            units.emplace_back(line);
        } else {
            run.append(line);
        }
        pos = end;
    }
    if (!run.empty()) units.push_back(std::move(run));
    if (units.empty()) units.emplace_back(); // preserve empty-input behavior
    return units;
}

} // namespace

struct repl::impl {
    std::unique_ptr<clang::Interpreter> interp;
    std::string                         init_err;
    bool                                ok = false;
};

repl::repl(std::vector<std::string> extra_include_dirs) : p_(std::make_unique<impl>())
{
    init_llvm_once();

    // -fno-color-diagnostics: diagnostics are captured to a GUI text panel, not
    // a terminal — ANSI color escapes would render as garbage.
    std::vector<const char*> args = {"-std=c++23", "-resource-dir",
                                     CLANG_RESOURCE_DIR, "-fno-color-diagnostics"};
    const std::string coast_inc = COAST_INCLUDE_DIR;
    if (!coast_inc.empty()) {
        args.push_back("-I");
        args.push_back(coast_inc.c_str());
    }
    for (const std::string& d : extra_include_dirs) {
        args.push_back("-I");
        args.push_back(d.c_str());
    }
    // macOS: the embedded driver skips the toolchain's sysroot/libc++ detection,
    // so JIT-compiled `#include <cstdio>` etc. go unfound. Supply them explicitly
    // (CMake injects this build's paths; both empty on other platforms). These
    // std::strings outlive CreateCpp() below, so their c_str()s stay valid.
    const std::string macos_sysroot = COAST_MACOS_SYSROOT;
    if (!macos_sysroot.empty()) {
        args.push_back("-isysroot");
        args.push_back(macos_sysroot.c_str());
    }
    const std::string libcxx_inc = COAST_LIBCXX_INCLUDE_DIR;
    if (!libcxx_inc.empty()) {
        args.push_back("-isystem");
        args.push_back(libcxx_inc.c_str());
    }
    // clang's builtin headers (stddef.h -> size_t/ptrdiff_t) that the SDK C
    // headers reference but do not define; must sit between libc++ and the SDK
    // headers so the #include_next chain resolves size_t before the SDK uses it.
    const std::string builtin_inc =
        macos_sysroot.empty() ? std::string() : std::string(CLANG_RESOURCE_DIR) + "/include";
    if (!builtin_inc.empty()) {
        args.push_back("-isystem");
        args.push_back(builtin_inc.c_str());
    }
    // The SDK's C library headers (mbstate_t, wint_t, wcschr, …) — -isysroot
    // alone doesn't put them on the embedded driver's search path, and libc++'s
    // wrappers #include_next into them. Must follow the libc++ dir above.
    const std::string macos_c_inc =
        macos_sysroot.empty() ? std::string() : macos_sysroot + "/usr/include";
    if (!macos_c_inc.empty()) {
        args.push_back("-isystem");
        args.push_back(macos_c_inc.c_str());
    }

    clang::IncrementalCompilerBuilder cb;
    cb.SetCompilerArgs(args);
    auto ci = cb.CreateCpp();
    if (!ci) {
        p_->init_err = "CreateCpp failed: " + llvm::toString(ci.takeError());
        return;
    }
    // Route object linking through the retaining arena so this interpreter's emitted
    // code outlives it. A null builder is the stock path (create()'s IEB defaults to
    // null), which is what COAST_REPL_RETAIN_JIT=0 selects.
    std::unique_ptr<clang::IncrementalExecutorBuilder> ieb;
    if (retain_jit_enabled() && jit_arena() != nullptr) {
        auto jb = std::make_unique<llvm::orc::LLJITBuilder>();
        // The trailing pack absorbs the JITLinkMemoryManager& that LLVM >= 23 passes
        // (ObjectLinkingLayerCreator gained it; older LLVM passes only the session).
        // It is ignored on purpose: linking must go through the retaining arena, not
        // LLJIT's own manager, whose allocations die with the interpreter.
        jb->setObjectLinkingLayerCreator(
            [](llvm::orc::ExecutionSession& es, auto&... /*llvm23_mem_mgr*/)
                -> llvm::Expected<std::unique_ptr<llvm::orc::ObjectLayer>> {
                auto oll = std::make_unique<llvm::orc::ObjectLinkingLayer>(es, *jit_arena());
                // Register eh-frames for JIT-emitted code. The stock LLJIT layer
                // does this via its default object-linking-layer creator; overriding
                // the creator (above) drops it, and this bare layer sets up no
                // MachO/ELF Platform to register frames either. Without it a
                // JIT-thrown exception carries no unwind info the host can follow, so
                // on macOS it aborts (libc++abi: terminating) instead of unwinding
                // into exec_guarded()'s host try/catch. Needed on Darwin; harmless
                // elsewhere.
                if (auto ehp = llvm::orc::EHFrameRegistrationPlugin::Create(es))
                    oll->addPlugin(std::move(*ehp));
                else
                    llvm::consumeError(ehp.takeError());
                return std::unique_ptr<llvm::orc::ObjectLayer>(std::move(oll));
            });
        ieb             = std::make_unique<clang::IncrementalExecutorBuilder>();
        ieb->JITBuilder = std::move(jb);
    }
    auto interp = clang::Interpreter::create(std::move(*ci), std::move(ieb));
    if (!interp) {
        p_->init_err =
            "Interpreter::create failed: " + llvm::toString(interp.takeError());
        return;
    }
    p_->interp = std::move(*interp);
    p_->ok     = true;

    // Preamble: pull in the console API and declare the host report hook.
    const auto pre = exec(
        "#include <cstdio>\n"
        "#include <cstdint>\n"
        "#include <string>\n"
        "#include \"h5browse/console_api.h\"\n"
        "using namespace h5b;\n"
        "using namespace h5b::console;\n"
        // Sim-control façade. A namespace ALIAS (not `using namespace`) so the
        // ergonomic `simc::run(c)` works while its free functions `run`/`config`
        // stay qualified — a user can still write `auto run = simc::run(c);`.
        "#include \"sim_console/sim_console_api.h\"\n"
        "namespace simc = coast::simc;\n"
        // coastd daemon-control façade. Alias `coastd` -> coast::coastd so the
        // REPL user writes `coastd::session::connect()`, `coastd::discover()`.
        "#include \"coastd_console/coastd_console_api.h\"\n"
        "namespace coastd = coast::coastd;\n"
        // Campaign meta-loader. Alias `camp` -> coast::campaign so the REPL user
        // writes `camp::open("<dir>")`. Not a `using namespace`: its free function
        // open() would otherwise collide with h5b::console::open(), which is pulled
        // in unqualified above and opens a single HDF5 file rather than a campaign.
        "#include \"campaign/campaign_api.h\"\n"
        "namespace camp = coast::campaign;\n");
    if (!pre.ok) {
        p_->ok       = false;
        p_->init_err = "preamble failed: " + pre.message +
                       " (is COAST_INCLUDE_DIR correct? '" + coast_inc + "')";
    }
}

repl::~repl() = default;

bool               repl::ready()      const { return p_->ok; }
const std::string& repl::init_error() const { return p_->init_err; }

exec_result repl::exec(const std::string& code)
{
    if (!p_->ok || !p_->interp) {
        return {false, "interpreter not ready"};
    }
    if (auto err = p_->interp->ParseAndExecute(code)) {
        return {false, llvm::toString(std::move(err))};
    }
    return {true, {}};
}

// Guarded execution of ONE unit (no #include-file-entry mixed with fallible
// code — the caller guarantees this via split_into_units).
//
// Run UNWRAPPED so top-level declarations persist across calls, and catch a
// runtime throw on the HOST side (verified: a JIT-thrown exception unwinds into
// a host try/catch rather than aborting). No interpreter-side wrapper means no
// scope is introduced around the user's code.
//
// Parse and Execute are split deliberately: a PARSE error registers no
// translation unit, so it must NOT be followed by Undo — undoing after a parse
// error rolls back the PREVIOUS good input, silently dropping the user's earlier
// declarations and, after several errors, unwinding past the preamble binding
// into a corrupted interpreter (observed to crash). Undo runs ONLY when a
// successfully-parsed unit fails at execution / symbol materialization, where it
// cleanly drops exactly that unit.
//
// A runtime THROW out of the JIT'd code is such an execution failure: Parse has
// already registered the declaration, then the initializer throws mid-Execute
// (e.g. open() on a missing/unreadable file). Without Undo the half-initialized
// declaration stays registered, so retrying the same name — a user fixing a
// typo'd path and re-running the cell — hits "redefinition of 'x'". So the catch
// blocks also Undo, but ONLY once Parse has succeeded (`parsed`): a throw out of
// Parse itself registered no unit and must fall through to the parse-error rule.
exec_result repl::exec_unit_guarded(const std::string& code)
{
    bool parsed = false;
    try {
        auto ptu = p_->interp->Parse(code);
        if (!ptu) {
            return {false, llvm::toString(ptu.takeError())};
        }
        parsed = true;
        if (auto err = p_->interp->Execute(*ptu)) {
            std::string msg = llvm::toString(std::move(err));
            if (auto u = p_->interp->Undo(1)) llvm::consumeError(std::move(u));
            return {false, std::move(msg)};
        }
        return {true, {}};
    } catch (const std::exception& e) {
        if (parsed) {
            if (auto u = p_->interp->Undo(1)) llvm::consumeError(std::move(u));
        }
        return {false, std::string("uncaught exception: ") + e.what()};
    } catch (...) {
        if (parsed) {
            if (auto u = p_->interp->Undo(1)) llvm::consumeError(std::move(u));
        }
        return {false, "uncaught exception"};
    }
}

exec_result repl::exec_guarded(const std::string& code)
{
    if (!p_->ok || !p_->interp) {
        return {false, "interpreter not ready"};
    }

    // Isolate every #include into its own unit: a fresh header file-entry that
    // shares a Parse() unit with code that then fails to parse corrupts the
    // interpreter (see split_into_units). Run units in order; stop and report at
    // the first failing one, matching whole-block failure semantics.
    for (std::string& unit : split_into_units(code)) {
        exec_result r = exec_unit_guarded(unit);
        if (!r.ok) return r;
    }
    return {true, {}};
}

exec_result repl::add_include_path(const std::string& dir)
{
    if (!p_->ok || !p_->interp) return {false, "interpreter not ready"};
    try {
        clang::CompilerInstance& ci = *p_->interp->getCompilerInstance();
        // IgnoreSysRoot=true: `dir` is a user-supplied absolute path, not a
        // path relative to the (macOS) -isysroot. With false, a sysroot-prefixed
        // lookup misses the header — benign on Linux (no sysroot), broken on macOS.
        ci.getHeaderSearchOpts().AddPath(dir, clang::frontend::Angled,
                                         /*IsFramework=*/false, /*IgnoreSysRoot=*/true);
        clang::ApplyHeaderSearchOptions(ci.getPreprocessor().getHeaderSearchInfo(),
                                        ci.getHeaderSearchOpts(), ci.getLangOpts(),
                                        ci.getTarget().getTriple());
        return {true, {}};
    } catch (const std::exception& e) {
        return {false, std::string("add_include_path: ") + e.what()};
    }
}

exec_result repl::load_library(const std::string& path)
{
    if (!p_->ok || !p_->interp) return {false, "interpreter not ready"};
    if (auto err = p_->interp->LoadDynamicLibrary(path.c_str())) {
        return {false, llvm::toString(std::move(err))};
    }
    return {true, {}};
}

// Macro files go through the SAME hardened path as cell input. They used to use
// the raw exec(), a leftover from the era when exec_guarded() wrapped its input
// in an interpreter-side try/catch and so could not host declarations. That
// wrapper is long gone — exec_guarded() runs unwrapped and declarations persist
// through it — but load() was never moved over, which left a user-supplied file
// running with no exception guard and no #include isolation. Both were measured:
// a throwing file-scope initializer aborted the process, and a file with a fresh
// #include plus a parse error poisoned the interpreter.
exec_result repl::load(const std::filesystem::path& file)
{
    std::ifstream in(file);
    if (!in) return {false, "cannot open macro file: " + file.string()};
    std::stringstream ss;
    ss << in.rdbuf();
    return exec_guarded(ss.str());
}

exec_result repl::run_macro(const std::filesystem::path& file)
{
    const auto r = load(file);
    if (!r.ok) return r;
    return exec_guarded(file.stem().string() + "();");
}

} // namespace console
