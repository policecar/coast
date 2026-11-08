#include "console/console_panel.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <string>
#include <utility>

namespace console {

// Animated "running" indicator: a spinner glyph cycling on ImGui's frame clock
// so a long-running cell reads as alive rather than hung. Amber to stand out
// against normal log text. No persistent state — driven by ImGui::GetTime().
static void running_indicator()
{
    static const char spinner[4] = {'|', '/', '-', '\\'};
    const int         frame       = static_cast<int>(ImGui::GetTime() * 8.0) & 3;
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.95f, 0.75f, 0.25f, 1.0f));
    ImGui::Text("%c running", spinner[frame]);
    ImGui::PopStyleColor();
}

console_panel::console_panel(hdf5::executor& exec) : session_(exec)
{
    file_browser_.SetTitle("Open HDF5 log");
    file_browser_.SetTypeFilters({".h5"});
    macro_browser_.SetTitle("Run macro (.x)");
    macro_browser_.SetTypeFilters({".cpp", ".C", ".cxx"});
}

std::string console_panel::open_file(const std::filesystem::path& path)
{
    // The session assigns (and dedupes) the variable name; just relay it.
    return session_.open_file(path);
}

std::string console_panel::reserve_var(const std::string& base)
{
    return session_.reserve_var(base);
}

void console_panel::submit(const std::string& code) { session_.submit(code); }

void console_panel::bind_dataset(const std::string& var, const std::string& file_var,
                                 const std::string& dataset)
{
    session_.bind_dataset(var, file_var, dataset);
}

void console_panel::run_current_input()
{
    // No file precondition: the console is usable the moment it is ready, and a
    // cell can open its own files via open("...").
    if (input_[0] != '\0') {
        dispatch_line(input_);
        input_[0] = '\0';
    }
}

void console_panel::dispatch_line(const std::string& line)
{
    const std::size_t start = line.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return;

    // Meta-commands are ".<cmd> [arg]". Split the command token from its
    // (trimmed) argument.
    if (line[start] == '.') {
        const std::size_t cmd_end = line.find_first_of(" \t\r\n", start);
        const std::string cmd =
            line.substr(start, (cmd_end == std::string::npos ? line.size() : cmd_end) - start);
        std::string arg;
        if (cmd_end != std::string::npos) {
            const std::size_t b = line.find_first_not_of(" \t\r\n", cmd_end);
            if (b != std::string::npos) {
                arg = line.substr(b, line.find_last_not_of(" \t\r\n") - b + 1);
            }
        }

        if (cmd == ".?") {
            show_help(arg);
            return;
        }
        if (cmd == ".x" && !arg.empty()) {
            session_.run_macro(std::filesystem::path{arg});
            return;
        }
        if (cmd == ".L" && !arg.empty()) {
            session_.load(std::filesystem::path{arg});
            return;
        }
        if (cmd == ".I" && !arg.empty()) {
            session_.add_include_path(arg);
            return;
        }
        if (cmd == ".load" && !arg.empty()) {
            session_.load_library(arg);
            return;
        }
        // Unrecognized ".command" falls through to the interpreter (which
        // reports a clear error).
    }
    session_.submit(line);
}

void console_panel::show_help(const std::string& topic)
{
    // echo the command (re-runnable cell)
    pending_.push_back({output_line::kind::input, topic.empty() ? ".?" : (".? " + topic)});

    auto out = [&](const char* s) { pending_.push_back({output_line::kind::output, s}); };

    std::string t = topic;
    for (char& c : t) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

    if (t.empty() || t == "help") {
        out("COAST HDF5 console — quick help");
        out("  open(\"f.h5\")     open a file -> handle     ->  .? open");
        out("  <a dataset>     from f->dataset(...)      ->  .? dataset");
        out("  <a row>         from scan() / .row(i)    ->  .? row");
        out("  coast metrics   cosine / minkowski       ->  .? metric");
        out("  live plots      ImPlot windows           ->  .? plot");
        out("  run sims        edit config, run, iterate ->  .? sim");
        out("Meta-commands:");
        out("  .L <file>       load a macro's declarations");
        out("  .x <file>       load + run a macro (calls the function named after the file)");
        out("  .I <dir>        add a header search directory (for external libs)");
        out("  .load <lib.so>  load a shared library so its symbols resolve");
        out("  .? [topic]      this help");
        out("Tip: no file needs to be open to start — call open(\"...\") for each file.");
        out("     Right-click a dataset in the HDF5 Browser tree to bind a variable here.");
    } else if (t == "open" || t == "h5" || t == "file") {
        out("open(path)  —  open an HDF5 file; returns shared_ptr<file_console>");
        out("  auto f = open(\"run.h5\");   // as many files at once as you like");
        out("  f->list()               -> vector<string>   dataset display names");
        out("  f->count()              -> size_t           number of datasets");
        out("  f->family()             -> string_view      \"observe\" | \"buffer log\" | ...");
        out("  f->dataset(\"name\")      -> dataset_handle   (throws if unknown)");
        out("  f->dataset(index)       -> dataset_handle");
        out("Example:");
        out("  auto f = open(\"run.h5\");");
        out("  for (auto& n : f->list()) printf(\"%s\\n\", n.c_str());");
        out("  auto d = f->dataset(\"neuron[3]/activity\");");
        out("Trap: bind the handle first. A dataset kept from a temporary dangles:");
        out("  auto d = open(\"run.h5\")->dataset(\"act\");  // WRONG: file dies at ';'");
        out("  auto f = open(\"run.h5\"); auto d = f->dataset(\"act\");  // correct");
    } else if (t == "dataset" || t == "datasets" || t == "d") {
        out("dataset_handle  —  one dataset (from f->dataset(...))");
        out("  .name()                 -> string_view");
        out("  .rows()                 -> uint64           number of rows");
        out("  .width()                -> uint64           values per row (1 = scalar)");
        out("  .dtype()                -> scalar_type");
        out("  .scan()                 -> range            stream all rows (single-pass)");
        out("  .scan(first, n)         -> range            stream a sub-range");
        out("  .row(i)                 -> row_view         random access to row i");
        out("  .column(c)              -> vector<double>   one column across all rows");
        out("  .steps()                -> vector<uint64>   sim step per row (if present)");
        out("Example:");
        out("  auto d = f->dataset(\"act\");");
        out("  double s = 0; for (auto r : d.scan()) s += r.at(0);");
        out("  printf(\"mean = %g\\n\", s / d.rows());");
    } else if (t == "row" || t == "row_view" || t == "r") {
        out("row_view  —  one row (from scan() or .row(i))");
        out("  .index()                -> uint64           absolute row index");
        out("  .width()                -> uint64           values in the row");
        out("  .dtype()                -> scalar_type");
        out("  .at(col)                -> double           value widened to double (any dtype)");
        out("  .as<float>()            -> span<const float> zero-copy typed view (throws on mismatch)");
        out("  .as_copy<float>()       -> vector<float>     owning copy (no lifetime coupling)");
        out("  .bytes()                -> span<const byte>");
        out("Note: scan() rows are valid only within the loop iteration;");
        out("      .row(i) rows stay valid while the row_view is held.");
        out("Trap: a span never outlives its row_view, and");
        out("        auto ref = d.row(i).as<float>();     // row_view dies at the ';'");
        out("      leaves ref dangling — it reads another row once the cache evicts");
        out("      the block, silently and without a crash. To keep a row's values");
        out("      (e.g. a fixed reference vector), use .as_copy<float>() instead.");
    } else if (t == "plot" || t == "plots") {
        out("coast plots   (add:  #include \"gui_plot/console_plot.h\" ; use std::make_shared)");
        out("  auto h = std::make_shared<coast::plot::histogram>(\"title\", bins = 64)");
        out("  auto l = std::make_shared<coast::plot::line>(\"title\")");
        out("  h->show()               register + show the window (nothing shows until show())");
        out("  h->update(y)            set data (histogram bins y; line plots y vs index)");
        out("  h->update(x, y)         line with explicit x");
        out("  h->close()   h->set_title(...)   h->set_theme(...)   h->point_count()");
        out("  histogram2d / pixels: set_axis_visible(plot_axis::x|y|z, false), set_palette(coast::colormap_t::plasma)");
        out("  auto img = std::make_shared<coast::plot::image>(\"title\")");
        out("  img->set_png(\"/path/plot.png\")   img->set_rgba(pixels, w, h)");
        out("  canvas: auto c = std::make_shared<coast::plot::canvas>(\"t\", rows, cols); c->show();");
        out("          c->place(fig, r, c)   (place a figure into a cell; do NOT show() it)");
        out("Example:");
        out("  #include \"gui_plot/console_plot.h\"");
        out("  auto h = std::make_shared<coast::plot::histogram>(\"activity[0]\");");
        out("  h->show();");
        out("  h->update(f->dataset(\"act\").column(0));   // live ImPlot window");
        out("Tip: update() sets data only; visibility is explicit — show()/close().");
        out("     A closed handle persists; show() (not update) brings the window back.");
    } else if (t == "metric" || t == "metrics") {
        out("coast metrics   (add:  #include \"tools/metric.h\" )");
        out("  coast::cosine_similarity(span<const float> a, span<const float> b) -> float");
        out("  coast::minkowski_distance(a, b, p) -> float   (p=1 Manhattan, p=2 Euclidean)");
        out("Example:");
        out("  #include \"tools/metric.h\"");
        out("  auto f = open(\"run.h5\");");
        out("  auto d = f->dataset(\"vec\");");
        out("  printf(\"cos = %f\\n\", coast::cosine_similarity(");
        out("             d.row(0).as<float>(), d.row(1).as<float>()));");
    } else if (t == "sim" || t == "sims" || t == "run") {
        out("run simulations   (namespace coast::simc; runs a child `coast` per step())");
        out("  auto c = simc::config(\"configs/mnist.json\");   // editable config");
        out("  c.set(\"/ngm2/dendrites/default/default_primary_learning_rate\", 0.03);");
        out("  c.run_dir(\"runs/lr_003\");   // observe+snapshots+config.json land here");
        out("  c.seed(0);                    // child --seed-offset");
        out("  auto h = simc::run(c);              // dir = c.run_dir() or runs/<stem>");
        out("  auto h = simc::run(c, \"runs/lr_003\"); // ...or pass the target dir here");
        out("  h->step(50000);               // run a chunk in a child process (BLOCKS)");
        out("  h->step(50000);               // continue: chains saved state");
        out("  h->step_async(50000);         // NON-blocking; opens the status dialog");
        out("  h->step_async(50000, false);  // ...without opening the dialog");
        out("  h->running()     -> bool      // a step (blocking or async) is in flight");
        out("  h->cancel();                  // stop this handle's in-flight run");
        out("  h->total_steps()  h->run_dir()  h->observe_dir()  h->state_path()");
        out("  h->last_log()    -> <run_dir>/logs/step_<timestamp>_NNN.log (per step())");
        out("  h->save_state(\"runs/lr_003/final.h5\");   h->load_state(path);");
        out("Concurrent runs (sweep): fire step_async on several handles — all appear");
        out("  together in the Simulation Runs table. simc::wait_all_runs() blocks til done;");
        out("  simc::active_run_count(); simc::clear_finished_runs();");
        out("  Pool: at most simc::max_concurrent_runs() (default 4) run at once; the rest");
        out("  QUEUE and start as slots free. simc::set_max_concurrent_runs(n) to change it");
        out("  (or env COAST_SIMC_MAX_CONCURRENT). simc::cancel_all_runs() stops everything.");
        out("  simc::show_runs_window() reopens the table if you closed it.");
        out("  In the table: failed bar=red, cancelled=amber; per-row Cancel + Cancel all;");
        out("  double-click a row to tail its log.");
        out("Config edits (raw, by JSON Pointer):");
        out("  c.set(ptr, 1.0e-4 | 42 | true | \"text\");   c.get_number(ptr)  c.get_string(ptr)");
        out("  c.dump()   -> pretty JSON string (inspect before running)");
        out("Filesystem: simc::mkdir(p)  simc::exists(p)  simc::ls(p)  simc::cwd()");
        out("Then inspect outputs with the h5 side:  auto f = open(h->observe_dir()+\"/...\");");
        out("Monitor a running step():  tail -f the file at h->last_log() (coast prints");
        out("      periodic progress). step() BLOCKS until the child exits; throws on failure.");
        out("Note: each step() reloads the dataset + (de)serializes state — size chunks");
        out("      at >=10k steps. step(0) is a no-op (coast rejects a raw -n 0).");
    } else if (t == "coastd" || t == "daemon") {
        out("coastd daemon control   (namespace coast::coastd; talks to a coastd over TCP)");
        out("  auto s = coastd::session::connect();              // localhost:51900");
        out("  auto s = coastd::session::connect(\"host\", 51900);  // a remote machine");
        out("  auto id = s.submit(\"main\", \"configs/mnist.json\", \"lbl\", \"note\", 500000);");
        out("      // ref, config (a path on THIS machine — its CONTENT is sent),");
        out("      //      label, message, steps_target");
        out("  s.list();                       // pretty-print the queue");
        out("  for (auto j : s.jobs())         // the queue as data (iterable handle)");
        out("      if (j.status == \"failed\") s.cancel(j.id);   // j.id/status/bucket/steps…");
        out("  s.reorder(id, 0);               // move within the pending queue (0 = front)");
        out("  s.cancel(id);");
        out("  s.print_info();  auto h = s.info();   // machine_info (h.hostname/os/slots/…)");
        out("  coastd::print_discovered();     // UDP broadcast; print daemons that answer");
        out("  for (auto d : coastd::discover()) …   // discovered daemons as data");
        out("Errors (connect refused, path-unsafe config, unknown id) throw std::runtime_error.");
    } else {
        pending_.push_back({output_line::kind::error,
                            "unknown help topic '" + topic +
                                "'  (try: .? h5 | dataset | row | metric | plot | sim | coastd)"});
    }
}

void console_panel::render_input_cell(const std::string& code, int id)
{
    ImGui::PushID(id);

    if (ImGui::SmallButton("Run")) dispatch_line(code);
    ImGui::SameLine();
    if (ImGui::SmallButton("Edit")) {
        std::snprintf(input_, sizeof input_, "%s", code.c_str()); // recall into input
    }
    ImGui::SameLine();
    if (ImGui::SmallButton("Copy")) ImGui::SetClipboardText(code.c_str());

    // Read-only, selectable code box (native selection + Ctrl+C), sized to the
    // line count and capped.
    int lines = 1;
    for (char c : code) {
        if (c == '\n') ++lines;
    }
    const float h = std::clamp(static_cast<float>(lines), 1.0f, 12.0f) *
                        ImGui::GetTextLineHeight() +
                    ImGui::GetStyle().FramePadding.y * 2.0f + 2.0f;
    ImGui::InputTextMultiline("##code", const_cast<char*>(code.c_str()),
                              code.size() + 1, ImVec2(-1.0f, h),
                              ImGuiInputTextFlags_ReadOnly);

    ImGui::PopID();
}

void console_panel::render()
{
    if (!show_) return;

    ImGui::SetNextWindowSize(ImVec2(720, 540), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("C++ Console", &show_)) {
        ImGui::End();
        return;
    }

    // Pull new session output.
    for (auto& line : session_.drain_output()) {
        log_.push_back(std::move(line));
        scroll_bottom_ = true;
    }
    // Splice locally-generated lines (e.g. .? help) here — never inside the log
    // render loop, where a cell's Run button could mutate log_ mid-iteration.
    if (!pending_.empty()) {
        for (auto& line : pending_) log_.push_back(std::move(line));
        pending_.clear();
        scroll_bottom_ = true;
    }
    if (log_.size() > max_log_lines) {
        log_.erase(log_.begin(),
                   log_.begin() + static_cast<std::ptrdiff_t>(log_.size() - max_log_lines));
    }

    // Toolbar. No file is required — the console runs from the start; open()
    // (or the Open .h5... button) adds files as session variables.
    const std::vector<std::string> files = session_.open_files();

    if (ImGui::Button("Open .h5...")) file_browser_.Open();
    ImGui::SameLine();
    if (ImGui::Button("Run macro...")) macro_browser_.Open();
    ImGui::SameLine();
    if (files.empty()) {
        ImGui::TextDisabled("no file open — call open(\"...\") or use Open .h5...");
    } else if (files.size() == 1) {
        ImGui::TextUnformatted(files.front().c_str());
    } else {
        ImGui::Text("%zu files open", files.size());
        if (ImGui::IsItemHovered()) {
            ImGui::BeginTooltip();
            for (const std::string& p : files) ImGui::TextUnformatted(p.c_str());
            ImGui::EndTooltip();
        }
    }
    if (session_.busy()) {
        ImGui::SameLine();
        running_indicator();
    }

    // Clear (log display) + Reset Console (rebuild the interpreter), right-aligned
    // to the toolbar's right edge. Reset recovers from a poisoned JIT — see
    // session::do_reset — and re-opens the files that were open.
    const ImGuiStyle& style   = ImGui::GetStyle();
    const float       clear_w = ImGui::CalcTextSize("Clear").x + style.FramePadding.x * 2.0f;
    const float       reset_w =
        ImGui::CalcTextSize("Reset Console").x + style.FramePadding.x * 2.0f;
    const float group_w = clear_w + reset_w + style.ItemSpacing.x;
    ImGui::SameLine(ImGui::GetContentRegionMax().x - group_w);
    if (ImGui::SmallButton("Clear")) log_.clear();
    ImGui::SameLine();
    if (ImGui::SmallButton("Reset Console")) session_.reset();

    // Layout: output log (fills), splitter, input box, run row.
    const float run_row = ImGui::GetFrameHeightWithSpacing();
    const float split_h = 6.0f;
    const float avail   = ImGui::GetContentRegionAvail().y;
    input_height_       = std::clamp(input_height_, 48.0f, std::max(60.0f, avail - 90.0f));
    float log_h         = avail - input_height_ - split_h - run_row;
    log_h               = std::max(log_h, 60.0f);

    // Output log.
    ImGui::BeginChild("log", ImVec2(0, log_h), true, ImGuiWindowFlags_HorizontalScrollbar);
    int cell_id = 0;
    for (const auto& line : log_) {
        if (line.k == output_line::kind::input) {
            render_input_cell(line.text, cell_id++);
            continue;
        }
        ImVec4 col;
        switch (line.k) {
            case output_line::kind::error: col = ImVec4(0.95f, 0.40f, 0.40f, 1.0f); break;
            case output_line::kind::info:  col = ImVec4(0.50f, 0.80f, 0.50f, 1.0f); break;
            default: col = ImGui::GetStyleColorVec4(ImGuiCol_Text); break;
        }
        ImGui::PushStyleColor(ImGuiCol_Text, col);
        ImGui::TextUnformatted(line.text.c_str());
        ImGui::PopStyleColor();
    }
    // Trailing indicator directly below the last (still-executing) cell and its
    // streamed output — the per-cell signal that work is in flight. Clears the
    // moment the session goes idle and the cell's final output has landed.
    if (session_.busy()) running_indicator();
    if (scroll_bottom_) {
        ImGui::SetScrollHereY(1.0f);
        scroll_bottom_ = false;
    }
    ImGui::EndChild();

    // Draggable splitter between log and input.
    ImGui::InvisibleButton("##vsplit", ImVec2(-1.0f, split_h));
    if (ImGui::IsItemActive()) input_height_ -= ImGui::GetIO().MouseDelta.y;
    if (ImGui::IsItemHovered() || ImGui::IsItemActive()) {
        ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
    }

    // Code input.
    ImGui::InputTextMultiline("##input", input_, sizeof input_,
                              ImVec2(-1.0f, input_height_),
                              ImGuiInputTextFlags_AllowTabInput);

    // Ctrl+Enter runs (Enter alone inserts a newline). Checked at window scope
    // with a raw key query — the focused multiline input claims the Enter key
    // when Ctrl is held, so an item-active gate misses it.
    bool run = false;
    if (ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows)) {
        const ImGuiIO& io = ImGui::GetIO();
        run = io.KeyCtrl && (ImGui::IsKeyPressed(ImGuiKey_Enter, false) ||
                             ImGui::IsKeyPressed(ImGuiKey_KeypadEnter, false));
    }

    // Run row. Always enabled — the console needs no open file to run.
    if (ImGui::Button("Run", ImVec2(80.0f, 0.0f))) run = true;
    ImGui::SameLine();
    ImGui::TextDisabled("Ctrl+Enter to run   ·   .? for help");

    if (run) run_current_input();

    // File chooser (standalone open).
    file_browser_.Display();
    if (file_browser_.HasSelected()) {
        open_file(file_browser_.GetSelected());
        file_browser_.ClearSelected();
    }

    // Macro chooser (.x load + run).
    macro_browser_.Display();
    if (macro_browser_.HasSelected()) {
        session_.run_macro(macro_browser_.GetSelected());
        macro_browser_.ClearSelected();
    }

    ImGui::End();
}

} // namespace console
