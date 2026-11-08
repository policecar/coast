#ifndef COAST_COASTD_LINE_TAILER_H
#define COAST_COASTD_LINE_TAILER_H

// -----------------------------------------------------------------------------
// coastd::line_tailer — turn an arbitrarily-chunked byte stream into complete
// lines. The supervisor tails a child's growing logfile and feeds whatever bytes
// it read; the tailer emits only COMPLETE lines (up to '\n') and retains any
// trailing partial across feeds, so a progress line written mid-read is never
// parsed torn (validated in Demonstrate spike 04, PART A: torn sentinel line
// recovered exactly once, human lines ignored, partial retained).
//
// Header-only and dependency-free: it is a pure buffer transform.
// -----------------------------------------------------------------------------

#include <cstddef>
#include <string>
#include <string_view>

namespace coastd {

class line_tailer {
public:
    // A single retained partial line may not exceed this. A pathological producer
    // that never emits '\n' must not grow the buffer without bound; beyond the cap
    // the over-long line is dropped and parsing resynchronizes at the next '\n'.
    // coast progress lines are tiny, so 1 MiB is only ever hit by a runaway.
    static constexpr std::size_t max_line_bytes = 1u << 20;

    // Append `chunk`, then invoke `on_line(std::string_view)` for each complete
    // line now available (the '\n' is excluded). A trailing partial line stays
    // buffered for the next feed.
    template <class OnLine>
    void feed(std::string_view chunk, OnLine&& on_line) {
        buf_.append(chunk);

        // Recovering from a dropped over-long line: discard bytes until the next
        // line boundary so parsing realigns cleanly rather than mid-line.
        if (resync_) {
            const std::size_t nl = buf_.find('\n');
            if (nl == std::string::npos) {
                if (buf_.size() > max_line_bytes) buf_.clear();  // still no boundary: stay bounded
                return;
            }
            buf_.erase(0, nl + 1);
            resync_ = false;
        }

        // Emit each complete line by advancing a start offset, then erase all
        // consumed bytes in ONE shift — O(n) per feed, not O(n·lines) as a
        // per-line erase(0,…) would be. The views handed to on_line stay valid:
        // buf_ is not mutated until the loop completes.
        std::size_t start = 0;
        for (std::size_t nl; (nl = buf_.find('\n', start)) != std::string::npos; start = nl + 1)
            on_line(std::string_view(buf_.data() + start, nl - start));
        if (start > 0) buf_.erase(0, start);

        if (buf_.size() > max_line_bytes) {  // an unterminated line blew the cap
            buf_.clear();
            resync_ = true;
        }
    }

    // The unterminated remainder (empty if the last byte fed was a newline).
    [[nodiscard]] std::string_view pending() const { return buf_; }

private:
    std::string buf_;
    bool        resync_ = false;  // true while discarding a dropped over-long line
};

} // namespace coastd

#endif // COAST_COASTD_LINE_TAILER_H
