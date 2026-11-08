#ifndef COAST_CONSOLE_QUOTE_CPP_H
#define COAST_CONSOLE_QUOTE_CPP_H

#include <cstdio>
#include <string>

namespace console::detail {

// A double-quoted C++ string literal for `s`, safe to splice into interpreter
// source such as `auto v = open("<s>")`.
//
// Escapes backslash and quote so the value cannot break out of the literal, and
// octal-escapes C0/DEL control bytes — a POSIX path may legally contain any byte
// but '/' and NUL, including a raw newline, which would otherwise terminate the
// literal and yield an uncompilable statement. The escape is fixed-width 3-digit
// octal (`\NNN`) rather than `\xNN`: a hex escape is greedy and would swallow a
// following hex digit of the path, whereas an octal escape takes at most 3
// digits, so it can never merge with the next character. Bytes >= 0x80 (UTF-8
// continuation/lead bytes) are left raw — they are valid in a narrow literal and
// round-trip as the same path bytes.
inline std::string quote_cpp(const std::string& s)
{
    std::string out = "\"";
    for (char c : s) {
        const auto uc = static_cast<unsigned char>(c);
        if (c == '\\' || c == '"') {
            out.push_back('\\');
            out.push_back(c);
        } else if (uc < 0x20 || uc == 0x7f) {
            char buf[5];
            std::snprintf(buf, sizeof buf, "\\%03o", uc);
            out.append(buf);
        } else {
            out.push_back(c);
        }
    }
    out.push_back('"');
    return out;
}

} // namespace console::detail

#endif // COAST_CONSOLE_QUOTE_CPP_H
