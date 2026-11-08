// console::detail::quote_cpp — produces a C++ string literal that the session
// splices into interpreter source (`auto v = open("<path>")`). The value must
// never break out of the literal, so control bytes (a POSIX path may contain a
// raw newline) have to be escaped, not passed through.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "console/quote_cpp.h"

#include <string>

using console::detail::quote_cpp;

TEST_CASE("quote_cpp: plain text is wrapped in quotes unchanged")
{
    CHECK(quote_cpp("abc") == "\"abc\"");
    CHECK(quote_cpp("") == "\"\"");
    CHECK(quote_cpp("/tmp/run.h5") == "\"/tmp/run.h5\"");
}

TEST_CASE("quote_cpp: backslash and double-quote are escaped")
{
    CHECK(quote_cpp("a\\b") == "\"a\\\\b\"");     // a\b   -> "a\\b"
    CHECK(quote_cpp("a\"b") == "\"a\\\"b\"");     // a"b   -> "a\"b"
    CHECK(quote_cpp("C:\\x") == "\"C:\\\\x\"");   // Windows-style path
}

TEST_CASE("quote_cpp: control bytes become fixed-width octal escapes")
{
    // This is the case the old two-char escaper got wrong: a raw newline in the
    // path terminated the literal and produced uncompilable injected source.
    CHECK(quote_cpp("a\nb") == "\"a\\012b\"");    // newline -> \012
    CHECK(quote_cpp("a\tb") == "\"a\\011b\"");    // tab     -> \011
    CHECK(quote_cpp("a\rb") == "\"a\\015b\"");    // CR      -> \015
    CHECK(quote_cpp(std::string("a\0b", 3)) == "\"a\\000b\""); // embedded NUL

    // Whatever the input, no raw control byte survives into the literal.
    const std::string q = quote_cpp(std::string("x\ny\rz\t\x7f"));
    CHECK(q.find('\n') == std::string::npos);
    CHECK(q.find('\r') == std::string::npos);
    CHECK(q.find('\t') == std::string::npos);
    CHECK(q.find('\x7f') == std::string::npos);
}

TEST_CASE("quote_cpp: an octal escape cannot merge with a following digit")
{
    // \x-style escapes are greedy; octal is capped at 3 digits, so a digit right
    // after a control byte stays a separate character.
    CHECK(quote_cpp("\n7") == "\"\\0127\"");      // newline then '7'
}

TEST_CASE("quote_cpp: high-bit (UTF-8) bytes pass through raw")
{
    // Valid in a narrow literal and must round-trip as the same path bytes.
    const std::string utf8 = "\xC3\xA9";           // 'é'
    CHECK(quote_cpp(utf8) == "\"" + utf8 + "\"");
}
