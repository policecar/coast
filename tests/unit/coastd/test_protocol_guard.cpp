#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <string>

#include "daemon/protocol.h"

using namespace coastd;

// The load-bearing anti-crash guard at every network edge, which had NO test.
//
// json_depth_ok exists because over-deep nesting is a STACK OVERFLOW in the DOM
// parser, not a parse error: allow_exceptions=false does not help, because the
// process dies before any error can be returned. All three edges that read
// untrusted bytes depend on it — the daemon's request reader, the client's reply
// reader, and discover()'s handling of broadcast datagrams from any LAN host — so
// a regression here is a remote crash of either side, and nothing else in the
// suite would notice.

TEST_CASE("json_depth_ok accepts ordinary payloads") {
    CHECK(proto::json_depth_ok(R"({"cmd":"list"})"));
    CHECK(proto::json_depth_ok(R"({"cmd":"submit","config_text":"{\"a\":[1,2,3]}"})"));
    CHECK(proto::json_depth_ok("{}"));
    CHECK(proto::json_depth_ok(""));
    // A realistic nested config: objects inside arrays inside objects.
    CHECK(proto::json_depth_ok(R"({"a":{"b":[{"c":[{"d":1}]}]}})"));
}

TEST_CASE("json_depth_ok rejects nesting past the limit") {
    // Exactly at the limit passes; one deeper is refused. Testing the BOUNDARY is
    // the point — a guard that only rejects absurd input would still let a
    // carefully-sized payload through.
    const std::string at_limit(proto::max_json_depth, '[');
    CHECK(proto::json_depth_ok(at_limit));

    const std::string past_limit(proto::max_json_depth + 1, '[');
    CHECK_FALSE(proto::json_depth_ok(past_limit));

    // The shape an attacker actually sends: tens of thousands of brackets.
    CHECK_FALSE(proto::json_depth_ok(std::string(60000, '[')));
    CHECK_FALSE(proto::json_depth_ok(std::string(60000, '{')));
}

TEST_CASE("json_depth_ok ignores brackets INSIDE strings") {
    // The scan is a byte walk, so it must honour string quoting — otherwise a
    // submitted config (which travels as a JSON *string* full of escaped brackets)
    // would be rejected as too deep. This is the false-positive direction, and it
    // would break the daemon's main use rather than merely weaken a guard.
    std::string s = R"({"config_text":")";
    s += std::string(200, '[');     // brackets inside a string: not structure
    s += R"("})";
    CHECK(proto::json_depth_ok(s));

    // An escaped quote must not be read as ending the string.
    CHECK(proto::json_depth_ok(R"({"a":"he said \"[[[[\" and left"})"));
}

TEST_CASE("parse_line is tolerant and never throws") {
    // Well-formed.
    const auto ok = proto::parse_line(R"({"cmd":"list"})");
    REQUIRE(ok.has_value());
    CHECK(ok->cmd == "list");

    // Every malformed shape returns nullopt rather than throwing — the documented
    // contract, since this runs at the transport edge before anything can catch.
    CHECK_FALSE(proto::parse_line("").has_value());
    CHECK_FALSE(proto::parse_line("not json").has_value());
    CHECK_FALSE(proto::parse_line("[1,2,3]").has_value());          // not an object
    CHECK_FALSE(proto::parse_line("42").has_value());               // not an object
    CHECK_FALSE(proto::parse_line(R"({"nocmd":1})").has_value());   // missing cmd
    CHECK_FALSE(proto::parse_line(R"({"cmd":""})").has_value());    // blank cmd
    CHECK_FALSE(proto::parse_line(std::string(60000, '[')).has_value());  // depth guard

    // The regression: a PRESENT but non-string "cmd". json::value() substitutes its
    // default only for an ABSENT key and THROWS type_error.302 here, which the
    // daemon's blanket handler reported as "internal error" — the wrong message for
    // a malformed field, and a trap for any caller trusting "never throws".
    CHECK_NOTHROW((void)proto::parse_line(R"({"cmd":1})"));
    CHECK_FALSE(proto::parse_line(R"({"cmd":1})").has_value());
    CHECK_FALSE(proto::parse_line(R"({"cmd":null})").has_value());
    CHECK_FALSE(proto::parse_line(R"({"cmd":["list"]})").has_value());
    CHECK_FALSE(proto::parse_line(R"({"cmd":{"a":1}})").has_value());
}

TEST_CASE("every reply envelope carries the protocol version") {
    // A client checks this BEFORE trusting a reply, so it must be present on the
    // failure envelope too — a rejection is exactly when "am I talking to the wrong
    // vintage of daemon?" matters most.
    const auto good = proto::ok(nlohmann::json{{"id", "j1"}});
    CHECK(good["ok"] == true);
    CHECK(good["proto_version"] == proto::proto_version);
    CHECK(good["id"] == "j1");

    const auto bad = proto::err("nope");
    CHECK(bad["ok"] == false);
    CHECK(bad["proto_version"] == proto::proto_version);
}

TEST_CASE("version_complaint speaks only when there is something to say") {
    CHECK(proto::version_complaint(proto::proto_version).empty());
    CHECK_FALSE(proto::version_complaint(0).empty());                      // pre-versioning
    CHECK_FALSE(proto::version_complaint(proto::proto_version + 1).empty());  // newer daemon
}
