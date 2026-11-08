#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <string>

#include <nlohmann/json.hpp>

#include "daemon/job_json.h"

using namespace coastd;

namespace {
// EVERY field distinctively non-default. The round-trip test below compares
// SERIALIZED FORMS, so a field left at its default here would round-trip
// "correctly" even if parse_job never read it — which is exactly how the
// no_checkpoint_launches / last_launch_ms drop (H1) survived the old test.
job sample() {
    job j;
    j.coastd_version   = "0.1.0-dev";
    j.id               = "20260720T100000Z_mnist_9f3a1c";
    j.submitted_at     = "2026-07-20T10:00:00Z";
    j.user_message     = "overnight run";
    j.ref_symbolic     = "feat/coastd";
    j.ref_sha          = "a1b2c3d4e5";
    j.run_dir          = "/data/coastd/runs/20260720T100000Z_mnist_9f3a1c";
    j.steps_target     = 500000;
    j.steps_done       = 120000;
    j.checkpoint       = "snapshots/mnist_20260720_101500_120000.h5";
    j.priority         = 3;
    j.cancel_requested = true;
    j.status           = job_status::running;
    j.pid              = 48213;
    j.start_token      = "1234567";
    j.launch_failures  = 2;
    j.no_checkpoint_launches = 1;
    j.last_launch_ms   = 1753008000000ULL;   // epoch ms, 2025-07-20T10:40:00Z
    j.failure_reason   = "crash-loop: 3 consecutive launches with no forward progress";
    j.load_state_from  = "20260719T090000Z_train_1a2b3c";
    j.load_state       = "/data/coastd/runs/20260719T090000Z_train_1a2b3c/final_state.h5";
    j.load_state_step  = 50000;
    return j;
}
} // namespace

// Written against the SERIALIZED FORM, not a hand-enumerated field list. A list
// of CHECKs can only test the fields whoever wrote it remembered; this cannot
// miss a field, and — the point — cannot miss a field ADDED LATER: any key
// dump_job writes but parse_job does not read back shows up as a text diff.
TEST_CASE("job_json: round-trip preserves every persisted field") {
    const job in = sample();

    const std::string once  = dump_job(in);
    const std::string twice = dump_job(parse_job(once));

    INFO("dump -> parse -> dump must be a fixed point; a difference names the\n"
         "field parse_job drops (or writes back differently):\n"
         "--- first dump ---\n" << once << "\n--- second dump ---\n" << twice);
    CHECK(once == twice);

    // Guard the guard: the fixed point above is only meaningful if dump_job
    // actually emits a populated file. A dump_job that wrote "{}" would also be
    // a fixed point.
    const nlohmann::json o = nlohmann::json::parse(once);
    CHECK(o.size() >= 23);                       // every field of `job`, none silently dropped
    CHECK(o.at("id") == in.id);                  // and it is THIS job, not a default-constructed one
}

TEST_CASE("job_json: status tokens map both ways") {
    CHECK(status_from(status_name(job_status::queued))     == job_status::queued);
    CHECK(status_from(status_name(job_status::running))    == job_status::running);
    CHECK(status_from(status_name(job_status::quarantine)) == job_status::quarantine);
    CHECK(status_from("not-a-status") == job_status::queued);  // unknown -> safe default
}

TEST_CASE("job_json: tolerant additive — unknown keys ignored, missing keys default") {
    // A file from a FUTURE additive schema (same version, extra key) parses fine.
    nlohmann::json o;
    o["schema_version"] = 1;
    o["id"]             = "j1";
    o["steps_target"]   = 42;
    o["future_field"]   = "ignored";   // unknown -> ignored
    // note: many fields omitted -> take defaults
    const job j = parse_job(o.dump());
    CHECK(j.id           == "j1");
    CHECK(j.steps_target == 42);
    CHECK(j.steps_done   == 0);          // missing -> default
    CHECK(j.ref_sha.empty());            // missing -> default
    CHECK(j.status       == job_status::queued);
}

TEST_CASE("job_json: an unsupported newer schema_version throws (never silently coerced)") {
    nlohmann::json o;
    o["schema_version"] = job_schema_version + 1;
    o["id"]             = "future";
    CHECK_THROWS_AS((void)parse_job(o.dump()), job_schema_error);
    try {
        (void)parse_job(o.dump());
    } catch (const job_schema_error& e) {
        CHECK(e.found == job_schema_version + 1);
    }
}

TEST_CASE("job_json: malformed JSON throws (store will quarantine)") {
    CHECK_THROWS((void)parse_job("{ this is not json"));
}

// ---- regression: present-but-null / wrong-typed fields are TOLERATED -----------
// nlohmann's value() throws on a present-but-null or wrong-typed field; parse_job
// must default such a field instead, so one sloppy key never quarantines an
// otherwise-live job (which would orphan its child). Only unparseable JSON or a
// future schema_version may throw. (review 2026-07-21)
TEST_CASE("job_json: a null field defaults instead of throwing") {
    nlohmann::json o;
    o["schema_version"] = job_schema_version;
    o["id"]             = "j1";
    o["checkpoint"]     = nullptr;            // explicit null
    o["failure_reason"] = nullptr;
    o["status"]         = nullptr;
    job j;
    CHECK_NOTHROW(j = parse_job(o.dump()));
    CHECK(j.checkpoint.empty());
    CHECK(j.failure_reason.empty());
    CHECK(j.status == job_status::queued);    // null status -> safe default
}

TEST_CASE("job_json: a wrong-typed scalar defaults instead of throwing") {
    nlohmann::json o;
    o["schema_version"] = job_schema_version;
    o["id"]             = "j1";
    o["priority"]       = "high";             // string where an int is expected
    o["steps_done"]     = "lots";
    job j;
    CHECK_NOTHROW(j = parse_job(o.dump()));
    CHECK(j.priority == 0);                   // defaulted, not quarantined
    CHECK(j.steps_done == 0);
}

// The case above passes even with the defect this one covers, which is why it did
// not catch it: a STRING where a number is expected is one of the few things
// nlohmann actually throws for. Corrupt NUMBERS are silently coerced instead
// (-1 -> 2^64-1, 3.7 -> 3), so the value reaching the daemon was not the safe
// default but a plausible wrong one. Every field here is load-bearing:
// steps_done >= steps_target is what reconcile calls "finished".
TEST_CASE("job_json: a corrupt NUMBER defaults instead of silently coercing") {
    nlohmann::json o;
    o["schema_version"] = job_schema_version;
    o["id"]             = "j1";
    o["steps_done"]     = -1;              // would coerce to 18446744073709551615
    o["steps_target"]   = -7;              // ... turning a bounded run unbounded
    o["priority"]       = 3.7;             // would truncate to 3
    o["pid"]            = 2.5;             // would truncate to 2 -> signals a stranger
    o["launch_failures"] = 1ull << 40;     // out of range for int
    o["cancel_requested"] = 1;             // int where a bool is expected

    job j;
    CHECK_NOTHROW(j = parse_job(o.dump()));
    CHECK(j.steps_done == 0);
    CHECK(j.steps_target == 0);
    CHECK(j.priority == 0);
    CHECK(j.pid == 0);
    CHECK(j.launch_failures == 0);
    CHECK(j.cancel_requested == false);
}

// The counterweight: tightening the type check must not start rejecting values
// that are perfectly legal. A guard that defaults EVERYTHING would pass the test
// above and be useless.
TEST_CASE("job_json: well-formed numbers still round-trip after the type check") {
    nlohmann::json o;
    o["schema_version"] = job_schema_version;
    o["id"]             = "j1";
    o["steps_done"]     = 42u;
    o["steps_target"]   = 1000000u;
    o["priority"]       = -5;              // negative priority is legal (signed int)
    o["pid"]            = 31337;
    o["launch_failures"] = 3;
    o["cancel_requested"] = true;

    job j;
    CHECK_NOTHROW(j = parse_job(o.dump()));
    CHECK(j.steps_done == 42);
    CHECK(j.steps_target == 1000000);
    CHECK(j.priority == -5);
    CHECK(j.pid == 31337);
    CHECK(j.launch_failures == 3);
    CHECK(j.cancel_requested == true);
}

// ---- additive schema evolution ----------------------------------------------
// The header documents the job schema as ADDITIVE. That promise is only real if a
// build that does not model a field PRESERVES it: parse rebuilds the job from a
// fixed key list and dump writes only those keys, so without carry-through an
// older daemon in a mixed fleet — or one binary rolled back — silently deletes
// every field a newer daemon added at the same schema_version, on the first
// status update.
TEST_CASE("job_json: unknown fields survive a parse/dump round-trip") {
    nlohmann::json o;
    o["schema_version"] = job_schema_version;
    o["id"]             = "j1";
    o["steps_done"]     = 7u;
    o["a_future_field"] = "from a newer daemon";
    o["a_future_count"] = 99u;
    o["a_future_object"] = nlohmann::json{{"nested", true}};

    const job j = parse_job(o.dump());
    const nlohmann::json back = nlohmann::json::parse(dump_job(j));

    CHECK(back["a_future_field"] == "from a newer daemon");
    CHECK(back["a_future_count"] == 99u);
    CHECK(back["a_future_object"]["nested"] == true);
    // The modelled fields are still authoritative and unaffected.
    CHECK(back["id"] == "j1");
    CHECK(back["steps_done"] == 7u);
}

TEST_CASE("job_json: a preserved field never shadows one this build owns") {
    // Belt and braces: extras are re-emitted only for keys the dump did not write,
    // so a stale carried value can never override this build's own state.
    nlohmann::json o;
    o["schema_version"] = job_schema_version;
    o["id"]             = "j1";
    o["steps_done"]     = 5u;
    job j = parse_job(o.dump());
    j.extra_fields = nlohmann::json{{"steps_done", 999u}, {"genuinely_new", 1u}}.dump();

    const nlohmann::json back = nlohmann::json::parse(dump_job(j));
    CHECK(back["steps_done"] == 5u);          // ours wins
    CHECK(back["genuinely_new"] == 1u);       // theirs still carried
}

// A schema_version that is PRESENT but not a number used to fall through get_or's
// type check to the default of 1, pass the newer-than-me gate, default every
// unrecognized field, and be rewritten AS v1 by the next put() — the silent
// mangling job_schema_error exists to prevent.
TEST_CASE("job_json: a non-integer schema_version is refused, not defaulted") {
    nlohmann::json o;
    o["id"] = "j1";

    o["schema_version"] = "2";                 // string, not a number
    CHECK_THROWS_AS((void)parse_job(o.dump()), job_schema_error);

    o["schema_version"] = 2.5;                 // float
    CHECK_THROWS_AS((void)parse_job(o.dump()), job_schema_error);

    o.erase("schema_version");                 // ABSENT is still fine: a pre-versioning file
    CHECK_NOTHROW((void)parse_job(o.dump()));
}
