#ifndef COAST_COASTD_NET_DEFAULTS_H
#define COAST_COASTD_NET_DEFAULTS_H

// Single source of truth for coastd's default ports (dependency-free so both the
// core vocabulary in types.h and the wire contract in protocol.h can share it
// without pulling json). The client façade keeps its OWN copy of the default
// port (coast::coastd::default_port) because its lean, JIT-parsed header cannot
// reach coastd/include — that constant carries a "keep in sync" note.

#include <cstdint>

namespace coastd {

inline constexpr std::uint16_t default_tcp_port = 51900;
inline constexpr std::uint16_t default_udp_port = 51901;

} // namespace coastd

#endif // COAST_COASTD_NET_DEFAULTS_H
