#ifndef COAST_COASTD_SOCK_COMPAT_H
#define COAST_COASTD_SOCK_COMPAT_H

#include <sys/socket.h>

#include <cerrno>
#include <string>

// -----------------------------------------------------------------------------
// sock_compat — "a vanished peer must never kill this process", portably.
//
// Writing to a stream socket whose peer has closed raises SIGPIPE, and SIGPIPE's
// DEFAULT disposition terminates the process. Both ends of the wire protocol
// need immunity, and for different reasons:
//   - daemon side: a client that disconnects mid-reply would otherwise kill the
//     parent of every supervised child;
//   - client side: coastd_client is linked into coast_lib, i.e. into the ordinary
//     `coast` binary — an unreachable daemon would take the whole application
//     down, GUI and all.
//
// Linux spells the suppression per CALL (MSG_NOSIGNAL on send); Darwin and the
// BSDs have no such flag and spell it per SOCKET (SO_NOSIGPIPE). Neither name is
// portable. An unguarded MSG_NOSIGNAL does not merely lose the guard on macOS —
// it is an undeclared identifier there, and because coast_lib links coastd_client
// PUBLIC it failed the ENTIRE `coast` build, every preset, GUI and headless
// alike. Hence this shim, and hence its two call sites are the only places in
// the tree that may name either constant.
//
// The switch is on the MACRO, not on __linux__, deliberately: it tests the
// property actually being relied on ("does this platform offer the per-call
// flag?") instead of a proxy for it, so it stays correct on any other libc that
// has or grows MSG_NOSIGNAL. This matches the existing convention in
// proc_util.cpp (cloexec_pipe/_accept/_socket), which likewise degrades a Linux
// atomic-CLOEXEC creator to a portable set-after-create.
//
// Residual: a platform with NEITHER mechanism gets the default disposition back.
// None is supported (or known); a port there must install a SIG_IGN handler.
// -----------------------------------------------------------------------------

namespace coastd {

// Flags for send() on a stream socket. MSG_NOSIGNAL where it exists; 0 elsewhere,
// where set_nosigpipe() below has already made the socket itself immune.
inline constexpr int send_nosignal_flags =
#if defined(MSG_NOSIGNAL)
    MSG_NOSIGNAL;
#else
    0;
#endif

// Make `fd` immune to SIGPIPE on platforms without the per-call flag. Call once
// per connected stream socket, as early as possible after connect()/accept() —
// before the first send(), which is the earliest point SIGPIPE can be raised.
// A no-op where MSG_NOSIGNAL exists. Best-effort and noexcept: a failure here
// leaves the pre-existing default disposition, so there is nothing to recover
// from and nothing the caller could usefully do differently.
inline void set_nosigpipe([[maybe_unused]] int fd) noexcept {
#if !defined(MSG_NOSIGNAL) && defined(SO_NOSIGPIPE)
    if (fd < 0) return;
    const int on = 1;
    ::setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &on, sizeof on);
#endif
}

// Send the whole buffer on a stream socket, retrying partial sends and EINTR.
// false on any hard send error — including a peer that vanished mid-reply,
// which the flags above turn into an error return instead of a SIGPIPE. Shared
// by both wire ends (comms_server and coastd_client), which each carried an
// identical copy of this loop.
inline bool write_all(int fd, const std::string& s) {
    std::size_t off = 0;
    while (off < s.size()) {
        const ssize_t n = ::send(fd, s.data() + off, s.size() - off, send_nosignal_flags);
        if (n < 0) { if (errno == EINTR) continue; return false; }
        off += static_cast<std::size_t>(n);
    }
    return true;
}

} // namespace coastd

#endif // COAST_COASTD_SOCK_COMPAT_H
