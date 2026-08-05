# Restart-and-replay recovery

Product recovery replaces a failed route and reconstructs a fresh session by replaying verified
history. It is not seamless failover and does not migrate KV cache.

## Detected failure classes

The coordinator and worker paths surface:

- worker-process termination and heartbeat expiry;
- control-RPC refusal, reset, or timeout;
- active stage-ring EOF, reset, broken pipe, integrity failure, or response timeout;
- stage execution or queue failure;
- route/session/request-generation mismatch;
- token-publication rejection or timeout; and
- explicit cancellation while running or recovering.

Closing only a listening socket does not prove active-dependency failure: an established
connection can continue. The deterministic product test injects closure into the connection
carrying a selected decode frame before its write and records the complete frame context.

## Recovery sequence

For a recoverable greedy request:

1. the active request reports the exact failing session, endpoint, route generation, token
   position, and sequence;
2. token acceptance from that generation is disabled;
3. the generation is retired and the implicated worker is marked unhealthy for replacement;
4. the coordinator selects exact-compatible trusted workers;
5. missing stages are loaded and a higher signed route generation is installed and peer-verified;
6. a fresh session and request generation are opened on every stage;
7. the original prompt is replayed;
8. every previously accepted greedy token is fed back through the new ring;
9. each replay result is compared with the durable accepted prefix;
10. replay token publications and client events are suppressed; and
11. generation resumes only at the first unaccepted token.

Old-generation responses and publications are rejected after the generation boundary. A failed
or timed-out pooled connection is evicted so retries cannot keep using a poisoned socket.

## Accepted-prefix verification

Durable state records prompt tokens, accepted generated token IDs, positions, request and route
generations, and checksums. Replay requires exact greedy token identity. The first mismatch marks
the request failed with a replay-divergence error; it is never averaged, hidden, or emitted as a
new client token.

Duplicate publications for the same request generation, position, and token are idempotently
acknowledged without a second stream event. A conflicting token at an accepted position fails
closed. The deterministic socket test therefore asserts both the exact final token sequence and
one client event per position.

## Cancellation during recovery

Cancellation uses the same bounded cleanup path in running and recovering states. It disables
further acceptance, cancels/cleans sessions on reachable stages, releases session-local KV state,
and leaves shared resident model stages loaded. Cancellation is idempotent and does not count as
a worker reliability failure.

If cancellation races replacement, the coordinator must not publish recovery completion or
resume generation. Unreachable stage cleanup is bounded; its failure is reported rather than
blocking shutdown indefinitely.

## Trust removal during recovery

An existing active route may finish after a trust-store removal. Recovery, however, installs a
new route and is therefore blocked if any selected worker is no longer trusted. New registration,
deployment, and admission are also blocked. Operators needing immediate removal should cancel
affected sessions before revoking trust.

## Limitations

- Only greedy accepted-prefix replay is currently proven.
- Recovery recomputes prompt and accepted tokens, increasing latency and work.
- The coordinator itself is not highly available.
- Durable evidence does not resurrect live sockets after coordinator restart; workers re-register.
- There is no KV checkpoint transfer, transparent live migration, or seamless failover.
- Repeated failures are bounded by configured attempts and timeouts.
- A malicious but correctly keyed worker can still withhold service or return plausible bad
  computation; exact replay detects divergence only where a verified reference/prefix exists.

Periodic distributed KV checkpointing is future work only. It must not be described as an
existing capability or used to reinterpret restart-and-replay measurements.
