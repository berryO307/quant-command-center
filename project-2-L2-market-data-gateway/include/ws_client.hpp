#pragma once
#include "types.hpp"
#include "spsc_ring_buffer.hpp"
#include "rdtsc.hpp"
#include <simdjson.h>
#include <string>
#include <atomic>
#include <chrono>

// Synchronous Boost.Beast WSS client for Binance Futures combined stream.
// Runs in its own thread (call run() from a std::thread).
// Low-latency design: shared dependencies by reference and in-place Tick population 
// to minimize copying and synchronization overhead.

class WsClient {
public:
    static constexpr int  MAX_RECONNECT_ATTEMPTS = 0; 
    static constexpr auto RECONNECT_BASE_DELAY   = std::chrono::seconds(1);
    static constexpr auto RECONNECT_MAX_DELAY    = std::chrono::seconds(30);
    static constexpr int  RECONNECT_BACKOFF_MULT = 2;

    // Takes dependencies by reference to extend their lifespan without taking ownership.
    WsClient(SpscRingBuffer<Tick, 1024>& queue,
             std::atomic<bool>& stop_flag,
             LatencyStore& latency, 
             std::atomic<uint64_t>& last_u);

    // Connect, handshake, and run the read loop until stop_flag is set
    // or an unrecoverable error occurs.
    void run(const std::string& symbol, const std::string& stream_suffix);

private:
    SpscRingBuffer<Tick, 1024>& queue_;
    std::atomic<bool>& stop_flag_;
    LatencyStore& latency_; // For latency measurement and diagnostics
    std::atomic<uint64_t>&      last_u_;
    std::string                 symbol_;
    std::atomic<uint64_t>       reconnect_count_{0};
    std::string                 stream_suffix_;

    // Reusable scratch buffers — never reallocated, just cleared and reused.
    // Sized for orderbook.200 worst case (200 levels per side).
    std::vector<PriceLevel> scratch_bids_;
    std::vector<PriceLevel> scratch_asks_;

    // Internal methods promised to exist in the .cpp file
    void connect_and_read();
    void trigger_resync();

    // Takes raw_msg by const& to read the network buffer without copying the JSON payload.
    void dispatch(simdjson::padded_string_view message);

    // Output parameter (Tick&): writes directly into the queue's memory slot, avoiding return-by-value copies.
    bool parse_depth(simdjson::dom::element data, Tick& tick, bool is_snapshot = false);
    bool parse_agg_trade(simdjson::dom::element data, Tick& tick);

    // Single parser instance reused across all messages to prevent per-tick memory allocations.
    simdjson::dom::parser parser_;
};