#pragma once
#include <atomic>
#include <cstddef>

template<typename T, size_t N>
class SpscRingBuffer {
    static_assert((N != 0) && ((N & (N - 1)) == 0),
                  "Capacity must be a power of 2");

private:
    static constexpr size_t MASK = N - 1;

    alignas(64) std::atomic<size_t> head_{0};  // written only by producer
    alignas(64) std::atomic<size_t> tail_{0};  // written only by consumer
    alignas(64) T buffer_[N];                  // own cache line(s), away from counters

public:
    // Called by PRODUCER thread only
    bool push(const T& item) {
        const size_t h = head_.load(std::memory_order_relaxed); // ① producer owns head — no sync needed
        const size_t t = tail_.load(std::memory_order_acquire); // ② acquire: see consumer's latest pop

        if (h - t >= N) return false;          // full — subtraction wraps safely for unsigned

        buffer_[h & MASK] = item;              // ③ write BEFORE publishing index

        head_.store(h + 1, std::memory_order_release); // ④ release: consumer now sees the written item
        return true;
    }

    // Called by CONSUMER thread only
    bool pop(T& item) {
        const size_t t = tail_.load(std::memory_order_relaxed); // ① consumer owns tail — no sync needed
        const size_t h = head_.load(std::memory_order_acquire); // ② acquire: see producer's latest push

        if (h == t) return false;              // empty

        item = buffer_[t & MASK];             // ③ read BEFORE publishing index

        tail_.store(t + 1, std::memory_order_release); // ④ release: producer now sees the freed slot
        return true;
    }

    // Approximate — safe for logging/monitoring only
    size_t size() const {
        return head_.load(std::memory_order_relaxed)
             - tail_.load(std::memory_order_relaxed);
    }

    bool empty() const { return size() == 0; }
    bool full()  const { return size() >= N; }
};