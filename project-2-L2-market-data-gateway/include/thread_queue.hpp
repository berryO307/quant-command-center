#pragma once
#include <queue>
#include <mutex>
#include <condition_variable>
#include <optional>
#include <chrono>

// Simple MPSC (or SPSC) thread-safe queue backed by mutex + condvar.

template <typename T>
class ThreadQueue {
public:
    // Sink argument idiom: Takes 'item' by value to safely handle both lvalues and rvalues.
    void push(T item) {
        {
            // SCOPE BLOCK: Tightly bounds the critical section to minimize lock contention.
            // RAII lock: Automatically acquires the mutex and guarantees release on scope exit.
            std::lock_guard<std::mutex> lk(mutex_);
            
            // Move semantics: O(1) pointer swap instead of deep copying data into the queue.
            queue_.push(std::move(item));
        }
        // Notify AFTER unlock: Prevents the waking consumer thread from instantly hitting a locked mutex.
        cv_.notify_one();
    }

    // Block until an item is available, then return it.
    T pop() {
        // unique_lock: Required because cv_.wait must temporarily unlock the mutex while sleeping.
        std::unique_lock<std::mutex> lk(mutex_);
        
        // Predicate lambda: Defends against OS-level spurious wakeups evaluating on an empty queue.
        cv_.wait(lk, [this] { return !queue_.empty(); });
        
        // Steal data: Transfers ownership of the memory directly out of the queue buffer.
        T item = std::move(queue_.front());
        queue_.pop();
        return item;
    }

    // Non-blocking: returns nullopt if queue is empty.
    // Non-blocking path: Prevents thread stalling; critical if the consumer has other polling duties.
    std::optional<T> try_pop() {
        std::lock_guard<std::mutex> lk(mutex_);
        if (queue_.empty()) return std::nullopt;
        
        T item = std::move(queue_.front());
        queue_.pop();
        return item;
    }

    // Pop with timeout. Returns nullopt on timeout.
    // Timed wait: Prevents infinite thread starvation and allows for graceful system shutdown.
    std::optional<T> pop_for(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lk(mutex_);
        
        if (!cv_.wait_for(lk, timeout, [this] { return !queue_.empty(); }))
            return std::nullopt;
            
        T item = std::move(queue_.front());
        queue_.pop();
        return item;
    }

    // const contract: Promises the caller that checking the size won't alter the queue's state.
    size_t size() const {
        std::lock_guard<std::mutex> lk(mutex_);
        return queue_.size();
    }

private:
    // mutable override: Legally allows a const method to alter the mutex's internal lock state.
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<T> queue_;
};