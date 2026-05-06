#pragma once
#include <iostream>
#include <string>
#include <thread>

#ifdef _WIN32
    #include <windows.h>
#else
    #include <pthread.h>
    #include <sched.h>
#endif

// pin_thread_self()
// Called from INSIDE the thread that wants to be pinned.
// Uses GetCurrentThread() on Windows — always returns a valid pseudo-handle
// for the calling thread, no pthread→HANDLE conversion required.
// This avoids the winpthreads limitation where pthread_getw32threadhandle_np
// is not available (that function only exists in the older pthreads-win32 lib).
// On Linux, uses pthread_setaffinity_np on the calling thread's handle.

// Internal configuration: Thread sets its own affinity and priority
inline void configure_self_high_performance(int core_id, const std::string& name) {
    std::cout << "[thread] Configuring " << name << " on Core " << core_id << "...\n";
#ifdef _WIN32
    // 1. Set Affinity for the calling thread
    HANDLE hThread = GetCurrentThread();
    DWORD_PTR mask = (static_cast<DWORD_PTR>(1) << core_id);
    if (!SetThreadAffinityMask(hThread, mask)) {
        std::cerr << "   [!] Failed Affinity. Error: " << GetLastError() << "\n";
    }

    // 2. Set Priority to Time Critical
    if (!SetThreadPriority(hThread, THREAD_PRIORITY_TIME_CRITICAL)) {
        std::cerr << "   [!] Failed Priority. Error: " << GetLastError() << "\n";
    }
#else
    // Linux Implementation
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

    sched_param sch_params;
    sch_params.sched_priority = sched_get_priority_max(SCHED_FIFO);
    pthread_setschedparam(pthread_self(), SCHED_FIFO, &sch_params);
#endif
}

inline void pin_thread_self(int core_id, const std::string& name = "") {
#ifdef _WIN32
    HANDLE self = GetCurrentThread();   // always valid — no conversion needed

    DWORD_PTR mask      = 1ULL << core_id;
    DWORD_PTR prev_mask = SetThreadAffinityMask(self, mask);

    if (prev_mask == 0) {
        std::cerr << "[pin] WARNING: SetThreadAffinityMask failed for '"
                  << name << "' on core " << core_id
                  << " (err=" << GetLastError() << ") — thread will float freely\n";
        return;
    }

    std::cout << "[pin] '" << name << "' pinned to core " << core_id
              << " (prev_mask=0x" << std::hex << prev_mask << std::dec << ")\n";

#else
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);

    int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        std::cerr << "[pin] WARNING: pthread_setaffinity_np failed for '"
                  << name << "' on core " << core_id
                  << " (rc=" << rc << ") — thread will float freely\n";
        return;
    }
    std::cout << "[pin] '" << name << "' pinned to core " << core_id << "\n";
#endif
}

// verify_pin_self()
// Called from INSIDE the thread to verify it's running on the expected core.
// On Windows: reads current affinity via SetThreadAffinityMask read-then-restore idiom
//             (there is no GetThreadAffinityMask for a single thread in Win32).
// On Linux:   reads via pthread_getaffinity_np.

inline void verify_pin_self(int expected_core, const std::string& name = "") {
#ifdef _WIN32
    HANDLE self = GetCurrentThread();

    DWORD_PTR expected_mask = 1ULL << expected_core;

    // Read-then-restore: SetThreadAffinityMask returns the previous mask on success.
    // Set to expected, capture old, restore old. Standard Win32 idiom for reading affinity.
    DWORD_PTR actual_mask = SetThreadAffinityMask(self, expected_mask);
    if (actual_mask == 0) {
        std::cerr << "[pin] WARNING: verify_pin_self read failed for '"
                  << name << "' (err=" << GetLastError() << ")\n";
        return;
    }
    SetThreadAffinityMask(self, actual_mask);   // restore original

    if (actual_mask == expected_mask) {
        std::cout << "[pin] verified '" << name << "' on core " << expected_core << "\n";
    } else {
        std::cerr << "[pin] WARNING: '" << name << "' expected mask=0x"
                  << std::hex << expected_mask
                  << " actual mask=0x" << actual_mask << std::dec
                  << " — OS may have overridden affinity\n";
    }

#else
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    int rc = pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        std::cerr << "[pin] WARNING: pthread_getaffinity_np failed for '" << name << "'\n";
        return;
    }
    if (!CPU_ISSET(expected_core, &cpuset)) {
        std::cerr << "[pin] WARNING: '" << name << "' is NOT on core "
                  << expected_core << " — affinity may have been overridden\n";
    } else {
        std::cout << "[pin] verified '" << name << "' on core " << expected_core << "\n";
    }
#endif
}

// Legacy external-pinning wrappers (kept for API compatibility).
// These work on Linux but on Windows/winpthreads the pthread_t→HANDLE
// conversion is unreliable. Use pin_thread_self() from inside the thread instead.

inline void pin_thread(std::thread& t, int core_id, const std::string& name = "") {
#ifdef _WIN32
    // External pinning is unreliable on winpthreads — delegate to self-pin.
    // This is a no-op here; the thread must call pin_thread_self() internally.
    (void)t; (void)core_id;
    std::cout << "[pin] '" << name << "': external pin skipped on Windows"
              << " — thread calls pin_thread_self() internally\n";
#else
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    int rc = pthread_setaffinity_np(t.native_handle(), sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        std::cerr << "[pin] WARNING: pthread_setaffinity_np failed for '"
                  << name << "' on core " << core_id
                  << " (rc=" << rc << ") — thread will float freely\n";
        return;
    }
    std::cout << "[pin] '" << name << "' pinned to core " << core_id << "\n";
#endif
}

inline void verify_pin(std::thread& t, int expected_core, const std::string& name = "") {
#ifdef _WIN32
    (void)t;
    (void)expected_core;
    std::cout << "[pin] '" << name << "': external verify skipped on Windows"
              << " — thread verifies itself via verify_pin_self()\n";
#else
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    int rc = pthread_getaffinity_np(t.native_handle(), sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        std::cerr << "[pin] WARNING: pthread_getaffinity_np failed for '" << name << "'\n";
        return;
    }
    if (!CPU_ISSET(expected_core, &cpuset)) {
        std::cerr << "[pin] WARNING: '" << name << "' is NOT on core " << expected_core << "\n";
    } else {
        std::cout << "[pin] verified '" << name << "' on core " << expected_core << "\n";
    }
#endif
}