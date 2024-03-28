#include <barrier>
#include <thread>
#include <vector>

struct runtime {
    using func = void (*)();
    std::barrier<> sync;
    std::vector<std::thread> threads;
    runtime(size_t n_threads, size_t n_cycles, func funcs[][2])
    : sync(n_threads) {
        for(size_t i = 0; i < n_threads; i++) {
            func f = funcs[i][0], g = funcs[i][1];
            threads.emplace_back([&, n_cycles, f, g]() {
                for(size_t j = 0; j < n_cycles; j++) {
                    sync.arrive_and_wait();
                    f();
                    sync.arrive_and_wait();
                    g();
                }
            });
        }
    }
    ~runtime() {
        for(auto &th: threads) {
            th.join();
        }
    }
};
