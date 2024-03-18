#include "${FILENAME}.par.h"

#include <atomic>
#include <barrier>
#include <thread>
#include <vector>
#include <iostream>

extern const size_t numWorkers;

std::barrier<> syncPoint(numWorkers);
volatile bool running = false;
std::vector<std::thread> threads;
extern void (*const f[][2])();

inline void run_iteration(void f(), void g()) {
    syncPoint.arrive_and_wait();
    f();
    syncPoint.arrive_and_wait();
}

void init() {
    running = true;
    auto worker = [&](void f(), void g()) {
        while(running) {
            run_iteration(f, g);
        }
    };
    threads.reserve(numWorkers - 1);
    for(size_t i = 1; i < numWorkers; i++) {
        threads.emplace_back(worker, f[i][0], f[i][1]);
    }
}

void eval() {
    run_iteration(f[0][0], f[0][1]);
    for(size_t i = 0; i < numWorkers; i++) {
        f[i][1]();
    }
}

void stop() {
    // std::cout << "stop" << std::endl;
    syncPoint.arrive_and_wait();
    running = false;
    syncPoint.arrive_and_wait();
    // globalSyncPoint.arrive_and_wait();
    for(auto& t : threads) {
        t.join();
    }
}
#define PARALLEL
#define InitFunc() init()
#define EvalFunc() eval()
#define StopFunc() stop()
#include "${FILENAME}.cpp"