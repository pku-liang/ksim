
#include <atomic>
#include <barrier>
#include <thread>
#include <vector>
#include <iostream>

extern const size_t numWorkers;

std::barrier<> globalSyncPoint(numWorkers + 1);
std::barrier<> localSyncPoint(numWorkers);
volatile bool running = false;

std::vector<std::thread> threads;
extern void (*const f[][2])();

void init() {
    running = true;
    threads.reserve(numWorkers);
    auto worker = [&](size_t i, void f(), void g()) {
        while(true) {
            globalSyncPoint.arrive_and_wait();
            f();
            localSyncPoint.arrive_and_wait();
            g();
            globalSyncPoint.arrive_and_wait();
            if(!running) break;
        }
    };
    for(size_t i = 0; i < numWorkers; i++) {
        threads.emplace_back(worker, i, f[i][0], f[i][1]);
    }
}

void eval() {
    globalSyncPoint.arrive_and_wait();
    globalSyncPoint.arrive_and_wait();
}

void stop() {
    // std::cout << "stop" << std::endl;
    globalSyncPoint.arrive_and_wait();
    running = false;
    globalSyncPoint.arrive_and_wait();
    for(auto& t : threads) {
        t.join();
    }
}
#define PARALLEL
#define InitFunc() init()
#define EvalFunc() eval()
#define StopFunc() stop()