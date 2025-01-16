#include "VMesh6x6.h"
#include <chrono>
#include <iostream>

double sc_time_stamp() {return 0;}

int main(int argc, char ** argv) {
    auto dut = new VMesh6x6;
    int cnt = atoi(argv[1]);
    dut->io_in_valid_0_0 = 1;
    dut->io_in_valid_1_0 = 1;
    dut->io_in_valid_2_0 = 1;
    dut->io_in_valid_3_0 = 1;
    dut->io_in_valid_4_0 = 1;
    dut->io_in_valid_5_0 = 1;
    dut->io_out_valid_0_0 = 1;
    dut->io_out_valid_1_0 = 1;
    dut->io_out_valid_2_0 = 1;
    dut->io_out_valid_3_0 = 1;
    dut->io_out_valid_4_0 = 1;
    dut->io_out_valid_5_0 = 1;
    auto start = std::chrono::system_clock::now();
    for(int i = 0; i < cnt; i++) {
        dut->clock = 0;
        dut->eval();
        dut->clock = 1;
        dut->eval();
    }
    auto stop = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() << std::endl;
    return 0;
}