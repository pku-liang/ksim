int main(int argc, char ** argv) {
    int cnt = atoi(argv[1]);
    reset = 1;
    for(auto i = SimpleFIR_reset_ahead; i >= 0; i--) {
        SimpleFIR();
        reset = 0;
    }
    auto start = system_clock::now();
    for(auto i = 0; i < cnt; i++) {
        in = i;        // !!! update here
        SimpleFIR();
        printf("cycle=%d in=%d sum=%d\n" , i, in, sum); // !!! update here
    }
    auto stop = system_clock::now();
    // ...
    return 0;
}
