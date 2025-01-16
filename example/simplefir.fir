circuit SimpleFIR:
  module SimpleFIR:
    input clock: Clock
    input reset: UInt<1>
    input in: SInt<16>
    output sum: SInt<16>

    reg r1: SInt<16>, clock
    reg r2: SInt<16>, clock
    reg r3: SInt<16>, clock
    r1 <= in
    r2 <= r1
    r3 <= r2

    sum <= add(add(in, r1), add(r2, r3))
