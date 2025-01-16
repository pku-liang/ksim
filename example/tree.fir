circuit Tree:
  module Tree:
    input clock: Clock
    input reset: UInt<1>
    input in: SInt<16>[4]
    output sum: SInt<16>

    reg r0: SInt<16>, clock
    reg r1: SInt<16>, clock
    reg r2: SInt<16>, clock
    r0 <= add(in[0], in[1])
    r1 <= add(in[2], in[3])
    r2 <= add(r0, r1)
    sum <= r2