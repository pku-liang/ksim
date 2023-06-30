#ifndef SLICE_PROPAGATION_TD
#define SLICE_PROPAGATION_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"
include "circt/Dialect/Comb/Comb.td"
include "circt/Dialect/Seq/Seq.td"

def BuildExtract : NativeCodeCall<
  "$_builder.create<::circt::comb::ExtractOp>($_loc, $1, cast<::circt::comb::ExtractOp>($0.getDefiningOp()).getLowBit(), hw::getBitWidth($0.getType()))"
>;

def BuildExtractVariadic : NativeCodeCall<
  "map_range($1, [&](auto x){"
    "return $_builder.create<::circt::comb::ExtractOp>($_loc, x, cast<::circt::comb::ExtractOp>($0.getDefiningOp()).getLowBit(), hw::getBitWidth($0.getType()));"
  "})"
>;

class ElementWiseUTVariadicPat<UTVariadicOp op> : Pat<
  (ExtractOp:$ex (op $range, $twoState), $lowBit),
  (op (BuildExtractVariadic $ex, $range), $twoState)
>;

def AndPat : ElementWiseUTVariadicPat<AndOp>;
def OrPat  : ElementWiseUTVariadicPat<OrOp>;
def XorPat : ElementWiseUTVariadicPat<XorOp>;

class TruncableUTVariadicPat<UTVariadicOp op> : Pat<
  (ExtractOp:$ex (op $range, $twoState), IntMaxValue<0>:$lowBit),
  (op (BuildExtractVariadic $ex, $range), $twoState)
>;

def AddPat : TruncableUTVariadicPat<AddOp>;
def MulPat : TruncableUTVariadicPat<MulOp>;

class TruncableUTBinPat<UTBinOp op> : Pat<
  (ExtractOp:$ex (op $lhs, $rhs, $twoState), IntMaxValue<0>:$lowBit),
  (op (BuildExtract $ex, $lhs), (BuildExtract $ex, $rhs), $twoState)
>;

def ShlPat : TruncableUTBinPat<ShlOp>;
def ShrPat : TruncableUTBinPat<ShrUOp>;
def SubPat : TruncableUTBinPat<SubOp>;

def MuxPat : Pat<
  (ExtractOp:$ex (MuxOp $cond, $trueValue, $falseValue, $twoState), $lowBit),
  (MuxOp $cond, (BuildExtract $ex, $trueValue), (BuildExtract $ex, $falseValue), $twoState)
  >;

#endif
