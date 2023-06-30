#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/TypeSwitch.h"

#define GEN_PASS_DEF_SLICEPROPAGATION
#include "PassDetails.h"
#include "ksim/KSimPasses.h"

using namespace ksim;
using namespace circt;
using namespace mlir;
using namespace llvm;

#include "SlicePropagation.cpp.inc"

struct SlicePropagationPass : ksim::impl::SlicePropagationBase<SlicePropagationPass> {
  using ksim::impl::SlicePropagationBase<SlicePropagationPass>::SlicePropagationBase;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    populateWithGenerated(patterns);
    if(failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};

std::unique_ptr<mlir::Pass> ksim::createSlicePropagationPass() {
  return std::make_unique<SlicePropagationPass>();
}