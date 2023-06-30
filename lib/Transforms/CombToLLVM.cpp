#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Support/Namespace.h"
#include "ksim/KSimDialect.h"
#include "ksim/KSimPasses.h"

#include "circt/Conversion/HWToLLVM.h"
#include "circt/Conversion/CombToLLVM.h"

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define GEN_PASS_DEF_COMBTOLLVM
#include "PassDetails.h"

using namespace ksim;
using namespace circt;
using namespace mlir;
using namespace llvm;

struct ShrUPattern : OpRewritePattern<comb::ShrUOp> {
  using OpRewritePattern<comb::ShrUOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(comb::ShrUOp op, PatternRewriter & rewriter) const final {
    auto type = op.getType();
    auto width = hw::getBitWidth(type);
    auto constW_1 = rewriter.create<LLVM::ConstantOp>(op.getLoc(), type, width - 1);
    auto constZero = rewriter.create<LLVM::ConstantOp>(op.getLoc(), type, 0);
    auto validShr = rewriter.create<LLVM::ICmpOp>(op.getLoc(), LLVM::ICmpPredicate::ule, op.getRhs(), constW_1);
    auto shr = rewriter.create<LLVM::LShrOp>(op.getLoc(), op.getLhs(), op.getRhs());
    auto repl = rewriter.create<LLVM::SelectOp>(op.getLoc(), validShr, shr, constZero);
    rewriter.replaceOp(op, ValueRange(repl.getResult()));
    return success();
  }
};

struct ShrSPattern : OpRewritePattern<comb::ShrSOp> {
  using OpRewritePattern<comb::ShrSOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(comb::ShrSOp op, PatternRewriter & rewriter) const final {
    auto type = op.getType();
    auto width = hw::getBitWidth(type);
    auto constW_1 = rewriter.create<LLVM::ConstantOp>(op.getLoc(), type, width - 1);
    auto validShr = rewriter.create<LLVM::ICmpOp>(op.getLoc(), LLVM::ICmpPredicate::ule, op.getRhs(), constW_1);
    auto shr = rewriter.create<LLVM::AShrOp>(op.getLoc(), op.getLhs(), op.getRhs());
    auto shrW_1 = rewriter.create<LLVM::AShrOp>(op.getLoc(), op.getLhs(), constW_1);
    auto repl = rewriter.create<LLVM::SelectOp>(op.getLoc(), validShr, shr, shrW_1);
    rewriter.replaceOp(op, ValueRange(repl.getResult()));
    return success();
  }
};

struct ShlPattern : OpRewritePattern<comb::ShlOp> {
  using OpRewritePattern<comb::ShlOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(comb::ShlOp op, PatternRewriter & rewriter) const final {
    auto type = op.getType();
    auto width = hw::getBitWidth(type);
    auto constW_1 = rewriter.create<LLVM::ConstantOp>(op.getLoc(), type, width - 1);
    auto constZero = rewriter.create<LLVM::ConstantOp>(op.getLoc(), type, 0);
    auto validShr = rewriter.create<LLVM::ICmpOp>(op.getLoc(), LLVM::ICmpPredicate::ule, op.getRhs(), constW_1);
    auto shr = rewriter.create<LLVM::ShlOp>(op.getLoc(), op.getLhs(), op.getRhs());
    auto repl = rewriter.create<LLVM::SelectOp>(op.getLoc(), validShr, shr, constZero);
    rewriter.replaceOp(op, ValueRange(repl.getResult()));
    return success();
  }
};

struct CombToLLVMPass: ksim::impl::CombToLLVMBase<CombToLLVMPass> {
  using CombToLLVMBase<CombToLLVMPass>::CombToLLVMBase;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }
  void runOnOperation() final {
    {
      RewritePatternSet patterns(&getContext());
      patterns.add<ShrUPattern, ShrSPattern, ShlPattern>(&getContext());
      if(failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
        return signalPassFailure();
    }
    {
      RewritePatternSet patterns(&getContext());
      auto converter = LLVMTypeConverter(&getContext());
      populateHWToLLVMTypeConversions(converter);
      ConversionTarget target(getContext());
      target.addLegalOp<UnrealizedConversionCastOp>();
      target.addLegalOp<mlir::ModuleOp>();
      target.addLegalDialect<LLVM::LLVMDialect>();
      target.addIllegalDialect<func::FuncDialect>();
      target.addIllegalDialect<comb::CombDialect>();
      target.addIllegalDialect<hw::HWDialect>();
      Namespace globals;
      llvm::DenseMap<std::pair<mlir::Type, ArrayAttr>, mlir::LLVM::GlobalOp> constAggregateGlobalsMap;
      populateFuncToLLVMConversionPatterns(converter, patterns);
      populateHWToLLVMConversionPatterns(converter, patterns, globals, constAggregateGlobalsMap);
      populateCombToLLVMConversionPatterns(converter, patterns);
      if (failed(applyFullConversion(getOperation(), target, std::move(patterns))))
        return signalPassFailure();
      patterns.clear();
      mlir::populateReconcileUnrealizedCastsPatterns(patterns);
      target.addIllegalOp<UnrealizedConversionCastOp>();
      if (failed(applyFullConversion(getOperation(), target, std::move(patterns))))
        return signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass> ksim::createCombToLLVMPass() {
  return std::make_unique<CombToLLVMPass>();
}
