#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "ksim/KSimDialect.h"
#include "ksim/KSimPasses.h"
#include "ksim/KSimOps.h"
#include "ksim/Utils/LLVMQueue.h"

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Support/LLVM.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <tuple>

#define GEN_PASS_DEF_LOWERTOLLVM
#include "PassDetails.h"

using namespace ksim;
using namespace circt;
using namespace mlir;
using namespace llvm;

struct QueueMapping {
  llvm::DenseMap<StringRef, std::unique_ptr<AbstractQueue>> qmap;
  bool disableQueueSelect;
  QueueMapping(bool disableQueueSelect=false): disableQueueSelect(disableQueueSelect) {}
  void createQueue(ksim::DefQueueOp op, OpBuilder & builder) {
    if(disableQueueSelect) {
      qmap[op.getSymName()] = ksim::createQueue(NaiveQueueType, op.getLoc(), op.getSymName(), op.getType(), op.getDepth(), op.isPublic(), builder);
    }
    else {
      qmap[op.getSymName()] = ksim::createQueue(op.getLoc(), op.getSymName(), op.getType(), op.getDepth(), op.isPublic(), op.getDelay(), builder);
    }
  }
  AbstractQueue * lookup(StringRef name) {
    return qmap[name].get();
  }
};

template<typename T>
struct QMapRewritePattern: OpRewritePattern<T> {
  QueueMapping & qmap;
  QMapRewritePattern(QueueMapping & qmap, MLIRContext * ctx)
    : OpRewritePattern<T>(ctx, 1), qmap(qmap) {}
};

struct DefQueueRewritePattern: QMapRewritePattern<ksim::DefQueueOp> {
  using QMapRewritePattern<ksim::DefQueueOp>::QMapRewritePattern;
  LogicalResult matchAndRewrite(ksim::DefQueueOp op, PatternRewriter & rewriter) const final {
    rewriter.setInsertionPoint(op);
    qmap.createQueue(op, rewriter);
    rewriter.eraseOp(op);
    return success();
  }
};

struct PushQueueRewritePattern: QMapRewritePattern<ksim::PushQueueOp> {
  using QMapRewritePattern<ksim::PushQueueOp>::QMapRewritePattern;
  LogicalResult matchAndRewrite(ksim::PushQueueOp op, PatternRewriter & rewriter) const final {
    rewriter.setInsertionPoint(op);
    auto queue = qmap.lookup(op.getQueue());
    queue->push(op->getLoc(), op.getInput(), op->getBlock(), rewriter);
    rewriter.eraseOp(op);
    return success();
  }
};

struct GetQueueRewritePattern: QMapRewritePattern<ksim::GetQueueOp> {
  using QMapRewritePattern<ksim::GetQueueOp>::QMapRewritePattern;
  LogicalResult matchAndRewrite(ksim::GetQueueOp op, PatternRewriter & rewriter) const final {
    rewriter.setInsertionPoint(op);
    auto queue = qmap.lookup(op.getQueue());
    auto value = queue->get(op->getLoc(), op.getIdx(), op->getBlock(), rewriter);
    rewriter.replaceOp(op, ValueRange(value));
    return success();
  }
};

using MemoryMapping = llvm::DenseMap<StringRef, LLVM::GlobalOp>;

template<typename T>
struct MMapRewritePattern: OpRewritePattern<T> {
  MemoryMapping & mmap;
  MMapRewritePattern(MemoryMapping & mmap, MLIRContext * ctx)
    : OpRewritePattern<T>(ctx, 1), mmap(mmap) {}
};

struct DefMemRewritePattern: MMapRewritePattern<ksim::DefMemOp> {
  using MMapRewritePattern<ksim::DefMemOp>::MMapRewritePattern;
  LogicalResult matchAndRewrite(ksim::DefMemOp op, PatternRewriter & rewriter) const final {
    rewriter.setInsertionPoint(op);
    auto name = op.getSymName();
    auto elemType = op.getType().getElementType();
    auto depth = op.getType().getDepth();
    auto queueType = LLVM::LLVMArrayType::get(elemType, depth);
    auto linkage = op.isPublic() ? mlir::LLVM::Linkage::LinkonceODR : mlir::LLVM::Linkage::Private;
    mmap[name] = rewriter.create<LLVM::GlobalOp>(op.getLoc(), queueType, false, linkage, name, nullptr, 0, 0, true);
    rewriter.eraseOp(op);
    return success();
  }
};

struct LowReadMemRewritePattern: MMapRewritePattern<ksim::LowReadMemOp> {
  using MMapRewritePattern<ksim::LowReadMemOp>::MMapRewritePattern;
  LogicalResult matchAndRewrite(ksim::LowReadMemOp op, PatternRewriter & rewriter) const final {
    auto glbOp = mmap.lookup(op.getMem());
    auto elemPtrType = LLVM::LLVMPointerType::get(op.getType());
    auto zero = rewriter.create<LLVM::ConstantOp>(op.getLoc(), op.getAddr().getType(), 0);
    auto addr = rewriter.create<LLVM::SelectOp>(op.getLoc(), op.getEn(), op.getAddr(), zero);
    auto glbAddr = rewriter.create<LLVM::AddressOfOp>(op.getLoc(), glbOp);
    auto baseAddr = rewriter.create<LLVM::GEPOp>(op.getLoc(), elemPtrType, glbAddr, ArrayRef(LLVM::GEPArg(0)));
    auto addrI32 = rewriter.create<LLVM::ZExtOp>(op.getLoc(), rewriter.getI32Type(), addr);
    auto elemAddr = rewriter.create<LLVM::GEPOp>(op.getLoc(), elemPtrType, baseAddr, addrI32.getResult());
    auto readData = rewriter.create<LLVM::LoadOp>(op.getLoc(), elemAddr).getResult();
    auto zeroData = rewriter.create<LLVM::ConstantOp>(op.getLoc(), op.getType(), 0);
    auto dataOrZero = rewriter.create<LLVM::SelectOp>(op.getLoc(), op.getEn(), readData, zeroData);
    rewriter.replaceOp(op, ValueRange(dataOrZero.getResult()));
    return success();
  }
};

struct LowWriteMemRewritePattern: MMapRewritePattern<ksim::LowWriteMemOp> {
  using MMapRewritePattern<ksim::LowWriteMemOp>::MMapRewritePattern;
  LogicalResult matchAndRewrite(ksim::LowWriteMemOp op, PatternRewriter & rewriter) const final {
    auto glbOp = mmap.lookup(op.getMem());
    auto elemType = op.getData().getType();
    auto elemPtrType = LLVM::LLVMPointerType::get(elemType);
    auto zero = rewriter.create<LLVM::ConstantOp>(op.getLoc(), op.getAddr().getType(), 0);
    auto addr = rewriter.create<LLVM::SelectOp>(op.getLoc(), op.getEn(), op.getAddr(), zero);
    auto glbAddr = rewriter.create<LLVM::AddressOfOp>(op.getLoc(), glbOp);
    auto baseAddr = rewriter.create<LLVM::GEPOp>(op.getLoc(), elemPtrType, glbAddr, ArrayRef(LLVM::GEPArg(0)));
    auto addrI32 = rewriter.create<LLVM::ZExtOp>(op.getLoc(), rewriter.getI32Type(), addr);
    auto elemAddr = rewriter.create<LLVM::GEPOp>(op.getLoc(), elemPtrType, baseAddr, addrI32.getResult());
    auto readData = rewriter.create<LLVM::LoadOp>(op.getLoc(), elemAddr).getResult();
    auto vecSize = hw::getBitWidth(elemType) / op.getMaskBits();
    auto vecElemType = rewriter.getIntegerType(op.getMaskBits());
    auto vecType = mlir::VectorType::get(vecSize, vecElemType);
    auto readDataVec = rewriter.create<LLVM::BitcastOp>(op.getLoc(), vecType, readData);
    auto writeDataVec = rewriter.create<LLVM::BitcastOp>(op.getLoc(), vecType, op.getData());
    auto mskElemType = rewriter.getIntegerType(1);
    auto mskType = mlir::VectorType::get(vecSize, mskElemType);
    auto mskVec = rewriter.create<LLVM::BitcastOp>(op.getLoc(), mskType, op.getMask());
    auto writeBackVec = rewriter.create<LLVM::UndefOp>(op.getLoc(), vecType).getResult();
    for(auto i = 0ul; i < vecSize; i++) {
      auto pos = rewriter.create<LLVM::ConstantOp>(op.getLoc(), rewriter.getI32Type(), i);
      auto mski = rewriter.create<LLVM::ExtractElementOp>(op.getLoc(), mskVec, pos).getResult();
      auto dati = rewriter.create<LLVM::ExtractElementOp>(op.getLoc(), writeDataVec, pos).getResult();
      auto readi = rewriter.create<LLVM::ExtractElementOp>(op.getLoc(), readDataVec, pos).getResult();
      auto wbi = rewriter.create<LLVM::SelectOp>(op.getLoc(), mski, dati, readi).getResult();
      writeBackVec = rewriter.create<LLVM::InsertElementOp>(op.getLoc(), writeBackVec, wbi, pos);
    }
    auto writeBack = rewriter.create<LLVM::BitcastOp>(op.getLoc(), elemType, writeBackVec);
    auto writeBackEn = rewriter.create<LLVM::SelectOp>(op.getLoc(), op.getEn(), writeBack, readData);
    rewriter.create<LLVM::StoreOp>(op.getLoc(), writeBackEn.getResult(), elemAddr);
    rewriter.eraseOp(op);
    return success();
  }
};

static uint32_t computeAlign(uint64_t bits) {
  return 1 << (std::min(std::max(Log2_64_Ceil(bits), 3u), 8u) - 3);
}

static uint32_t computeAlign(mlir::Type type) {
  return computeAlign(hw::getBitWidth(type));
}

static void sortGlobals(mlir::ModuleOp mod) {
  llvm::DenseMap<StringRef, unsigned> order;
  unsigned idx = 0;
  mod->walk([&](ksim::GetQueueOp pushQ) {
    order[pushQ.getQueue()] = idx++;
  });
  SmallVector<std::tuple<uint8_t, unsigned, ksim::DefQueueOp>> ops;
  for(auto op: mod.getOps<ksim::DefQueueOp>()) {
    ops.push_back(std::make_tuple(op.isPublic(), order[op.getSymName()], op));
  }
  llvm::sort(ops);
  for(auto [is_private, align, op]: ops) op->remove();
  OpBuilder builder(mod->getContext());
  builder.setInsertionPointToStart(mod.getBody());
  for(auto [is_private, align, op]: ops)
    builder.insert(op);
}

struct LowerToLLVMPass : ksim::impl::LowerToLLVMBase<LowerToLLVMPass> {
  using ksim::impl::LowerToLLVMBase<LowerToLLVMPass>::LowerToLLVMBase;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }
  void addCommonLegalDialect(ConversionTarget & target) {
    target.addLegalDialect<comb::CombDialect>();
    target.addLegalDialect<sv::SVDialect>();
    target.addLegalDialect<hw::HWDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addLegalOp<circt::ModuleOp>();
  }
  void runOnOperation() final {
    QueueMapping qmap;
    MemoryMapping mmap;
    auto & context = getContext();
    sortGlobals(getOperation());
    {
      ConversionTarget target(context);
      addCommonLegalDialect(target);
      target.addLegalDialect<ksim::KSimDialect>();
      target.addIllegalOp<ksim::DefQueueOp>();
      target.addIllegalOp<ksim::DefMemOp>();
      RewritePatternSet patterns(&context);
      patterns.add<DefQueueRewritePattern>(qmap, &context);
      patterns.add<DefMemRewritePattern>(mmap, &context);
      if (failed(applyFullConversion(getOperation(), target, std::move(patterns))))
        return signalPassFailure();
    }
    {
      ConversionTarget target(context);
      addCommonLegalDialect(target);
      target.addIllegalDialect<ksim::KSimDialect>();
      RewritePatternSet patterns(&context);
      patterns.add<PushQueueRewritePattern>(qmap, &context);
      patterns.add<GetQueueRewritePattern>(qmap, &context);
      patterns.add<LowReadMemRewritePattern>(mmap, &context);
      patterns.add<LowWriteMemRewritePattern>(mmap, &context);
      if (failed(applyFullConversion(getOperation(), target, std::move(patterns))))
        return signalPassFailure();
    }
    // sortGlobals(getOperation());
  }
};

std::unique_ptr<mlir::Pass> ksim::createLowerToLLVMPass(LowerToLLVMOptions options) {
  return std::make_unique<LowerToLLVMPass>(options);
}
