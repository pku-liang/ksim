#include "ksim/Utils/LLVMQueue.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MathExtras.h"
#include <memory>

using namespace ksim;
using namespace circt;
using namespace mlir;
using namespace llvm;

static uint32_t computeAlign(uint64_t bits) {
  return 1 << (std::min(std::max(Log2_64_Ceil(bits), 3u), 8u) - 3);
}

static uint32_t computeAlign(mlir::Type type) {
  return computeAlign(hw::getBitWidth(type));
}

NaiveQueue::NaiveQueue(mlir::Location loc, StringRef name, mlir::Type elemType, int64_t depth, bool isPublic, OpBuilder & builder)
  : AbstractQueue(NaiveQueueType, name, elemType, depth, isPublic) {
  auto queueType = LLVM::LLVMArrayType::get(elemType, depth);
  elemPtrType = LLVM::LLVMPointerType::get(elemType);
  llvmGlobal = builder.create<LLVM::GlobalOp>(loc, queueType, false, linkage, name, nullptr, computeAlign(elemType));
}

mlir::Value NaiveQueue::get(mlir::Location loc, int64_t k, Block * blk, OpBuilder & builder) {
  auto gblAddr = builder.create<LLVM::AddressOfOp>(loc, llvmGlobal);
  auto addr = builder.create<LLVM::GEPOp>(loc, elemPtrType, gblAddr, ArrayRef(LLVM::GEPArg(0)));
  auto ptr = builder.create<LLVM::GEPOp>(loc, elemPtrType, addr, ArrayRef(LLVM::GEPArg(k)));
  auto ret = builder.create<LLVM::LoadOp>(loc, ptr);
  return ret;
}

void NaiveQueue::push(mlir::Location loc, mlir::Value value, Block * blk, OpBuilder & builder) {
  auto gblAddr = builder.create<LLVM::AddressOfOp>(loc, llvmGlobal);
  auto addr = builder.create<LLVM::GEPOp>(loc, elemPtrType, gblAddr, ArrayRef(LLVM::GEPArg(0)));
  SmallVector<mlir::Value> addrs(depth);
  for(int i = 0; i < depth; i++) {
    addrs[i] = builder.create<LLVM::GEPOp>(loc, elemPtrType, addr, ArrayRef(LLVM::GEPArg(i)));
  }
  for(int i = depth - 1; i >= 1; i--) {
    auto prev = builder.create<LLVM::LoadOp>(loc, addrs[i - 1]);
    builder.create<LLVM::StoreOp>(loc, prev, addrs[i]);
  }
  builder.create<LLVM::StoreOp>(loc, value, addrs[0]);
}

ShiftQueue::ShiftQueue(mlir::Location loc, StringRef name, mlir::Type elemType, int64_t depth, bool isPublic, OpBuilder & builder):
  AbstractQueue(ShiftQueueType, name, elemType, depth, isPublic) {
  elemWidth = hw::getBitWidth(elemType);
  queueType = builder.getIntegerType(elemWidth * depth);
  llvmGlobal = builder.create<LLVM::GlobalOp>(loc, queueType, false, linkage, name, nullptr, computeAlign(elemWidth * depth));
}

mlir::Value ShiftQueue::get(mlir::Location loc, int64_t k, mlir::Block * blk, mlir::OpBuilder & builder) {
  auto addr = builder.create<LLVM::AddressOfOp>(loc, llvmGlobal);
  auto queueValue = builder.create<LLVM::LoadOp>(loc, addr);
  auto shrValue = builder.create<LLVM::ConstantOp>(loc, queueType, k * elemWidth);
  auto shr = builder.create<LLVM::LShrOp>(loc, queueType, queueValue, shrValue);
  auto trunc = builder.create<LLVM::TruncOp>(loc, elemType, shr);
  return trunc;
}

void ShiftQueue::push(mlir::Location loc, mlir::Value value, mlir::Block *blk, mlir::OpBuilder &builder) {
  auto addr = builder.create<LLVM::AddressOfOp>(loc, llvmGlobal);
  mlir::Value nextQueue = nullptr;
  if(depth == 1) {
    nextQueue = value;
  }
  else {
    auto queueValue = builder.create<LLVM::LoadOp>(loc, addr);
    auto shlValue = builder.create<LLVM::ConstantOp>(loc, queueType, elemWidth);
    auto queueShl = builder.create<LLVM::ShlOp>(loc, queueType, queueValue, shlValue);
    auto valuePadded = builder.create<LLVM::ZExtOp>(loc, queueType, value);
    nextQueue = builder.create<LLVM::OrOp>(loc, queueShl, valuePadded);
  }
  builder.create<LLVM::StoreOp>(loc, nextQueue, addr);
}

static uint64_t padDepth(uint64_t depth) {
  auto align = 1ll << std::min(Log2_64_Ceil(depth), 5u);
  return (depth + align - 1) / align * align;
}

VecQueue::VecQueue(mlir::Location loc, StringRef name, mlir::Type elemType, int64_t depth, bool isPublic, OpBuilder & builder):
  AbstractQueue(ShiftQueueType, name, elemType, padDepth(depth), isPublic) {
  depth = padDepth(depth);
  queueType = mlir::VectorType::get(ArrayRef(depth), elemType);
  llvmGlobal = builder.create<LLVM::GlobalOp>(loc, queueType, false, linkage, name, nullptr);
}

mlir::Value VecQueue::get(mlir::Location loc, int64_t k, mlir::Block * blk, mlir::OpBuilder & builder) {
  auto addr = builder.create<LLVM::AddressOfOp>(loc, llvmGlobal);
  auto queueValue = builder.create<LLVM::LoadOp>(loc, addr);
  auto pos = builder.create<LLVM::ConstantOp>(loc, builder.getI32Type(), k);
  return builder.create<LLVM::ExtractElementOp>(loc, elemType, queueValue, pos);
}

void VecQueue::push(mlir::Location loc, mlir::Value value, mlir::Block *blk, mlir::OpBuilder &builder) {
  SmallVector<int32_t, 8> shuffle(depth);
  for(int i = 1; i < depth; i++) shuffle[i] = i - 1;
  auto undef = builder.create<LLVM::UndefOp>(loc, queueType);
  auto addr = builder.create<LLVM::AddressOfOp>(loc, llvmGlobal);
  auto queueValue = builder.create<LLVM::LoadOp>(loc, addr);
  auto shuffled = builder.create<LLVM::ShuffleVectorOp>(loc, queueValue, undef, shuffle);
  auto pos = builder.create<LLVM::ConstantOp>(loc, builder.getI32Type(), 0);
  auto inserted = builder.create<LLVM::InsertElementOp>(loc, shuffled, value, pos);
  builder.create<LLVM::StoreOp>(loc, inserted, addr);
}

PtrQueue::PtrQueue(mlir::Location loc, StringRef name, mlir::Type elemType, int64_t depth, bool isPublic, OpBuilder & builder):
  AbstractQueue(ShiftQueueType, name, elemType, padDepth(depth), isPublic) {
  depth = padDepth(depth);
  auto queueType = builder.getType<LLVM::LLVMArrayType>(elemType, depth);
  ptrType = builder.getIntegerType(Log2_32_Ceil(depth) + 1); // plus 1 to make it signed value
  auto ptrInit = builder.getIntegerAttr(ptrType, 0);
  std::string ptrName = name.str() + "__ptr";
  bufferOp = builder.create<LLVM::GlobalOp>(loc, queueType, false, linkage, name, nullptr, computeAlign(elemType));
  ptrOp = builder.create<LLVM::GlobalOp>(loc, ptrType, false, linkage, ptrName, ptrInit);
  elemPtrType = LLVM::LLVMPointerType::get(elemType);
}

mlir::Value PtrQueue::get(mlir::Location loc, int64_t k, mlir::Block * blk, mlir::OpBuilder & builder) {
  assert(k < depth);
  auto gblAddr = builder.create<LLVM::AddressOfOp>(loc, bufferOp);
  auto bufAddr = builder.create<LLVM::GEPOp>(loc, elemPtrType, gblAddr, ArrayRef(LLVM::GEPArg(0)));
  auto ptrAddr = builder.create<LLVM::AddressOfOp>(loc, ptrOp);
  auto ptrValue = builder.create<LLVM::LoadOp>(loc, ptrAddr);
  if(k == 0) {
    auto frontAddr = builder.create<LLVM::GEPOp>(loc, elemPtrType, bufAddr, ptrValue.getResult());
    return builder.create<LLVM::LoadOp>(loc, frontAddr);
  }
  else {
    auto constantK = builder.create<LLVM::ConstantOp>(loc, ptrType, k); // L
    auto constantQK = builder.create<LLVM::ConstantOp>(loc, ptrType, depth - k); // D - K
    auto ptrAddK = builder.create<LLVM::AddOp>(loc, ptrValue, constantK); // P + K
    auto ptrSubQK = builder.create<LLVM::SubOp>(loc, ptrValue, constantQK); // P - (D - K)
    auto overflow = builder.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::uge, ptrValue, constantQK); // P >= D - K
    auto realPtrValue = builder.create<LLVM::SelectOp>(loc, overflow, ptrSubQK, ptrAddK); // P >= D - K ? P - (D - K) : P + K
    auto frontAddr = builder.create<LLVM::GEPOp>(loc, elemPtrType, bufAddr, realPtrValue.getResult());
    return builder.create<LLVM::LoadOp>(loc, frontAddr);
  }
}

void PtrQueue::push(mlir::Location loc, mlir::Value value, mlir::Block *blk, mlir::OpBuilder &builder) {
  auto gblAddr = builder.create<LLVM::AddressOfOp>(loc, bufferOp);
  auto bufAddr = builder.create<LLVM::GEPOp>(loc, elemPtrType, gblAddr, ArrayRef(LLVM::GEPArg(0)));
  auto ptrAddr = builder.create<LLVM::AddressOfOp>(loc, ptrOp);
  auto ptrValue = builder.create<LLVM::LoadOp>(loc, ptrAddr);
  auto const0 = builder.create<LLVM::ConstantOp>(loc, ptrType, 0);
  auto const1 = builder.create<LLVM::ConstantOp>(loc, ptrType, 1);
  auto constD_1 = builder.create<LLVM::ConstantOp>(loc, ptrType, depth - 1);
  auto ptrSub1 = builder.create<LLVM::SubOp>(loc, ptrValue, const1);
  auto ptrEq0 = builder.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, ptrValue, const0);
  auto nextPtrValue = builder.create<LLVM::SelectOp>(loc, ptrEq0, constD_1, ptrSub1);
  auto storeAddr = builder.create<LLVM::GEPOp>(loc, elemPtrType, bufAddr, nextPtrValue.getResult());
  builder.create<LLVM::StoreOp>(loc, value, storeAddr);
  builder.create<LLVM::StoreOp>(loc, nextPtrValue, ptrAddr);
}

OneSlotQueue::OneSlotQueue(mlir::Location loc, StringRef name, mlir::Type elemType, int64_t depth, bool isPublic, OpBuilder & builder):
  AbstractQueue(ShiftQueueType, name, elemType, depth, isPublic) {
  assert(depth == 1);
  llvmGlobal = builder.create<LLVM::GlobalOp>(loc, elemType, false, linkage, name, nullptr, computeAlign(elemType));
}

mlir::Value OneSlotQueue::get(mlir::Location loc, int64_t k, mlir::Block * blk, mlir::OpBuilder & builder) {
  auto addr = builder.create<LLVM::AddressOfOp>(loc, llvmGlobal);
  return builder.create<LLVM::LoadOp>(loc, addr);
}

void OneSlotQueue::push(mlir::Location loc, mlir::Value value, mlir::Block *blk, mlir::OpBuilder &builder) {
  auto addr = builder.create<LLVM::AddressOfOp>(loc, llvmGlobal);
  builder.create<LLVM::StoreOp>(loc, value, addr);
}

std::unique_ptr<AbstractQueue> ksim::createQueue(QueueType type, mlir::Location loc, mlir::StringRef name, mlir::Type elemType, int64_t depth, bool isPublic, mlir::OpBuilder &builder) {
  switch(type) {
    case NaiveQueueType: return std::make_unique<NaiveQueue>(loc, name, elemType, depth, isPublic, builder);
    case ShiftQueueType: return std::make_unique<ShiftQueue>(loc, name, elemType, depth, isPublic, builder);
    case VecQueueType: return std::make_unique<VecQueue>(loc, name, elemType, depth, isPublic, builder);
    case PtrQueueType: return std::make_unique<PtrQueue>(loc, name, elemType, depth, isPublic, builder);
    case OneSlotQueueType: return std::make_unique<OneSlotQueue>(loc, name, elemType, depth, isPublic, builder);
  }
  assert(false && "unknown queue type");
}

std::unique_ptr<AbstractQueue> ksim::createQueue(mlir::Location loc, mlir::StringRef name, mlir::Type elemType, int64_t depth, bool isPublic, ArrayRef<int64_t> delays, OpBuilder &builder) {
  auto width = hw::getBitWidth(elemType);
  if(depth == 1) return createQueue(OneSlotQueueType, loc, name, elemType, depth, isPublic, builder);
  if(depth <= 4) return createQueue(NaiveQueueType, loc, name, elemType, depth, isPublic, builder);
  if(width <= 4 && depth * width <= 128) return createQueue(ShiftQueueType, loc, name, elemType, depth, isPublic, builder);
  if(width * depth <= 128) return createQueue(VecQueueType, loc, name, elemType, depth, isPublic, builder);
  return createQueue(PtrQueueType, loc, name, elemType, depth, isPublic, builder);
  //   if(depth == 1)
  //   return createQueue(OneSlotQueueType, loc, name, elemType, depth, isPublic, builder);
  // if(depth <= 4)
  //   return createQueue(NaiveQueueType, loc, name, elemType, depth, isPublic, builder);
  // if((width == 8 || width == 16 || width == 32) && width * depth <= 128) {
  //   return createQueue(VecQueueType, loc, name, elemType, depth, isPublic, builder);
  // }
  // if(depth * width <= 128) {
  //   if(depth <= 16 || width <= 4)
  //     return createQueue(ShiftQueueType, loc, name, elemType, depth, isPublic, builder);
  // }
  // return createQueue(PtrQueueType, loc, name, elemType, depth, isPublic, builder);
}
