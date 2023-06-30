#pragma once

#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Builders.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace ksim {

enum QueueType {NaiveQueueType, ShiftQueueType, VecQueueType, PtrQueueType, OneSlotQueueType};

struct AbstractQueue {
  const QueueType type;
  mlir::StringRef name;
  mlir::Type elemType;
  int64_t depth;
  mlir::LLVM::Linkage linkage;
  AbstractQueue(QueueType type, mlir::StringRef name, mlir::Type elemType, int64_t depth, bool isPublic)
    : type(type), name(name), elemType(elemType), depth(depth), linkage(isPublic ? mlir::LLVM::Linkage::LinkonceODR : mlir::LLVM::Linkage::Internal) {}
  inline QueueType getQueueType() const {return type;}
  virtual ~AbstractQueue() = default;
  virtual mlir::Value get(mlir::Location loc, int64_t k, mlir::Block * blk, mlir::OpBuilder & builder) = 0;
  virtual void push(mlir::Location loc, mlir::Value value, mlir::Block * blk, mlir::OpBuilder & builder) = 0;
};

struct NaiveQueue: public AbstractQueue {
  mlir::LLVM::GlobalOp llvmGlobal = nullptr;
  mlir::Type elemPtrType = nullptr;
  NaiveQueue(mlir::Location loc, mlir::StringRef name, mlir::Type elemType, int64_t depth, bool isPublic, mlir::OpBuilder & builder);
  static bool classof(const AbstractQueue *S) {return S->getQueueType() == NaiveQueueType;}
  mlir::Value get(mlir::Location loc, int64_t k, mlir::Block * blk, mlir::OpBuilder & builder);
  void push(mlir::Location loc, mlir::Value value, mlir::Block * blk, mlir::OpBuilder & builder);
};

struct ShiftQueue: public AbstractQueue {
  int64_t elemWidth = 0;
  mlir::Type queueType = nullptr;
  mlir::LLVM::GlobalOp llvmGlobal = nullptr;
  ShiftQueue(mlir::Location loc, mlir::StringRef name, mlir::Type elemType, int64_t depth, bool isPublic, mlir::OpBuilder &builder);
  static bool classof(const AbstractQueue *S) {return S->getQueueType() == ShiftQueueType;}
  mlir::Value get(mlir::Location loc, int64_t k, mlir::Block * blk, mlir::OpBuilder & builder);
  void push(mlir::Location loc, mlir::Value value, mlir::Block * blk, mlir::OpBuilder & builder);
};

struct VecQueue: public AbstractQueue {
  mlir::Type queueType = nullptr;
  mlir::LLVM::GlobalOp llvmGlobal = nullptr;
  VecQueue(mlir::Location loc, mlir::StringRef name, mlir::Type elemType, int64_t depth, bool isPublic, mlir::OpBuilder & builder);
  static bool classof(const AbstractQueue *S) {return S->getQueueType() == VecQueueType;}
  mlir::Value get(mlir::Location loc, int64_t k, mlir::Block * blk, mlir::OpBuilder & builder);
  void push(mlir::Location loc, mlir::Value value, mlir::Block * blk, mlir::OpBuilder & builder);
};

struct PtrQueue: public AbstractQueue {
  mlir::LLVM::GlobalOp bufferOp = nullptr, ptrOp = nullptr;
  mlir::Type ptrType = nullptr;
  mlir::Type elemPtrType = nullptr;
  mlir::LLVM::GlobalOp llvmGlobal = nullptr;
  PtrQueue(mlir::Location loc, mlir::StringRef name, mlir::Type elemType, int64_t depth, bool isPublic, mlir::OpBuilder & builder);
  static bool classof(const AbstractQueue *S) {return S->getQueueType() == PtrQueueType;}
  mlir::Value get(mlir::Location loc, int64_t k, mlir::Block * blk, mlir::OpBuilder & builder);
  void push(mlir::Location loc, mlir::Value value, mlir::Block * blk, mlir::OpBuilder & builder);
};

struct OneSlotQueue: public AbstractQueue {
  mlir::LLVM::GlobalOp llvmGlobal = nullptr;
  OneSlotQueue(mlir::Location loc, mlir::StringRef name, mlir::Type elemType, int64_t depth, bool isPublic, mlir::OpBuilder & builder);
  static bool classof(const AbstractQueue *S) {return S->getQueueType() == OneSlotQueueType;}
  mlir::Value get(mlir::Location loc, int64_t k, mlir::Block * blk, mlir::OpBuilder & builder);
  void push(mlir::Location loc, mlir::Value value, mlir::Block * blk, mlir::OpBuilder & builder);
};

std::unique_ptr<AbstractQueue> createQueue(QueueType type, mlir::Location loc, mlir::StringRef name, mlir::Type elemType, int64_t depth, bool isPublic, mlir::OpBuilder &builder);

std::unique_ptr<AbstractQueue> createQueue(mlir::Location loc, mlir::StringRef name, mlir::Type elemType, int64_t depth, bool isPublic, mlir::ArrayRef<int64_t> delays, mlir::OpBuilder &builder);

}