#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Plugins/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

static std::string GetTypeName(const Function *F) {
  const std::string Name = F->getName().str();
  const std::size_t SuffixA = Name.rfind("clE");
  const std::size_t SuffixB = Name.rfind("clI");
  if (SuffixA != std::string::npos && SuffixB != std::string::npos) {
    const std::size_t Suffix = std::max(SuffixA, SuffixB);
    return Name.substr(0, Suffix) + "E";
  }
  if (SuffixA != std::string::npos)
    return Name.substr(0, SuffixA) + "E";
  if (SuffixB != std::string::npos)
    return Name.substr(0, SuffixB) + "E";
  return Name;
}

static std::vector<std::uint64_t> GetAttrArgs(const GlobalVariable *ArgsGV) {
  std::vector<std::uint64_t> Result;
  // No args.
  if (!ArgsGV)
    return Result;
  // Constant struct of args.
  if (const ConstantStruct *CS =
          dyn_cast<ConstantStruct>(ArgsGV->getInitializer())) {
    for (unsigned I = 0; I < CS->getNumOperands(); I++)
      if (const ConstantInt *Arg = dyn_cast<ConstantInt>(CS->getOperand(I))) {
        const std::uint64_t Value = *Arg->getValue().getRawData();
        Result.push_back(Value);
      }
  }
  // Zero-initialized struct.
  else if (const Constant *C = dyn_cast<Constant>(ArgsGV->getInitializer())) {
    const auto NumArgs = C->getType()->getStructNumElements();
    for (unsigned I = 0; I < NumArgs; I++)
      Result.push_back(0);
  }
  return Result;
}

class SyclPass : public PassInfoMixin<SyclPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM) {

    struct Attribute {
      std::string Name;
      std::vector<std::uint64_t> Args;
      std::string Function;
      std::string TranslationUnit;
    };

    struct Kernel {
      std::string TypeName;
      std::string Entrypoint;
    };

    std::vector<Attribute> Attributes;
    std::vector<Kernel> Kernels;

    // Iterate through the annotations in @llvm.global.annotations.
    if (const GlobalVariable *GV =
            M.getGlobalVariable("llvm.global.annotations")) {
      // [{ ptr, ptr, ptr, i32, ptr }]
      if (const ConstantArray *CA =
              dyn_cast<ConstantArray>(GV->getInitializer())) {
        for (unsigned i = 0; i < CA->getNumOperands(); ++i) {
          if (const ConstantStruct *CS =
                  dyn_cast<ConstantStruct>(CA->getOperand(i))) {
            if (CS->getNumOperands() != 5) {
              errs() << "Warning: Unexpected number of operands in annotation "
                        "struct: "
                     << CS->getNumOperands() << "\n";
              continue;
            }

            bool FoundKernel = false;
            std::string Annotation;
            std::string TypeName;
            std::string FunctionName;
            std::string TranslationUnit;

            // The 1st operand is the kernel/device function.
            if (const Function *F = dyn_cast<Function>(
                    CS->getOperand(0)->stripPointerCasts())) {
              FunctionName = F->getName().str();
              TypeName = GetTypeName(F);
            } else {
              continue;
            }

            // The 2nd operand is the annotation string ptr.
            if (const GlobalVariable *AnnoGV =
                    dyn_cast<GlobalVariable>(CS->getOperand(1))) {
              if (const ConstantDataArray *CDA =
                      dyn_cast<ConstantDataArray>(AnnoGV->getInitializer())) {
                if (CDA->isString()) {
                  Annotation = CDA->getAsString().str();
                  if (Annotation.starts_with("sycl_kernel_entrypoint"))
                    FoundKernel = true;
                }
              }
            } else {
              continue;
            }

            // The 3rd variable is a pointer to the TU name string.
            if (const GlobalVariable *TUGV =
                    dyn_cast<GlobalVariable>(CS->getOperand(2))) {
              if (const ConstantDataArray *CDA =
                      dyn_cast<ConstantDataArray>(TUGV->getInitializer())) {
                if (CDA->isString()) {
                  TranslationUnit = CDA->getAsString().str();
                }
              }
            } else {
              continue;
            }

            // The 5th operand is a struct of integer args.
            const std::vector<std::uint64_t> Args =
                GetAttrArgs(dyn_cast<GlobalVariable>(CS->getOperand(4)));

            if (FoundKernel) {
              Kernels.push_back(Kernel{TypeName, FunctionName});
            } else {
              Attributes.push_back(
                  Attribute{Annotation, Args, FunctionName, TranslationUnit});
            }
          }
        }
      }
    }

    LLVMContext &Context = M.getContext();
    IRBuilder<> Builder(Context);

    // Declare a function that takes two string arguments and will be linked in.
    std::vector<Type *> ParamTypes = {Builder.getPtrTy(), Builder.getPtrTy(),
                                      Builder.getInt32Ty()};
    FunctionType *RegFuncType =
        FunctionType::get(Builder.getVoidTy(), ParamTypes, true);
    FunctionCallee RegFuncCallee =
        M.getOrInsertFunction("sycl_register_attribute", RegFuncType);

    // Define a registration function to register the attributes.
    FunctionType *FuncType = FunctionType::get(Type::getVoidTy(Context), false);
    Function *RegFunc = Function::Create(FuncType, Function::InternalLinkage,
                                         "sycl_global_ctor", &M);
    BasicBlock *EntryBB = BasicBlock::Create(Context, "entry", RegFunc);
    Builder.SetInsertPoint(EntryBB);

    // Create a string for each attribute and call the registration function.
    for (const Attribute &Attr : Attributes) {
      // Find the kernel typename.
      const auto K =
          std::find_if(Kernels.begin(), Kernels.end(), [&](const Kernel &K) {
            return K.Entrypoint == Attr.Function;
          });
      if (K == Kernels.end()) {
        errs() << "Warning: No kernel found for attribute function "
               << Attr.Function << "\n";
        continue;
      }
      // Create the arguments for the registration function.
      SmallVector<Value *, 16> ArgValues;
      ArgValues.push_back(Builder.CreateGlobalString(K->TypeName));
      ArgValues.push_back(Builder.CreateGlobalString(Attr.Name));
      ArgValues.push_back(Builder.getInt32(Attr.Args.size()));
      for (const std::uint64_t Arg : Attr.Args)
        ArgValues.push_back(Builder.getInt64(Arg));
      // Call the registration function.
      Builder.CreateCall(RegFuncCallee, ArgValues);
    }

    Builder.CreateRetVoid();

    // Get the type for a global ctor, which is a struct of {i32, ptr, ptr}.
    StructType *CtorStructTy =
        StructType::get(Context, {Builder.getInt32Ty(), Builder.getPtrTy(),
                                  Builder.getPtrTy()});

    // Create a ConstantStruct for our new ctor, with priority 65535 and a
    // pointer to the registration function.
    Constant *CtorStruct = ConstantStruct::get(
        CtorStructTy, {Builder.getInt32(65535), RegFunc,
                       Constant::getNullValue(Builder.getPtrTy())});

    // Find or create the llvm.global_ctors global variable.
    GlobalVariable *GlobalCtors = M.getNamedGlobal("llvm.global_ctors");
    if (!GlobalCtors) {
      // The variable doesn't exist, so create it.
      ArrayType *CtorArrayTy = ArrayType::get(CtorStructTy, 1);
      GlobalCtors = new GlobalVariable(
          M, CtorArrayTy, false, GlobalVariable::AppendingLinkage,
          ConstantArray::get(CtorArrayTy, {CtorStruct}), "llvm.global_ctors");
    } else {
      // The variable exists, so we need to append our new ctor.
      ConstantArray *OldCtors =
          dyn_cast<ConstantArray>(GlobalCtors->getInitializer());
      if (!OldCtors) {
        errs() << "Warning: llvm.global_ctors is not a ConstantArray!\n";
        return PreservedAnalyses::all();
      }

      // Create a vector to hold the new list of constructors.
      std::vector<Constant *> NewCtors;
      for (unsigned i = 0; i < OldCtors->getNumOperands(); ++i)
        NewCtors.push_back(OldCtors->getOperand(i));
      NewCtors.push_back(CtorStruct);

      // Create a new array and set it as the initializer.
      M.removeGlobalVariable(GlobalCtors);
      ArrayType *CtorArrayTy = ArrayType::get(CtorStructTy, NewCtors.size());
      GlobalCtors = new GlobalVariable(
          M, CtorArrayTy, false, GlobalVariable::AppendingLinkage,
          ConstantArray::get(CtorArrayTy, NewCtors), "llvm.global_ctors");
    }

    // Since we are adding a new element and not modifying existing ones,
    // we can preserve all analyses. This is generally a good practice for
    // passes that don't alter the program's control flow or data dependencies.
    return PreservedAnalyses::all();
  }
};

} // end anonymous namespace

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "SyclPass", "v0.1", [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &MPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "sycl") {
                    MPM.addPass(SyclPass());
                    return true;
                  }
                  return false;
                });

            PB.registerOptimizerLastEPCallback(
                [&](ModulePassManager &MPM, OptimizationLevel Level,
                    ThinOrFullLTOPhase) { MPM.addPass(SyclPass()); });
          }};
}
