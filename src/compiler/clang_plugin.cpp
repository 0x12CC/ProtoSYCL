#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Attrs.inc"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Mangle.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Type.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Sema/ParsedAttr.h"
#include "clang/Sema/Sema.h"
#include "clang/Tooling/Tooling.h"

namespace {

using namespace clang;

class KernelVisitor : public RecursiveASTVisitor<KernelVisitor> {
public:
  explicit KernelVisitor(ASTContext &Context) : Context(Context) {}

  bool VisitVarDecl(VarDecl *VD) {
    if (!VD->hasAttr<AnnotateAttr>())
      return true;

    AnnotateAttr *AA = VD->getAttr<AnnotateAttr>();
    if (!AA || AA->getAnnotation() != "sycl_kernel_registration")
      return true;

    VD->dropAttr<AnnotateAttr>();

    VarTemplateSpecializationDecl *VTSD =
        dyn_cast<VarTemplateSpecializationDecl>(VD);
    if (!VTSD)
      return true;

    auto RD = VTSD->getTemplateArgs()[1].getAsType()->getAsCXXRecordDecl();
    for (auto *D : RD->decls()) {
      // Non-template call operator
      if (auto *MD = dyn_cast<CXXMethodDecl>(D)) {
        if (!MD->isOverloadedOperator() ||
            MD->getOverloadedOperator() != OO_Call)
          continue;
        MD->addAttr(AnnotateAttr::Create(Context, "sycl_kernel_entrypoint",
                                         nullptr, 0));
        continue;
      }

      // Template call operator
      if (auto *FTD = dyn_cast<FunctionTemplateDecl>(D)) {
        auto *MD = dyn_cast<CXXMethodDecl>(FTD->getTemplatedDecl());
        if (!MD || !MD->isOverloadedOperator() ||
            MD->getOverloadedOperator() != OO_Call)
          continue;
        for (auto *Spec : FTD->specializations()) {
          Spec->addAttr(AnnotateAttr::Create(Context, "sycl_kernel_entrypoint",
                                             nullptr, 0));
        }
        continue;
      }
    }
    return true;
  }

private:
  ASTContext &Context;
};

class KernelConsumer : public ASTConsumer {
private:
  KernelVisitor Visitor;

public:
  explicit KernelConsumer(ASTContext &Context) : Visitor(Context) {}

  void HandleTranslationUnit(ASTContext &Context) override {
    Visitor.TraverseDecl(Context.getTranslationUnitDecl());
  }
};

class KernelAction : public PluginASTAction {
protected:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef InFile) override {
    return std::make_unique<KernelConsumer>(CI.getASTContext());
  }

  bool ParseArgs(const CompilerInstance &CI,
                 const std::vector<std::string> &Args) override {
    return true;
  }

  ActionType getActionType() override { return AddBeforeMainAction; }
};

} // namespace

static FrontendPluginRegistry::Add<KernelAction>
    X("protosycl-kernel-visitor",
      "Visits all SYCL kernels and process the attributes.");
