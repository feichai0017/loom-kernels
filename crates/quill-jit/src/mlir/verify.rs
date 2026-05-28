use crate::JitError;
use crate::JitResult;

use super::MlirModule;

use melior::{
    dialect::DialectRegistry,
    ir::{operation::OperationLike, Module},
    utility::{register_all_dialects, register_all_llvm_translations},
    Context,
};

pub(super) fn verify_module(module: &MlirModule) -> JitResult<()> {
    verify_mlir_text(&module.text)
}

fn verify_mlir_text(text: &str) -> JitResult<()> {
    let context = mlir_context();
    let module = Module::parse(&context, text)
        .ok_or_else(|| JitError::Backend("MLIR parser rejected generated module".to_string()))?;
    if module.as_operation().verify() {
        Ok(())
    } else {
        Err(JitError::Backend(
            "MLIR verifier rejected generated module".to_string(),
        ))
    }
}

pub(super) fn mlir_context() -> Context {
    let context = Context::new();
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    context.append_dialect_registry(&registry);
    quill_mlir::register_dialect(&context);
    quill_mlir::register_passes();
    context.load_all_available_dialects();
    register_all_llvm_translations(&context);
    context
}
