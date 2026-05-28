#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JitOptions {
    pub enabled: bool,
    pub mlir_execution: bool,
}

impl Default for JitOptions {
    fn default() -> Self {
        Self::mlir_execution()
    }
}

impl JitOptions {
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            mlir_execution: false,
        }
    }

    pub fn runtime() -> Self {
        Self {
            enabled: true,
            mlir_execution: false,
        }
    }

    pub fn mlir_execution() -> Self {
        Self {
            enabled: true,
            mlir_execution: true,
        }
    }

    pub fn from_env() -> Self {
        std::env::var("QUILL_JIT")
            .ok()
            .as_deref()
            .and_then(Self::parse)
            .unwrap_or_default()
    }

    pub fn mlir_execution_enabled(self) -> bool {
        self.enabled && self.mlir_execution
    }

    pub fn enabled(self) -> bool {
        self.enabled
    }

    fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "mlir" | "compiled" | "on" | "1" | "true" => Some(Self::mlir_execution()),
            "" => Some(Self::mlir_execution()),
            "runtime" => Some(Self::runtime()),
            "off" | "0" | "false" => Some(Self::disabled()),
            _ => None,
        }
    }
}
