use std::sync::LazyLock;

pub(crate) mod bounds;
pub mod builder;
pub mod types;
pub(crate) mod validation;

pub use builder::*;
pub use types::*;
pub use validation::{
    bind_socket_addr, parse_bind_host, validate_mesh_listener_port, validate_mesh_server_name,
    validate_worker_url,
};

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("Validation failed: {reason}")]
    ValidationFailed { reason: String },

    #[error("Invalid value for field '{field}': {value} - {reason}")]
    InvalidValue {
        field: String,
        value: String,
        reason: String,
    },

    #[error("Incompatible configuration: {reason}")]
    IncompatibleConfig { reason: String },

    #[error("Missing required field: {field}")]
    MissingRequired { field: String },
}

pub type ConfigResult<T> = Result<T, ConfigError>;

/// Process-wide opt-in, read once so request-body forwarding and the
/// Messages handlers always agree on whether normalization is required.
pub(crate) fn strip_anthropic_preamble_enabled() -> bool {
    static ENABLED: LazyLock<bool> = LazyLock::new(|| {
        std::env::var("SMG_STRIP_ANTHROPIC_PREAMBLE").is_ok_and(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "on" | "yes"
            )
        })
    });
    *ENABLED
}
