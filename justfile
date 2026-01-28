set shell := ["bash", "-lc"]

check:
    cargo fmt
    cargo clippy --all-targets --all-features -- -D warnings
    cargo test --all-features

check_core:
    cargo fmt
    cargo clippy --all-targets --no-default-features -- -D warnings
    cargo test --no-default-features
