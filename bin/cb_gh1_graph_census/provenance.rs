use serde_json::{Value, json};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

const CANONICAL_RUSTFLAGS: &str = "-C target-cpu=x86-64-v3";

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct FileSeal {
    pub(crate) path: String,
    pub(crate) bytes: u64,
    pub(crate) sha256: String,
}

impl FileSeal {
    pub(crate) fn json(&self) -> Value {
        json!({
            "path": self.path,
            "bytes": self.bytes,
            "sha256": self.sha256,
        })
    }
}

pub(crate) fn unix_millis() -> Result<u128, String> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .map_err(|error| format!("system clock is before UNIX epoch: {error}"))
}

pub(crate) fn manifest_dir() -> Result<PathBuf, String> {
    fs::canonicalize(env!("CARGO_MANIFEST_DIR")).map_err(|error| {
        format!(
            "failed to canonicalize CARGO_MANIFEST_DIR {}: {error}",
            env!("CARGO_MANIFEST_DIR")
        )
    })
}

fn git_safe_directory(manifest: &Path) -> String {
    let rendered = manifest.to_string_lossy().replace('\\', "/");
    if let Some(unc) = rendered.strip_prefix("//?/UNC/") {
        format!("//{unc}")
    } else if let Some(drive_path) = rendered.strip_prefix("//?/") {
        drive_path.to_string()
    } else {
        rendered
    }
}

fn invoke_git(manifest: &Path, args: &[&str]) -> Result<String, String> {
    let safe = format!("safe.directory={}", git_safe_directory(manifest));
    let output = Command::new("git")
        .arg("-c")
        .arg(safe)
        .args(args)
        .current_dir(manifest)
        .output()
        .map_err(|error| format!("failed to invoke git {args:?}: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "git {args:?} failed status={}: stdout={} stderr={}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    String::from_utf8(output.stdout)
        .map(|value| value.trim().to_string())
        .map_err(|error| format!("git {args:?} emitted non-UTF-8 output: {error}"))
}

pub(crate) fn seal_file(path: &Path) -> Result<FileSeal, String> {
    let payload = fs::read(path)
        .map_err(|error| format!("failed to read sealed file {}: {error}", path.display()))?;
    let canonical = path.canonicalize().map_err(|error| {
        format!(
            "failed to canonicalize sealed file {}: {error}",
            path.display()
        )
    })?;
    Ok(FileSeal {
        path: canonical.display().to_string(),
        bytes: u64::try_from(payload.len())
            .map_err(|_| format!("file length does not fit u64: {}", path.display()))?,
        sha256: sha256_hex(&payload),
    })
}

pub(crate) fn source_identity(
    expected_ancestor: &str,
    critical_sources: &[(&str, &[u8])],
) -> Result<Value, String> {
    let manifest = manifest_dir()?;
    let head = invoke_git(&manifest, &["rev-parse", "HEAD"])?;
    if head.len() != 40 || !head.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!("unexpected git HEAD {head:?}"));
    }
    invoke_git(
        &manifest,
        &["merge-base", "--is-ancestor", expected_ancestor, "HEAD"],
    )
    .map_err(|error| {
        format!("HEAD does not descend from preregistration {expected_ancestor}: {error}")
    })?;

    let worktree_status = invoke_git(
        &manifest,
        &["status", "--porcelain", "--untracked-files=all"],
    )?;
    if !worktree_status.is_empty() {
        return Err(format!(
            "worktree is dirty before/after census: {worktree_status}"
        ));
    }

    let mut files = Vec::with_capacity(critical_sources.len());
    for &(relative, compiled_bytes) in critical_sources {
        invoke_git(&manifest, &["ls-files", "--error-unmatch", "--", relative])
            .map_err(|error| format!("critical source is not tracked ({relative}): {error}"))?;
        let disk_path = manifest.join(relative);
        let disk_bytes = fs::read(&disk_path)
            .map_err(|error| format!("failed to read critical source {relative}: {error}"))?;
        if disk_bytes != compiled_bytes {
            return Err(format!(
                "running executable was not compiled from current critical source {relative}"
            ));
        }
        let seal = seal_file(&disk_path)?;
        files.push(json!({
            "relative_path": relative,
            "seal": seal.json(),
            "compiled_bytes_match": true,
            "compiled_sha256": sha256_hex(compiled_bytes),
        }));
    }

    Ok(json!({
        "git_head": head,
        "expected_preregister_ancestor": expected_ancestor,
        "worktree_status": worktree_status,
        "executable_critical_sources_match_worktree": true,
        "critical_sources": files,
    }))
}

pub(crate) fn executable_identity(expected_stem: &str) -> Result<Value, String> {
    if cfg!(debug_assertions) {
        return Err("authoritative census executable must be a release build".to_string());
    }
    let path = env::current_exe().map_err(|error| format!("current_exe failed: {error}"))?;
    let observed_stem = path
        .file_stem()
        .and_then(|value| value.to_str())
        .ok_or_else(|| "current executable has no UTF-8 stem".to_string())?;
    if !observed_stem.eq_ignore_ascii_case(expected_stem) {
        return Err(format!(
            "unexpected executable stem {observed_stem:?}, expected {expected_stem:?}"
        ));
    }
    let profile_dir = path
        .parent()
        .and_then(Path::file_name)
        .and_then(|value| value.to_str())
        .ok_or_else(|| "current executable has no UTF-8 profile directory".to_string())?;
    if !profile_dir.eq_ignore_ascii_case("release") {
        return Err(format!(
            "authoritative executable is under profile directory {profile_dir:?}, expected \"release\""
        ));
    }
    Ok(seal_file(&path)?.json())
}

pub(crate) fn environment_identity(canonical_build: &str) -> Result<Value, String> {
    #[cfg(target_arch = "x86_64")]
    if !cfg!(target_feature = "avx2") || !cfg!(target_feature = "bmi2") {
        return Err("compiled target lacks AVX2/BMI2 required by the x86-64-v3 build".to_string());
    }
    let rustflags = env::var("RUSTFLAGS")
        .map_err(|_| format!("runtime RUSTFLAGS must equal {CANONICAL_RUSTFLAGS:?}"))?;
    if rustflags != CANONICAL_RUSTFLAGS {
        return Err(format!(
            "runtime RUSTFLAGS mismatch: observed={rustflags:?} expected={CANONICAL_RUSTFLAGS:?}"
        ));
    }
    let mut noru_names = env::vars_os()
        .filter_map(|(name, _)| {
            let rendered = name.to_string_lossy().into_owned();
            rendered
                .to_ascii_uppercase()
                .starts_with("NORU_")
                .then_some(rendered)
        })
        .collect::<Vec<_>>();
    noru_names.sort();
    if !noru_names.is_empty() {
        return Err(format!(
            "NORU_* environment overrides are forbidden: {noru_names:?}"
        ));
    }
    Ok(json!({
        "runtime_RUSTFLAGS": rustflags,
        "compiled_target_features": {
            "avx2": cfg!(target_feature = "avx2"),
            "bmi2": cfg!(target_feature = "bmi2"),
            "fma": cfg!(target_feature = "fma"),
        },
        "canonical_build": canonical_build,
        "noru_prefixed_variables": noru_names,
    }))
}

pub(crate) fn toolchain_identity() -> Result<Value, String> {
    let output = Command::new("rustc")
        .arg("-Vv")
        .output()
        .map_err(|error| format!("failed to invoke rustc -Vv: {error}"))?;
    if !output.status.success() {
        return Err(format!("rustc -Vv failed with {}", output.status));
    }
    let stdout = String::from_utf8(output.stdout)
        .map_err(|error| format!("rustc -Vv emitted non-UTF-8 output: {error}"))?;
    Ok(json!({"rustc_vv": stdout}))
}

#[cfg(target_arch = "x86_64")]
pub(crate) fn cpu_identity() -> Result<Value, String> {
    use std::arch::x86_64::{__cpuid, __cpuid_count};

    let leaf0 = unsafe { __cpuid(0) };
    let mut vendor_bytes = Vec::with_capacity(12);
    vendor_bytes.extend_from_slice(&leaf0.ebx.to_le_bytes());
    vendor_bytes.extend_from_slice(&leaf0.edx.to_le_bytes());
    vendor_bytes.extend_from_slice(&leaf0.ecx.to_le_bytes());
    let vendor = String::from_utf8(vendor_bytes)
        .map_err(|error| format!("CPUID vendor is not UTF-8: {error}"))?;
    let leaf1 = unsafe { __cpuid(1) };
    let base_family = (leaf1.eax >> 8) & 0x0f;
    let ext_family = (leaf1.eax >> 20) & 0xff;
    let family = if base_family == 0x0f {
        base_family + ext_family
    } else {
        base_family
    };
    let base_model = (leaf1.eax >> 4) & 0x0f;
    let ext_model = (leaf1.eax >> 16) & 0x0f;
    let model = if base_family == 0x06 || base_family == 0x0f {
        (ext_model << 4) | base_model
    } else {
        base_model
    };
    let stepping = leaf1.eax & 0x0f;
    let leaf7 = unsafe { __cpuid_count(7, 0) };
    Ok(json!({
        "arch": "x86_64",
        "vendor": vendor,
        "family": family,
        "model": model,
        "stepping": stepping,
        "logical_processors": std::thread::available_parallelism()
            .map(|value| value.get())
            .map_err(|error| format!("available_parallelism failed: {error}"))?,
        "features": {
            "sse2": std::arch::is_x86_feature_detected!("sse2"),
            "avx2": std::arch::is_x86_feature_detected!("avx2"),
            "avx512f": std::arch::is_x86_feature_detected!("avx512f"),
            "bmi2": (leaf7.ebx & (1 << 8)) != 0,
        }
    }))
}

#[cfg(not(target_arch = "x86_64"))]
pub(crate) fn cpu_identity() -> Result<Value, String> {
    Ok(json!({
        "arch": env::consts::ARCH,
        "logical_processors": std::thread::available_parallelism()
            .map(|value| value.get())
            .map_err(|error| format!("available_parallelism failed: {error}"))?,
    }))
}

fn hex_bytes(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    for &byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

pub(crate) fn sha256_hex(input: &[u8]) -> String {
    hex_bytes(&crate::graph::sha256(input))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_matches_known_vector() {
        assert_eq!(
            sha256_hex(b"abc"),
            "BA7816BF8F01CFEA414140DE5DAE2223B00361A396177A9CB410FF61F20015AD"
        );
    }

    #[test]
    fn windows_verbatim_safe_directory_is_normalized() {
        assert_eq!(
            git_safe_directory(Path::new(r"\\?\C:\repo")),
            "C:/repo".to_string()
        );
    }
}
