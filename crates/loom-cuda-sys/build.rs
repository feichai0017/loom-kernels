use std::env;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=LOOM_CUDA_ARCHS");

    let manifest_dir =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is always set"));
    let cuda_dir = manifest_dir.join("../../cuda");
    let header = cuda_dir.join("include/loom_cuda.h");
    let source = cuda_dir.join("src/attention_kernels.cu");
    println!("cargo:rerun-if-changed={}", header.display());
    println!("cargo:rerun-if-changed={}", source.display());

    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }

    let cuda_home = cuda_home();
    let nvcc = cuda_home.join("bin/nvcc");
    if !nvcc.is_file() {
        panic!(
            "CUDA feature enabled but nvcc was not found at {}; set CUDA_HOME",
            nvcc.display()
        );
    }

    let archs = env::var("LOOM_CUDA_ARCHS").unwrap_or_else(|_| "80,89,90".to_owned());
    let mut build = cc::Build::new();
    build
        // cc-rs' native compiler defaults (`-ffunction-sections`, `-G`, ...)
        // are not nvcc flags. Keep the CUDA invocation explicit and portable.
        .no_default_flags(true)
        .warnings(false)
        .cuda(true)
        .compiler(&nvcc)
        .include(cuda_dir.join("include"))
        .file(&source)
        .flag("-O3")
        .flag("-Xcompiler=-fPIC")
        .flag("-std=c++17")
        .flag("--expt-relaxed-constexpr")
        .flag("-lineinfo");
    for arch in archs
        .split(',')
        .map(str::trim)
        .filter(|arch| !arch.is_empty())
    {
        if !arch.chars().all(|character| character.is_ascii_digit()) {
            panic!("invalid CUDA architecture {arch:?} in LOOM_CUDA_ARCHS");
        }
        build.flag("-gencode");
        build.flag(format!("arch=compute_{arch},code=sm_{arch}"));
    }
    build.compile("loom_cuda_kernels");

    let library_dir = cuda_library_dir(&cuda_home);
    println!("cargo:rustc-link-search=native={}", library_dir.display());
    println!("cargo:rustc-link-lib=dylib=cudart");
}

fn cuda_home() -> PathBuf {
    env::var_os("CUDA_HOME")
        .or_else(|| env::var_os("CUDA_PATH"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/usr/local/cuda"))
}

fn cuda_library_dir(cuda_home: &Path) -> PathBuf {
    for candidate in [cuda_home.join("lib64"), cuda_home.join("lib")] {
        if candidate.is_dir() {
            return candidate;
        }
    }
    panic!(
        "CUDA runtime library directory was not found below {}",
        cuda_home.display()
    );
}
