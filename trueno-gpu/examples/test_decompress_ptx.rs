use trueno_gpu::kernels::lz4::Lz4WarpDecompressKernel;
use trueno_gpu::kernels::Kernel;

fn main() {
    let kernel = Lz4WarpDecompressKernel::new(1000);
    let ptx = kernel.emit_ptx();

    // Print first 200 lines
    for (i, line) in ptx.lines().enumerate() {
        if i < 200 {
            println!("{:>4}: {}", i + 1, line);
        }
    }

    println!("...");
    println!("Total lines: {}", ptx.lines().count());

    // Check for common PTX issues
    if !ptx.contains(".version") {
        println!("ERROR: Missing .version");
    }
    if !ptx.contains(".target") {
        println!("ERROR: Missing .target");
    }
    if !ptx.contains(".entry") {
        println!("ERROR: Missing .entry");
    }
    if !ptx.contains("lz4_warp_decompress") {
        println!("ERROR: Missing lz4_warp_decompress entry");
    }
}
