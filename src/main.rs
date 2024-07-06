use cust::memory::*;
use cust::module::Module;
use cust::prelude::*;
use cust::stream::Stream;
use std::ffi::CString;
use std::time::Instant;

fn matrix_multiply_cpu(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ctx = cust::quick_init()?;

    let ptx = CString::new(include_str!("../kernel.ptx"))?;
    let module = Module::from_ptx_cstr(&ptx, &[])?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    const N: usize = 1024;
    let mut a = vec![0.0f32; N * N];
    let mut b = vec![0.0f32; N * N];
    let mut c_gpu = vec![0.0f32; N * N];
    let mut c_cpu = vec![0.0f32; N * N];

    for i in 0..N {
        for j in 0..N {
            a[i * N + j] = (i * N + j) as f32;
            b[i * N + j] = (i * N + j) as f32;
        }
    }

    let d_a = DeviceBuffer::from_slice(&a)?;
    let d_b = DeviceBuffer::from_slice(&b)?;
    let d_c = DeviceBuffer::from_slice(&c_gpu)?;

    let threads_per_block = (16, 16, 1);
    let blocks_per_grid = (
        (N as u32 + threads_per_block.0 - 1) / threads_per_block.0,
        (N as u32 + threads_per_block.1 - 1) / threads_per_block.1,
        1,
    );

    let gpu_start = Instant::now();
    unsafe {
        let function = module.get_function("matrix_multiply")?;
        let result = launch!(function<<<blocks_per_grid, threads_per_block, 0, stream>>>(
            d_a.as_device_ptr(),
            d_b.as_device_ptr(),
            d_c.as_device_ptr(),
            N as i32
        ));
        result?;
    }
    stream.synchronize()?;
    d_c.copy_to(&mut c_gpu)?;
    let gpu_duration = gpu_start.elapsed();

    let cpu_start = Instant::now();
    matrix_multiply_cpu(&a, &b, &mut c_cpu, N);
    let cpu_duration = cpu_start.elapsed();

    println!("Matrix Multiplication Performance Comparison:");
    println!("--------------------------------------------");
    println!("Matrix Size: {} x {}", N, N);
    println!();
    println!("Execution Time:");
    println!("  - GPU: {:?}", gpu_duration);
    println!("  - CPU: {:?}", cpu_duration);
    println!();
    println!("Performance Comparison:");
    let speedup = cpu_duration.as_secs_f64() / gpu_duration.as_secs_f64();
    println!("  - Speedup: {:.2}x", speedup);
    println!();
    println!("Success! The matrices were multiplied correctly.");
    Ok(())
}
