//! trueno-gemm-codegen: Compile-time GEMM microkernel code generation.
//!
//! Contract: cgp-gemm-codegen-v1.yaml (C-CODEGEN-001 through C-CODEGEN-004)
//!
//! Generates shape-specialized AVX-512 microkernels at compile time via proc macros.
//! Sovereign implementation — no external BLAS dependencies.
//!
//! # Usage
//! ```ignore
//! use trueno_gemm_codegen::avx512_microkernel;
//!
//! avx512_microkernel!(mr = 8, nr = 32);
//! // Generates: pub unsafe fn microkernel_8x32_avx512_gen(k, a, b, c, ldc)
//! ```

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::parse::{Parse, ParseStream};
use syn::{LitInt, Token};

/// Parameters for microkernel generation.
struct MicrokernelParams {
    mr: usize,
    nr: usize,
}

impl Parse for MicrokernelParams {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        // Parse: mr = N, nr = M
        let _mr_ident: syn::Ident = input.parse()?;
        let _eq: Token![=] = input.parse()?;
        let mr_lit: LitInt = input.parse()?;
        let _comma: Token![,] = input.parse()?;
        let _nr_ident: syn::Ident = input.parse()?;
        let _eq2: Token![=] = input.parse()?;
        let nr_lit: LitInt = input.parse()?;

        Ok(MicrokernelParams { mr: mr_lit.base10_parse()?, nr: nr_lit.base10_parse()? })
    }
}

/// Generate an AVX-512 row-major C microkernel.
///
/// Layout: A is MR×K packed column-major, B is K×NR packed row-major,
/// C is MR×NR row-major with stride `ldc`.
///
/// Strategy for row-major C (MR rows, NR columns):
/// - Each C row spans ceil(NR/16) zmm registers
/// - Total accumulators = MR * ceil(NR/16)
/// - Per K step: load ceil(NR/16) B zmm, broadcast MR A scalars, MR*ceil(NR/16) FMAs
///
/// Register budget check (C-CODEGEN-004):
///   accumulators + B loads + headroom <= 32 zmm
#[proc_macro]
pub fn avx512_microkernel(input: TokenStream) -> TokenStream {
    let params = syn::parse_macro_input!(input as MicrokernelParams);
    let mr = params.mr;
    let nr = params.nr;

    let b_regs = nr.div_ceil(16); // ceil(NR/16) zmm registers for B
    let acc_count = mr * b_regs; // Total accumulator registers
    let total_regs = acc_count + b_regs + 4; // +4 headroom for A broadcasts

    if total_regs > 32 {
        return syn::Error::new(
            proc_macro2::Span::call_site(),
            format!(
                "C-CODEGEN-004: {mr}x{nr} needs {total_regs} zmm registers (max 32). \
                 Reduce MR or NR. accumulators={acc_count}, B_loads={b_regs}"
            ),
        )
        .to_compile_error()
        .into();
    }

    let fn_name = format_ident!("microkernel_{}x{}_avx512_gen", mr, nr);

    // Generate accumulator identifiers: c{row}_{half}
    let mut acc_idents = Vec::new();
    for row in 0..mr {
        for h in 0..b_regs {
            acc_idents.push(format_ident!("c{}_{}", row, h));
        }
    }

    // Generate C load statements
    let c_loads = generate_c_loads(mr, b_regs, &acc_idents);

    // Generate the inner K-loop body
    let k_body = generate_k_body(mr, nr, b_regs, &acc_idents);

    // Generate C store statements
    let c_stores = generate_c_stores(mr, b_regs, &acc_idents);

    let doc = format!(
        "Generated {mr}x{nr} AVX-512 microkernel ({acc_count} zmm accumulators, \
         {b_regs} B loads, {} FMAs/K-step). Contract: cgp-gemm-codegen-v1.yaml.",
        mr * b_regs
    );

    let output = quote! {
        #[doc = #doc]
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx512f", enable = "fma")]
        pub unsafe fn #fn_name(
            k: usize,
            a: *const f32,
            b: *const f32,
            c: *mut f32,
            ldc: usize,
        ) {
            use std::arch::x86_64::*;

            // Load C accumulators
            #(#c_loads)*

            // Main K loop
            for p in 0..k {
                #(#k_body)*
            }

            // Store C accumulators
            #(#c_stores)*
        }
    };

    output.into()
}

/// Generate C load statements for all accumulators.
fn generate_c_loads(
    mr: usize,
    b_regs: usize,
    acc_idents: &[proc_macro2::Ident],
) -> Vec<TokenStream2> {
    let mut loads = Vec::new();
    for row in 0..mr {
        for h in 0..b_regs {
            let ident = &acc_idents[row * b_regs + h];
            let offset = if row == 0 && h == 0 {
                quote! { c }
            } else if h == 0 {
                let row_val = row;
                quote! { c.add(#row_val * ldc) }
            } else {
                let row_val = row;
                let col_offset = h * 16;
                quote! { c.add(#row_val * ldc + #col_offset) }
            };
            loads.push(quote! {
                let mut #ident = _mm512_loadu_ps(#offset);
            });
        }
    }
    loads
}

/// Generate the inner K-loop body.
fn generate_k_body(
    mr: usize,
    nr: usize,
    b_regs: usize,
    acc_idents: &[proc_macro2::Ident],
) -> Vec<TokenStream2> {
    let mut body = Vec::new();

    // Load B registers
    let nr_val = nr;
    for h in 0..b_regs {
        let b_ident = format_ident!("b{}", h);
        let offset = h * 16;
        body.push(quote! {
            let #b_ident = _mm512_loadu_ps(b.add(p * #nr_val + #offset));
        });
    }

    // Broadcast A and FMA for each row
    let mr_val = mr;
    for row in 0..mr {
        let a_ident = format_ident!("a{}", row);
        body.push(quote! {
            let #a_ident = _mm512_set1_ps(*a.add(p * #mr_val + #row));
        });
        for h in 0..b_regs {
            let c_ident = &acc_idents[row * b_regs + h];
            let b_ident = format_ident!("b{}", h);
            body.push(quote! {
                #c_ident = _mm512_fmadd_ps(#a_ident, #b_ident, #c_ident);
            });
        }
    }

    body
}

/// Generate C store statements.
fn generate_c_stores(
    mr: usize,
    b_regs: usize,
    acc_idents: &[proc_macro2::Ident],
) -> Vec<TokenStream2> {
    let mut stores = Vec::new();
    for row in 0..mr {
        for h in 0..b_regs {
            let ident = &acc_idents[row * b_regs + h];
            let offset = if row == 0 && h == 0 {
                quote! { c }
            } else if h == 0 {
                let row_val = row;
                quote! { c.add(#row_val * ldc) }
            } else {
                let row_val = row;
                let col_offset = h * 16;
                quote! { c.add(#row_val * ldc + #col_offset) }
            };
            stores.push(quote! {
                _mm512_storeu_ps(#offset, #ident);
            });
        }
    }
    stores
}
