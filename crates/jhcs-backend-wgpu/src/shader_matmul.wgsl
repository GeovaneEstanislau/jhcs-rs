struct MatrixUniforms {
    m: u32,
    k: u32,
    n: u32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> uniforms: MatrixUniforms;
@group(0) @binding(1) var<storage, read> matrix_a: array<f32>;
@group(0) @binding(2) var<storage, read> matrix_b: array<f32>;
@group(0) @binding(3) var<storage, read_write> matrix_c: array<f32>;

@compute @workgroup_size(16, 16)
fn matmul_main(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let col = global_id.x;
    let row = global_id.y;

    if (row >= uniforms.m || col >= uniforms.n) {
        return;
    }

    var sum: f32 = 0.0;
    let k = uniforms.k;
    let n = uniforms.n;

    for (var p: u32 = 0u; p < k; p = p + 1u) {
        let a_val = matrix_a[row * k + p];
        let b_val = matrix_b[p * n + col];
        sum = sum + a_val * b_val;
    }

    matrix_c[row * n + col] = sum;
}
