#include <benchmark/benchmark.h>
#include <vector>
#include <random>
#include <algorithm>
#include <immintrin.h>
#include <vector>

// Наивное матричное умножение
void naive_matmul(int m, int n, int p, const float* a, const float* b, float* c) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += a[i * n + k] * b[k * p + j]; 
            }
            c[i * p + j] = sum;
        }
    }
}

// Наивная свертка (упрощенная версия без padding/stride для базового сравнения)
void naive_conv(int output_channels, int output_height, int output_width,
                int input_channels, int kernel_size,
                const float* input, const float* weight, float* output) {
    
    int in_h = output_height + kernel_size - 1;
    int in_w = output_width + kernel_size - 1;

    for (int oc = 0; oc < output_channels; oc++) {
        for (int oh = 0; oh < output_height; oh++) {
            for (int ow = 0; ow < output_width; ow++) {
                float sum = 0.0f;
                for (int ic = 0; ic < input_channels; ic++) {
                    for (int kh = 0; kh < kernel_size; kh++) {
                        for (int kw = 0; kw < kernel_size; kw++) {
                            int in_idx = ic * (in_h * in_w) + (oh + kh) * in_w + (ow + kw);
                            int wt_idx = oc * (input_channels * kernel_size * kernel_size) + 
                                         ic * (kernel_size * kernel_size) + kh * kernel_size + kw;
                            
                            sum += input[in_idx] * weight[wt_idx];
                        }
                    }
                }
                int out_idx = oc * (output_height * output_width) + oh * output_width + ow;
                output[out_idx] = sum;
            }
        }
    }
}

// Макрос для замера наивного матричного умножения
static void BM_NaiveMatMul(benchmark::State& state) {
    int size = state.range(0);
    
    std::vector<float> a(size * size, 1.0f);
    std::vector<float> b(size * size, 2.0f);
    std::vector<float> c(size * size, 0.0f);

    for (auto _ : state) {
        naive_matmul(size, size, size, a.data(), b.data(), c.data());
    }
}

// для квадратных матриц от 64x64 до 512x512
BENCHMARK(BM_NaiveMatMul)->RangeMultiplier(2)->Range(64, 512);

// Cache-friendly матричное умножение
void cache_friendly_matmul(int m, int n, int p, const float* a, const float* b, float* c) {
    std::vector<float> b_t(n * p);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            b_t[j * n + i] = b[i * p + j];
        }
    }

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += a[i * n + k] * b_t[j * n + k];
            }
            c[i * p + j] = sum;
        }
    }
}

// Макрос для замера
static void BM_CacheFriendlyMatMul(benchmark::State& state) {
    int size = state.range(0);
    
    std::vector<float> a(size * size, 1.0f);
    std::vector<float> b(size * size, 2.0f);
    std::vector<float> c(size * size, 0.0f);

    for (auto _ : state) {
        cache_friendly_matmul(size, size, size, a.data(), b.data(), c.data());
    }
}

BENCHMARK(BM_CacheFriendlyMatMul)->RangeMultiplier(2)->Range(64, 512);

// Cache-friendly + Tiling
void tiled_matmul(int m, int n, int p, const float* a, const float* b, float* c) {
    const int BLOCK_SIZE = 32;
    
    std::vector<float> b_t(n * p);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            b_t[j * n + i] = b[i * p + j];
        }
    }

    for (int i = 0; i < m * p; i++) {
        c[i] = 0.0f;
    }

    for (int i = 0; i < m; i += BLOCK_SIZE) {
        for (int j = 0; j < p; j += BLOCK_SIZE) {
            for (int k = 0; k < n; k += BLOCK_SIZE) {
                
                for (int ii = i; ii < std::min(i + BLOCK_SIZE, m); ii++) {
                    for (int jj = j; jj < std::min(j + BLOCK_SIZE, p); jj++) {
                        
                        float sum = 0.0f;
                        for (int kk = k; kk < std::min(k + BLOCK_SIZE, n); kk++) {
                            sum += a[ii * n + kk] * b_t[jj * n + kk];
                        }
                        c[ii * p + jj] += sum;
                    }
                }

            }
        }
    }
}

// для замера Tiled версии
static void BM_TiledMatMul(benchmark::State& state) {
    int size = state.range(0);
    
    std::vector<float> a(size * size, 1.0f);
    std::vector<float> b(size * size, 2.0f);
    std::vector<float> c(size * size, 0.0f);

    for (auto _ : state) {
        tiled_matmul(size, size, size, a.data(), b.data(), c.data());
    }
}

BENCHMARK(BM_TiledMatMul)->RangeMultiplier(2)->Range(64, 512);

// AVX + Tiling + Zero Padding
void avx_tiled_matmul(int m, int n, int p, const float* a, const float* b, float* c) {
    const int BLOCK_SIZE = 32;

    int m_p = (m + 7) / 8 * 8;
    int n_p = (n + 7) / 8 * 8;
    int p_p = (p + 7) / 8 * 8;

    std::vector<float> a_pad(m_p * n_p, 0.0f);
    std::vector<float> b_pad(n_p * p_p, 0.0f);
    std::vector<float> c_pad(m_p * p_p, 0.0f);

    for(int i = 0; i < m; ++i) 
        for(int j = 0; j < n; ++j) 
            a_pad[i * n_p + j] = a[i * n + j];
            
    for(int i = 0; i < n; ++i) 
        for(int j = 0; j < p; ++j) 
            b_pad[i * p_p + j] = b[i * p + j];

    for (int i_b = 0; i_b < m_p; i_b += BLOCK_SIZE) {
        for (int k_b = 0; k_b < n_p; k_b += BLOCK_SIZE) {
            for (int j_b = 0; j_b < p_p; j_b += BLOCK_SIZE) {

                for (int i = i_b; i < std::min(i_b + BLOCK_SIZE, m_p); i++) {
                    for (int k = k_b; k < std::min(k_b + BLOCK_SIZE, n_p); k++) {

                        __m256 a_vec = _mm256_set1_ps(a_pad[i * n_p + k]);
                        for (int j = j_b; j < std::min(j_b + BLOCK_SIZE, p_p); j += 8) {
                            
                            __m256 b_vec = _mm256_loadu_ps(&b_pad[k * p_p + j]);
                            __m256 c_vec = _mm256_loadu_ps(&c_pad[i * p_p + j]);

                            __m256 mul_res = _mm256_mul_ps(a_vec, b_vec);
                            c_vec = _mm256_add_ps(c_vec, mul_res);
                            _mm256_storeu_ps(&c_pad[i * p_p + j], c_vec);
                        }
                    }
                }
            }
        }
    }

    for(int i = 0; i < m; ++i) {
        for(int j = 0; j < p; ++j) {
            c[i * p + j] = c_pad[i * p_p + j];
        }
    }
}

// для замера
static void BM_AvxTiledMatMul(benchmark::State& state) {
    int size = state.range(0);
    
    std::vector<float> a(size * size, 1.0f);
    std::vector<float> b(size * size, 2.0f);
    std::vector<float> c(size * size, 0.0f);

    for (auto _ : state) {
        avx_tiled_matmul(size, size, size, a.data(), b.data(), c.data());
    }
}

BENCHMARK(BM_AvxTiledMatMul)->RangeMultiplier(2)->Range(64, 512);

// Im2Col с поддержкой Stride и Padding
void im2col_advanced(const float* input, int input_channels, int height, int width, 
                     int kernel_size, int stride, int padding, float* col_matrix) {

    int out_h = (height + 2 * padding - kernel_size) / stride + 1;
    int out_w = (width + 2 * padding - kernel_size) / stride + 1;
    
    int col_idx = 0;

    for (int y = 0; y < out_h; y++) {
        for (int x = 0; x < out_w; x++) {
            
            int row_idx = 0;
            for (int c = 0; c < input_channels; c++) {
                for (int ky = 0; ky < kernel_size; ky++) {
                    for (int kx = 0; kx < kernel_size; kx++) {
                        
                        int in_y = y * stride + ky - padding;
                        int in_x = x * stride + kx - padding;
                        
                        if (in_y >= 0 && in_y < height && in_x >= 0 && in_x < width) {
                            int in_idx = c * (height * width) + in_y * width + in_x;
                            col_matrix[row_idx * (out_h * out_w) + col_idx] = input[in_idx];
                        } else {
                            col_matrix[row_idx * (out_h * out_w) + col_idx] = 0.0f;
                        }
                        
                        row_idx++;
                    }
                }
            }
            col_idx++;
        }
    }
}

// оптимизированная свертка (Wrapper для Im2Col + GEMM)
void im2col_conv_advanced(int output_channels, int height, int width,
                          int input_channels, int kernel_size, int stride, int padding,
                          const float* input, const float* weight, float* output) {
    
    int out_h = (height + 2 * padding - kernel_size) / stride + 1;
    int out_w = (width + 2 * padding - kernel_size) / stride + 1;

    int m = output_channels;                               
    int n = input_channels * kernel_size * kernel_size;    
    int p = out_h * out_w;                                 

    std::vector<float> col_matrix(n * p);

    im2col_advanced(input, input_channels, height, width, kernel_size, stride, padding, col_matrix.data());

    avx_tiled_matmul(m, n, p, weight, col_matrix.data(), output);
}

// замер наивной свертки
static void BM_NaiveConv(benchmark::State& state) {
    int spatial_size = state.range(0);
    int in_c = 16;
    int out_c = 16;
    int k_size = 3;

    std::vector<float> input(in_c * spatial_size * spatial_size, 1.0f);
    std::vector<float> weight(out_c * in_c * k_size * k_size, 1.0f);
    
    int out_h = spatial_size - k_size + 1;
    int out_w = spatial_size - k_size + 1;
    std::vector<float> output(out_c * out_h * out_w, 0.0f);

    for (auto _ : state) {
        naive_conv(out_c, out_h, out_w, in_c, k_size, input.data(), weight.data(), output.data());
    }
}

static void BM_Im2ColConvAdvanced(benchmark::State& state) {
    int spatial_size = state.range(0);
    int in_c = 16;
    int out_c = 16;
    int k_size = 3;
    int stride = 2;
    int padding = 1;

    std::vector<float> input(in_c * spatial_size * spatial_size, 1.0f);
    std::vector<float> weight(out_c * in_c * k_size * k_size, 1.0f);
    
    int out_h = (spatial_size + 2 * padding - k_size) / stride + 1;
    int out_w = (spatial_size + 2 * padding - k_size) / stride + 1;
    std::vector<float> output(out_c * out_h * out_w, 0.0f);

    for (auto _ : state) {
        im2col_conv_advanced(out_c, spatial_size, spatial_size, in_c, k_size, stride, padding, 
                             input.data(), weight.data(), output.data());
    }
}

BENCHMARK(BM_NaiveConv)->RangeMultiplier(2)->Range(16, 128);
BENCHMARK(BM_Im2ColConvAdvanced)->RangeMultiplier(2)->Range(16, 128);

BENCHMARK_MAIN();