#include <immintrin.h>
#include <omp.h>
#include <algorithm>
#include <memory>
#include <vector>
#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <optional>

// --------------------------------------------------
// RAII Aligned Array (64-byte aligned, portable)
// --------------------------------------------------
class AlignedArray {
    std::unique_ptr<double[], void(*)(void*)> data_;
    size_t size_ = 0;

    static void deleter(void* p) { _mm_free(p); }

public:
    explicit AlignedArray(size_t n)
        : data_(nullptr, deleter), size_(n)
    {
        if (n == 0) return;
        double* ptr = static_cast<double*>(_mm_malloc(n * sizeof(double), 64));
        if (!ptr) throw std::bad_alloc();
        data_.reset(ptr);
        std::fill(ptr, ptr + n, 0.0);
    }

    double* data() noexcept { return data_.get(); }
    const double* data() const noexcept { return data_.get(); }
    size_t size() const noexcept { return size_; }

    AlignedArray(const AlignedArray&) = delete;
    AlignedArray& operator=(const AlignedArray&) = delete;

    AlignedArray(AlignedArray&& other) noexcept = default;
    AlignedArray& operator=(AlignedArray&& other) noexcept = default;
};

// --------------------------------------------------
// Ultra-Optimized Oja Neuron Layer (10/10 Prod)
// --------------------------------------------------
alignas(64) class OjaNeuronLayer {
    size_t num_neurons_;
    size_t input_dim_;
    size_t stride_;  

    AlignedArray synapses_; // [input_dim_ x stride_]
    AlignedArray state_;    // [stride_]

    double learning_rate_;
    double alpha_;
    size_t block_size_;
    bool use_omp_;

public:
    OjaNeuronLayer(size_t num_neurons, size_t input_dim,
                   double learning_rate = 0.005,
                   double alpha = 0.1,
                   size_t block_size = 512,
                   bool use_omp = true)
        : num_neurons_(num_neurons),
          input_dim_(input_dim),
          stride_(((num_neurons + 63) & ~63) + 8),
          synapses_(input_dim * stride_),
          state_(stride_),
          learning_rate_(learning_rate),
          alpha_(alpha),
          block_size_(block_size),
          use_omp_(use_omp)
    {
        if (num_neurons == 0 || input_dim == 0)
            throw std::invalid_argument("num_neurons and input_dim must be > 0");
    }

    void set_learning_rate(double lr) { learning_rate_ = lr; }
    void set_alpha(double a) { alpha_ = a; }
    void set_block_size(size_t bs) { block_size_ = bs; }

private:
    inline __m512d activate(__m512d d) const noexcept {
        const __m512d one   = _mm512_set1_pd(1.0);
        const __m512d alpha = _mm512_set1_pd(alpha_);
        __m512d denom = _mm512_fmadd_pd(alpha, d, one);
        __m512d inv = _mm512_rcp14_pd(denom);
        inv = _mm512_mul_pd(inv, _mm512_fnmadd_pd(denom, inv, _mm512_set1_pd(2.0)));
        inv = _mm512_mul_pd(inv, _mm512_fnmadd_pd(denom, inv, _mm512_set1_pd(2.0)));
        return _mm512_mul_pd(d, inv);
    }

    inline void compute_dot(const double* __restrict__ input,
                            size_t neuron_idx,
                            __m512d &dot0, __m512d &dot1,
                            __mmask8 mask0, __mmask8 mask1) const noexcept
    {
        dot0 = _mm512_setzero_pd();
        dot1 = _mm512_setzero_pd();
        for (size_t k = 0; k < input_dim_; ++k) {
            __m512d x = _mm512_set1_pd(input[k]);
            const double* syn = &synapses_.data()[k * stride_ + neuron_idx];
            dot0 = _mm512_fmadd_pd(x, _mm512_maskz_load_pd(mask0, syn), dot0);
            dot1 = _mm512_fmadd_pd(x, _mm512_maskz_load_pd(mask1, syn + 8), dot1);
        }
    }

    inline void update_weights(const double* __restrict__ input,
                               size_t neuron_idx,
                               const __m512d &y0, const __m512d &y1,
                               __mmask8 mask0, __mmask8 mask1) noexcept
    {
        const __m512d neg_lr = _mm512_set1_pd(-learning_rate_);
        __m512d delta0 = _mm512_mul_pd(neg_lr, y0);
        __m512d delta1 = _mm512_mul_pd(neg_lr, y1);

        for (size_t k = 0; k < input_dim_; ++k) {
            __m512d x = _mm512_set1_pd(input[k]);
            double* wptr = &synapses_.data()[k * stride_ + neuron_idx];

            __m512d w0 = _mm512_maskz_load_pd(mask0, wptr);
            __m512d w1 = _mm512_maskz_load_pd(mask1, wptr + 8);

            __m512d dw0 = _mm512_fmsub_pd(y0, w0, x);
            __m512d dw1 = _mm512_fmsub_pd(y1, w1, x);

            _mm512_mask_store_pd(wptr, mask0, _mm512_fmadd_pd(delta0, dw0, w0));
            _mm512_mask_store_pd(wptr + 8, mask1, _mm512_fmadd_pd(delta1, dw1, w1));
        }
    }

public:
    void forward(const double* __restrict__ input) {
        if (!input) throw std::invalid_argument("Input pointer is null");

        auto loop_body = [this, input](size_t ii, size_t limit_i) {
            for (size_t i = ii; i < limit_i; i += 16) {
                size_t rem = limit_i - i;
                __mmask8 mask0 = _cvtu32_mask8((1U << std::min(size_t(8), rem)) - 1);
                __mmask8 mask1 = _cvtu32_mask8((rem > 8) ? (1U << std::min(size_t(8), rem - 8)) - 1 : 0);

                __m512d dot0, dot1;
                compute_dot(input, i, dot0, dot1, mask0, mask1);

                __m512d y0 = activate(dot0);
                __m512d y1 = activate(dot1);

                _mm512_mask_store_pd(&state_.data()[i], mask0, y0);
                _mm512_mask_store_pd(&state_.data()[i + 8], mask1, y1);

                update_weights(input, i, y0, y1, mask0, mask1);
            }
        };

        if (use_omp_) {
            #pragma omp parallel for schedule(static)
            for (size_t ii = 0; ii < num_neurons_; ii += block_size_) {
                size_t limit_i = std::min(ii + block_size_, num_neurons_);
                loop_body(ii, limit_i);
            }
        } else {
            for (size_t ii = 0; ii < num_neurons_; ii += block_size_) {
                size_t limit_i = std::min(ii + block_size_, num_neurons_);
                loop_body(ii, limit_i);
            }
        }
    }

    const double* get_state() const noexcept { return state_.data(); }
    size_t size() const noexcept { return num_neurons_; }

    double max_weight_norm() const noexcept {
        double max_norm = 0.0;
        for (size_t n = 0; n < num_neurons_; ++n) {
            double norm = 0.0;
            for (size_t k = 0; k < input_dim_; ++k) {
                double w = synapses_.data()[k * stride_ + n];
                norm += w * w;
            }
            max_norm = std::max(max_norm, std::sqrt(norm));
        }
        return max_norm;
    }
}; 
