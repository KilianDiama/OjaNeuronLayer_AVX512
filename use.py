#pragma once
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>
#include <limits>
#include <cstdlib>

#ifdef _OPENMP
#include <omp.h>
#endif

// -----------------------------------------------------------------------------
// oja : Oja's rule neuron layer, HPC-grade, deterministic, easy integration
// -----------------------------------------------------------------------------
// - Single-header, header-only
// - Deterministic RNG (xorshift64*)
// - 64B-aligned storage
// - AVX2 / AVX-512 optimized dot + update (if available), scalar fallback
// - Parallel over neurons (OpenMP optional)
// - Explicit invariants + NaN/Inf guard
// -----------------------------------------------------------------------------
namespace oja {

// -----------------------------------------------------------------------------
// Compile-time feature flags
// -----------------------------------------------------------------------------
#ifndef OJA_ENABLE_AVX2
#define OJA_ENABLE_AVX2 1
#endif

#ifndef OJA_ENABLE_AVX512
#define OJA_ENABLE_AVX512 1
#endif

#ifndef OJA_ENABLE_DEBUG_CHECKS
#define OJA_ENABLE_DEBUG_CHECKS 1
#endif

// -----------------------------------------------------------------------------
// Small deterministic RNG (xorshift64*)
// -----------------------------------------------------------------------------
class XorShift64Star {
    std::uint64_t state_;
    static std::uint64_t scramble_seed(std::uint64_t s) noexcept {
        // simple splitmix-like scrambling to avoid bad low-entropy seeds
        s += 0x9E3779B97F4A7C15ULL;
        s = (s ^ (s >> 30)) * 0xBF58476D1CE4E5B9ULL;
        s = (s ^ (s >> 27)) * 0x94D049BB133111EBULL;
        s ^= (s >> 31);
        return s ? s : 1ULL;
    }
public:
    explicit XorShift64Star(std::uint64_t seed = 1u) noexcept
        : state_(scramble_seed(seed ? seed : 1u)) {}

    std::uint64_t next_u64() noexcept {
        std::uint64_t x = state_;
        x ^= x >> 12;
        x ^= x >> 25;
        x ^= x >> 27;
        state_ = x;
        return x * 0x2545F4914F6CDD1DULL;
    }

    double next_uniform() noexcept {
        // [0,1)
        constexpr double inv = 1.0 / static_cast<double>(UINT64_C(1) << 53);
        return static_cast<double>(next_u64() >> 11) * inv;
    }

    double next_gaussian(double stddev) noexcept {
        // Box-Muller, deterministic
        double u1 = next_uniform();
        double u2 = next_uniform();
        if (u1 <= 0.0) u1 = std::numeric_limits<double>::min();
        const double r = std::sqrt(-2.0 * std::log(u1));
        const double theta = 2.0 * 3.14159265358979323846 * u2;
        return stddev * (r * std::cos(theta));
    }
};

// -----------------------------------------------------------------------------
// NaN / Inf guard
// -----------------------------------------------------------------------------
inline bool is_finite(double x) noexcept {
    return std::isfinite(x);
}

// -----------------------------------------------------------------------------
// AlignedArray : 64-byte aligned, RAII, move-only
// -----------------------------------------------------------------------------
template<typename T, std::size_t Alignment = 64>
class AlignedArray {
    static_assert(Alignment >= alignof(T), "Alignment must be >= alignof(T)");
    T* data_ = nullptr;
    std::size_t size_ = 0;

    static T* allocate(std::size_t n) {
        if (n == 0) return nullptr;
        const std::size_t bytes = n * sizeof(T);
#if defined(_MSC_VER)
        void* p = _aligned_malloc(bytes, Alignment);
        if (!p) throw std::bad_alloc();
        return static_cast<T*>(p);
#else
        void* p = nullptr;
        const std::size_t aligned_bytes =
            ((bytes + Alignment - 1) / Alignment) * Alignment;
        p = std::aligned_alloc(Alignment, aligned_bytes);
        if (!p) throw std::bad_alloc();
        return static_cast<T*>(p);
#endif
    }

    static void deallocate(T* p) noexcept {
        if (!p) return;
#if defined(_MSC_VER)
        _aligned_free(p);
#else
        std::free(p);
#endif
    }

public:
    explicit AlignedArray(std::size_t n = 0)
        : data_(allocate(n)), size_(n) {}

    ~AlignedArray() {
        deallocate(data_);
    }

    AlignedArray(const AlignedArray&) = delete;
    AlignedArray& operator=(const AlignedArray&) = delete;

    AlignedArray(AlignedArray&& other) noexcept
        : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    AlignedArray& operator=(AlignedArray&& other) noexcept {
        if (this != &other) {
            deallocate(data_);
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    std::size_t size() const noexcept { return size_; }
    T* data() noexcept { return data_; }
    const T* data() const noexcept { return data_; }

    T& operator[](std::size_t i) noexcept { return data_[i]; }
    const T& operator[](std::size_t i) const noexcept { return data_[i]; }

    void fill(T value) noexcept {
        for (std::size_t i = 0; i < size_; ++i) data_[i] = value;
    }
};

// -----------------------------------------------------------------------------
// Rational activation with clamp
// y = d / (1 + alpha * d)
// -----------------------------------------------------------------------------
struct RationalActivation {
    double alpha = 0.1;
    double clamp_abs = 10.0;
    double eps = 1e-12;

    double operator()(double d) const noexcept {
        double denom = 1.0 + alpha * d;
        if (std::fabs(denom) < eps) {
            denom = (denom >= 0.0 ? eps : -eps);
        }
        double y = d / denom;
        if (y > clamp_abs) y = clamp_abs;
        else if (y < -clamp_abs) y = -clamp_abs;
        return y;
    }
};

// -----------------------------------------------------------------------------
// Low-level dot product kernels
// -----------------------------------------------------------------------------
inline double dot_scalar(const double* __restrict__ x,
                         const double* __restrict__ w,
                         std::size_t D) noexcept
{
    double acc = 0.0;
#pragma omp simd reduction(+:acc)
    for (std::size_t k = 0; k < D; ++k) {
        acc += x[k] * w[k];
    }
    return acc;
}

#if (OJA_ENABLE_AVX2 || OJA_ENABLE_AVX512)
#include <immintrin.h>
#endif

#if OJA_ENABLE_AVX512 && defined(__AVX512F__)
inline double dot_avx512(const double* __restrict__ x,
                         const double* __restrict__ w,
                         std::size_t D) noexcept
{
    const std::size_t step = 8;
    std::size_t k = 0;
    __m512d vacc = _mm512_setzero_pd();

    const std::size_t D_unroll = (D / step) * step;
    for (; k < D_unroll; k += step) {
        _mm_prefetch(reinterpret_cast<const char*>(x + k + 64), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(w + k + 64), _MM_HINT_T0);

        __m512d vx = _mm512_load_pd(x + k);
        __m512d vw = _mm512_load_pd(w + k);
        vacc = _mm512_fmadd_pd(vx, vw, vacc);
    }

    alignas(64) double tmp[8];
    _mm512_store_pd(tmp, vacc);
    double acc = tmp[0] + tmp[1] + tmp[2] + tmp[3]
               + tmp[4] + tmp[5] + tmp[6] + tmp[7];

    for (; k < D; ++k) {
        acc += x[k] * w[k];
    }
    return acc;
}
#endif

#if OJA_ENABLE_AVX2 && defined(__AVX2__)
inline double dot_avx2(const double* __restrict__ x,
                       const double* __restrict__ w,
                       std::size_t D) noexcept
{
    const std::size_t step = 4;
    std::size_t k = 0;
    __m256d vacc0 = _mm256_setzero_pd();
    __m256d vacc1 = _mm256_setzero_pd();

    const std::size_t D_unroll = (D / (2 * step)) * (2 * step);
    for (; k < D_unroll; k += 2 * step) {
        _mm_prefetch(reinterpret_cast<const char*>(x + k + 64), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(w + k + 64), _MM_HINT_T0);

        __m256d vx0 = _mm256_load_pd(x + k);
        __m256d vw0 = _mm256_load_pd(w + k);
        vacc0 = _mm256_fmadd_pd(vx0, vw0, vacc0);

        __m256d vx1 = _mm256_load_pd(x + k + step);
        __m256d vw1 = _mm256_load_pd(w + k + step);
        vacc1 = _mm256_fmadd_pd(vx1, vw1, vacc1);
    }

    __m256d vsum = _mm256_add_pd(vacc0, vacc1);
    alignas(32) double tmp[4];
    _mm256_store_pd(tmp, vsum);
    double acc = tmp[0] + tmp[1] + tmp[2] + tmp[3];

    for (; k < D; ++k) {
        acc += x[k] * w[k];
    }
    return acc;
}
#endif

inline double dot_product_optimized(const double* __restrict__ x,
                                    const double* __restrict__ w,
                                    std::size_t D) noexcept
{
#if OJA_ENABLE_AVX512 && defined(__AVX512F__)
    return dot_avx512(x, w, D);
#elif OJA_ENABLE_AVX2 && defined(__AVX2__)
    return dot_avx2(x, w, D);
#else
    return dot_scalar(x, w, D);
#endif
}

// -----------------------------------------------------------------------------
// WeightMatrix : [num_neurons x stride], stride padded to multiple of 4
// -----------------------------------------------------------------------------
class WeightMatrix {
    std::size_t input_dim_;
    std::size_t num_neurons_;
    std::size_t stride_;
    AlignedArray<double> weights_;

public:
    WeightMatrix(std::size_t input_dim, std::size_t num_neurons)
        : input_dim_(input_dim),
          num_neurons_(num_neurons),
          // stride multiple of 4 doubles (32B) for AVX2, AVX-512 ok too
          stride_(((input_dim + 3) / 4) * 4),
          weights_(num_neurons * stride_)
    {
        if (input_dim_ == 0 || num_neurons_ == 0)
            throw std::invalid_argument("input_dim and num_neurons must be > 0");
    }

    std::size_t input_dim() const noexcept { return input_dim_; }
    std::size_t num_neurons() const noexcept { return num_neurons_; }
    std::size_t stride() const noexcept { return stride_; }

    double* data() noexcept { return weights_.data(); }
    const double* data() const noexcept { return weights_.data(); }

    double& w(std::size_t n, std::size_t k) noexcept {
        return weights_.data()[n * stride_ + k];
    }
    double w(std::size_t n, std::size_t k) const noexcept {
        return weights_.data()[n * stride_ + k];
    }

    void init_random(unsigned int seed, double scale) {
        XorShift64Star rng(seed ? seed : 1u);
        double* w_base = weights_.data();
        for (std::size_t n = 0; n < num_neurons_; ++n) {
            double* row = w_base + n * stride_;
            for (std::size_t k = 0; k < input_dim_; ++k) {
                row[k] = rng.next_gaussian(scale);
            }
            for (std::size_t k = input_dim_; k < stride_; ++k) {
                row[k] = 0.0;
            }
        }
    }

    double dot_product(const double* __restrict__ x,
                       std::size_t neuron_index) const noexcept
    {
        assert(neuron_index < num_neurons_);
        const double* __restrict__ row =
            weights_.data() + neuron_index * stride_;
        return dot_product_optimized(x, row, input_dim_);
    }

    // Oja update: w <- w + lr * y * (x - y * w)
    void oja_update_neuron(const double* __restrict__ x,
                           std::size_t neuron_index,
                           double y,
                           double learning_rate) noexcept
    {
        assert(neuron_index < num_neurons_);
        double* __restrict__ row = weights_.data() + neuron_index * stride_;
        const std::size_t D = input_dim_;
        const double factor = learning_rate * y;

#if OJA_ENABLE_AVX512 && defined(__AVX512F__)
        const std::size_t step = 8;
        std::size_t k = 0;
        __m512d vfactor = _mm512_set1_pd(factor);
        __m512d vy = _mm512_set1_pd(y);

        const std::size_t D_unroll = (D / step) * step;
        for (; k < D_unroll; k += step) {
            _mm_prefetch(reinterpret_cast<const char*>(x + k + 64), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(row + k + 64), _MM_HINT_T0);

            __m512d vx = _mm512_load_pd(x + k);
            __m512d vw = _mm512_load_pd(row + k);
            __m512d vyw = _mm512_mul_pd(vy, vw);
            __m512d diff = _mm512_sub_pd(vx, vyw);
            __m512d upd = _mm512_fmadd_pd(vfactor, diff, vw);
            _mm512_store_pd(row + k, upd);
        }
        for (; k < D; ++k) {
            double& w_ = row[k];
            const double xk = x[k];
            w_ += factor * (xk - y * w_);
        }
#elif OJA_ENABLE_AVX2 && defined(__AVX2__)
        const std::size_t step = 4;
        std::size_t k = 0;
        __m256d vfactor = _mm256_set1_pd(factor);
        __m256d vy = _mm256_set1_pd(y);

        const std::size_t D_unroll = (D / step) * step;
        for (; k < D_unroll; k += step) {
            _mm_prefetch(reinterpret_cast<const char*>(x + k + 64), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(row + k + 64), _MM_HINT_T0);

            __m256d vx = _mm256_load_pd(x + k);
            __m256d vw = _mm256_load_pd(row + k);
            __m256d vyw = _mm256_mul_pd(vy, vw);
            __m256d diff = _mm256_sub_pd(vx, vyw);
            __m256d upd = _mm256_fmadd_pd(vfactor, diff, vw);
            _mm256_store_pd(row + k, upd);
        }
        for (; k < D; ++k) {
            double& w_ = row[k];
            const double xk = x[k];
            w_ += factor * (xk - y * w_);
        }
#else
#pragma omp simd
        for (std::size_t k = 0; k < D; ++k) {
            double& w_ = row[k];
            const double xk = x[k];
            w_ += factor * (xk - y * w_);
        }
#endif
    }

    void renormalize_all(double max_weight_norm) noexcept {
        if (max_weight_norm <= 0.0) return;
        double* __restrict__ w_base = weights_.data();
        const double max2 = max_weight_norm * max_weight_norm;
        const std::size_t D = input_dim_;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (std::ptrdiff_t n = 0; n < static_cast<std::ptrdiff_t>(num_neurons_); ++n) {
            double* row = w_base + static_cast<std::size_t>(n) * stride_;
            double norm2 = 0.0;
#pragma omp simd reduction(+:norm2)
            for (std::size_t k = 0; k < D; ++k) {
                const double w_ = row[k];
                norm2 += w_ * w_;
            }
            if (norm2 > max2 && norm2 > 0.0) {
                const double norm = std::sqrt(norm2);
                const double scale = max_weight_norm / norm;
#pragma omp simd
                for (std::size_t k = 0; k < D; ++k) {
                    row[k] *= scale;
                }
            }
        }
    }

    double max_norm() const noexcept {
        double max_norm = 0.0;
        const double* __restrict__ w_base = weights_.data();
        const std::size_t D = input_dim_;

#ifdef _OPENMP
#pragma omp parallel
        {
            double local_max = 0.0;
#pragma omp for nowait schedule(static)
            for (std::ptrdiff_t n = 0; n < static_cast<std::ptrdiff_t>(num_neurons_); ++n) {
                const double* row = w_base + static_cast<std::size_t>(n) * stride_;
                double norm2 = 0.0;
#pragma omp simd reduction(+:norm2)
                for (std::size_t k = 0; k < D; ++k) {
                    const double w_ = row[k];
                    norm2 += w_ * w_;
                }
                const double norm = std::sqrt(norm2);
                if (norm > local_max) local_max = norm;
            }
#pragma omp critical
            {
                if (local_max > max_norm) max_norm = local_max;
            }
        }
#else
        for (std::size_t n = 0; n < num_neurons_; ++n) {
            const double* row = w_base + n * stride_;
            double norm2 = 0.0;
            for (std::size_t k = 0; k < D; ++k) {
                const double w_ = row[k];
                norm2 += w_ * w_;
            }
            const double norm = std::sqrt(norm2);
            if (norm > max_norm) max_norm = norm;
        }
#endif
        return max_norm;
    }
};

// -----------------------------------------------------------------------------
// OjaNeuronLayer
// -----------------------------------------------------------------------------
template<typename Activation = RationalActivation>
class OjaNeuronLayer {
    WeightMatrix W_;
    std::vector<double> state_;
    double learning_rate_;
    Activation activation_;
    double max_weight_norm_;
    std::size_t step_counter_;
    std::size_t renorm_period_;
    std::vector<double> dot_buffer_;

    void check_finite_vector(const double* v, std::size_t n,
                             const char* context) const
    {
#if OJA_ENABLE_DEBUG_CHECKS
        for (std::size_t i = 0; i < n; ++i) {
            if (!is_finite(v[i])) {
                throw std::runtime_error(std::string("Non-finite value in ") + context);
            }
        }
#else
        (void)v; (void)n; (void)context;
#endif
    }

public:
    OjaNeuronLayer(std::size_t num_neurons,
                   std::size_t input_dim,
                   double learning_rate = 0.005,
                   Activation activation = Activation{},
                   double max_weight_norm = 10.0,
                   std::size_t renorm_period = 256,
                   unsigned int seed = 42)
        : W_(input_dim, num_neurons),
          state_(num_neurons, 0.0),
          learning_rate_(learning_rate),
          activation_(activation),
          max_weight_norm_(max_weight_norm),
          step_counter_(0),
          renorm_period_(renorm_period),
          dot_buffer_(num_neurons, 0.0)
    {
        if (learning_rate_ <= 0.0)
            throw std::invalid_argument("learning_rate must be > 0");
        if (num_neurons == 0 || input_dim == 0)
            throw std::invalid_argument("num_neurons and input_dim must be > 0");
        if (renorm_period_ == 0)
            throw std::invalid_argument("renorm_period must be > 0");
        if (max_weight_norm_ <= 0.0)
            throw std::invalid_argument("max_weight_norm must be > 0");

        const double scale = 1.0 / std::sqrt(static_cast<double>(input_dim));
        W_.init_random(seed, scale);
    }

    std::size_t num_neurons() const noexcept { return W_.num_neurons(); }
    std::size_t input_dim() const noexcept { return W_.input_dim(); }

    const double* get_state() const noexcept { return state_.data(); }
    const double* get_weights() const noexcept { return W_.data(); }

    void set_learning_rate(double lr) {
        if (lr <= 0.0) throw std::invalid_argument("learning_rate must be > 0");
        learning_rate_ = lr;
    }

    void set_max_weight_norm(double m) {
        if (m <= 0.0) throw std::invalid_argument("max_weight_norm must be > 0");
        max_weight_norm_ = m;
    }

    void set_renorm_period(std::size_t p) {
        if (p == 0) throw std::invalid_argument("renorm_period must be > 0");
        renorm_period_ = p;
    }

    Activation& activation() noexcept { return activation_; }
    const Activation& activation() const noexcept { return activation_; }

    // forward: y = f(W^T x)
    void forward(const double* __restrict__ input,
                 double* __restrict__ output) const
    {
        if (!input || !output)
            throw std::invalid_argument("Null pointer in forward");

        const std::size_t N = num_neurons();

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (std::ptrdiff_t n = 0; n < static_cast<std::ptrdiff_t>(N); ++n) {
            const double d = W_.dot_product(input, static_cast<std::size_t>(n));
            double y = activation_(d);
            output[static_cast<std::size_t>(n)] = y;
        }

        check_finite_vector(output, N, "forward output");
    }

    // forward_batch: inputs [B x D], outputs [B x N]
    void forward_batch(const double* __restrict__ inputs,
                       std::size_t batch_size,
                       double* __restrict__ outputs) const
    {
        if (!inputs || !outputs)
            throw std::invalid_argument("Null pointer in forward_batch");
        if (batch_size == 0) return;

        const std::size_t N = num_neurons();
        const std::size_t D = input_dim();

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (std::ptrdiff_t b = 0; b < static_cast<std::ptrdiff_t>(batch_size); ++b) {
            const double* x = inputs + static_cast<std::size_t>(b) * D;
            double* y = outputs + static_cast<std::size_t>(b) * N;
            for (std::size_t n = 0; n < N; ++n) {
                const double d = W_.dot_product(x, n);
                double out = activation_(d);
                y[n] = out;
            }
            // Optionally check per batch row
            // check_finite_vector(y, N, "forward_batch output row");
        }
    }

    // train_step: one input, updates state_
    void train_step(const double* __restrict__ input)
    {
        if (!input)
            throw std::invalid_argument("Null pointer in train_step");

        const std::size_t N = num_neurons();

        // Pass 1: dot products (parallel)
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (std::ptrdiff_t n = 0; n < static_cast<std::ptrdiff_t>(N); ++n) {
            dot_buffer_[static_cast<std::size_t>(n)] =
                W_.dot_product(input, static_cast<std::size_t>(n));
        }

        // Pass 2: activation + update (parallel over neurons)
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (std::ptrdiff_t n = 0; n < static_cast<std::ptrdiff_t>(N); ++n) {
            const std::size_t idx = static_cast<std::size_t>(n);
            const double d = dot_buffer_[idx];
            double y = activation_(d);

#if OJA_ENABLE_DEBUG_CHECKS
            if (!is_finite(y)) {
                throw std::runtime_error("Non-finite activation in train_step");
            }
#endif
            state_[idx] = y;
            W_.oja_update_neuron(input, idx, y, learning_rate_);
        }

        if (++step_counter_ % renorm_period_ == 0) {
            W_.renormalize_all(max_weight_norm_);
        }
    }

    // train_batch: loop over train_step (deterministic)
    void train_batch(const double* __restrict__ inputs,
                     std::size_t batch_size)
    {
        if (!inputs)
            throw std::invalid_argument("Null pointer in train_batch");
        if (batch_size == 0) return;

        const std::size_t D = input_dim();
        for (std::size_t b = 0; b < batch_size; ++b) {
            const double* x = inputs + b * D;
            train_step(x);
        }
    }

    double current_max_weight_norm() const noexcept {
        return W_.max_norm();
    }

    std::size_t step_counter() const noexcept { return step_counter_; }
};

} // namespace oja
