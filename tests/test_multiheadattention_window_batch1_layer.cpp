// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

// Contract tests for MultiHeadAttention param 19 = window_batch1.
// window_batch1=1: q/k/v are 3-D (w=feature, h=seqlen, c=num_windows), every
// channel is an independent 2-D attention, output is (qdim, src_seqlen, num_windows).
// The optional attn_mask is 2-D (shared by all heads) or 3-D (c=num_heads) and is
// shared by all windows. forward returns -1 for kv_cache != 0, int8_scale_term != 0
// and inconsistent shapes (q.c == k.c == v.c and k.h == v.h are required).

#include "testutil.h"

#include "layer.h"
#include "modelbin.h"

#include <math.h>
#include <vector>

// mask_type: 0 = none, 2 = 2-D mask (dst_seqlen, src_seqlen), 3 = 3-D mask (dst_seqlen, src_seqlen, num_heads)
static ncnn::Mat make_mask(int mask_type, int dst_seqlen, int src_seqlen, int num_heads)
{
    if (mask_type == 2)
        return RandomMat(dst_seqlen, src_seqlen);
    if (mask_type == 3)
        return RandomMat(dst_seqlen, src_seqlen, num_heads);
    return ncnn::Mat();
}

static void make_weights(std::vector<ncnn::Mat>& weights, int embed_dim, int qdim, int kdim, int vdim)
{
    weights.resize(8);
    weights[0] = RandomMat(embed_dim * qdim);
    weights[1] = RandomMat(embed_dim);
    weights[2] = RandomMat(embed_dim * kdim);
    weights[3] = RandomMat(embed_dim);
    weights[4] = RandomMat(embed_dim * vdim);
    weights[5] = RandomMat(embed_dim);
    weights[6] = RandomMat(qdim * embed_dim);
    weights[7] = RandomMat(qdim);
}

static void make_pd(ncnn::ParamDict& pd, int embed_dim, int num_heads, int qdim, int kdim, int vdim, int attn_mask, int window_batch1)
{
    pd.set(0, embed_dim);
    pd.set(1, num_heads);
    pd.set(2, embed_dim * qdim);
    pd.set(3, kdim);
    pd.set(4, vdim);
    pd.set(5, attn_mask);
    pd.set(19, window_batch1);
}

static ncnn::Option fp32_option()
{
    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_packing_layout = false;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_bf16_packed = false;
    opt.use_bf16_storage = false;
    opt.use_vulkan_compute = false;
    return opt;
}

// storage modes for the equivalence test
#define MODE_FP32 0
#define MODE_BF16 1
#define MODE_FP16 2

static const char* mode_name(int mode)
{
    return mode == MODE_BF16 ? "bf16" : mode == MODE_FP16 ? "fp16" : "fp32";
}

static ncnn::Option mode_option(int mode)
{
    ncnn::Option opt = fp32_option();
    if (mode == MODE_BF16)
    {
        opt.use_packing_layout = true;
        opt.use_bf16_storage = true;
    }
    if (mode == MODE_FP16)
    {
        opt.use_packing_layout = true;
        opt.use_fp16_storage = true;
        opt.use_fp16_arithmetic = true;
    }
    return opt;
}

// cast and pack an fp32 input the way Net does for a layer (mirrors tests/testutil.cpp convert_to_optimal_layout)
static void to_layer_layout(const ncnn::Mat& a, ncnn::Mat& b, int mode, const ncnn::Option& opt, const ncnn::Layer* op)
{
    ncnn::Mat a4;
    if (mode == MODE_BF16)
        ncnn::cast_float32_to_bfloat16(a, a4, opt);
    else if (mode == MODE_FP16)
        ncnn::cast_float32_to_float16(a, a4, opt);
    else
        a4 = a;

    if (!opt.use_packing_layout || !op->support_packing)
    {
        b = a4;
        return;
    }

    const int elemcount = a4.dims == 1 ? a4.w * a4.elempack : a4.dims == 2 ? a4.h * a4.elempack : a4.c * a4.elempack;
    const int elembits = a4.elembits();
    int dst_elempack = 1;
    if (elembits == 32)
    {
#if NCNN_AVX512
        if (elemcount % 16 == 0 && ncnn::cpu_support_x86_avx512())
            dst_elempack = 16;
        else if (elemcount % 8 == 0 && ncnn::cpu_support_x86_avx())
            dst_elempack = 8;
        else if (elemcount % 4 == 0)
            dst_elempack = 4;
#elif NCNN_AVX
        if (elemcount % 8 == 0 && ncnn::cpu_support_x86_avx())
            dst_elempack = 8;
        else if (elemcount % 4 == 0)
            dst_elempack = 4;
#else
        if (elemcount % 4 == 0)
            dst_elempack = 4;
#endif
    }
    if (elembits == 16)
    {
#if NCNN_ARM82
        if (elemcount % 8 == 0 && ncnn::cpu_support_arm_asimdhp() && opt.use_fp16_arithmetic && op->support_fp16_storage)
            dst_elempack = 8;
        else if (elemcount % 4 == 0)
            dst_elempack = 4;
#elif NCNN_AVX512
        if (elemcount % 16 == 0 && ncnn::cpu_support_x86_avx512())
            dst_elempack = 16;
        else if (elemcount % 8 == 0 && ncnn::cpu_support_x86_avx())
            dst_elempack = 8;
        else if (elemcount % 4 == 0)
            dst_elempack = 4;
#elif NCNN_AVX
        if (elemcount % 8 == 0 && ncnn::cpu_support_x86_avx())
            dst_elempack = 8;
        else if (elemcount % 4 == 0)
            dst_elempack = 4;
#else
        if (elemcount % 4 == 0)
            dst_elempack = 4;
#endif
    }

    ncnn::convert_packing(a4, b, dst_elempack, opt);
}

// unpack and cast an output back to fp32
static void to_fp32_vanilla(const ncnn::Mat& c4, ncnn::Mat& c, int mode, const ncnn::Option& opt)
{
    ncnn::Mat c1;
    if (c4.elempack != 1)
        ncnn::convert_packing(c4, c1, 1, opt);
    else
        c1 = c4;

    if (c1.elembits() == 16 && mode == MODE_BF16)
        ncnn::cast_bfloat16_to_float32(c1, c, opt);
    else if (c1.elembits() == 16 && mode == MODE_FP16)
        ncnn::cast_float16_to_float32(c1, c, opt);
    else
        c = c1;
}

#define RUN_SKIPPED 233

// returns the first non-zero status of load_param / load_model / create_pipeline / forward,
// or RUN_SKIPPED when the layer does not support the requested storage type on this cpu
static int run_layer_mode(ncnn::Layer* op, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, int mode, const std::vector<ncnn::Mat>& bottoms, int top_count, ncnn::Mat& top)
{
    ncnn::Option opt = mode_option(mode);

    int ret = op->load_param(pd);
    if (ret != 0)
        return ret;

    ncnn::ModelBinFromMatArray mb(weights.data());
    ret = op->load_model(mb);
    if (ret != 0)
        return ret;

    ret = op->create_pipeline(opt);
    if (ret != 0)
        return ret;

    if ((mode == MODE_BF16 && !op->support_bf16_storage) || (mode == MODE_FP16 && !op->support_fp16_storage))
    {
        op->destroy_pipeline(opt);
        return RUN_SKIPPED;
    }

    std::vector<ncnn::Mat> bottoms4(bottoms.size());
    for (size_t i = 0; i < bottoms.size(); i++)
    {
        if (bottoms[i].empty())
            bottoms4[i] = bottoms[i];
        else
            to_layer_layout(bottoms[i], bottoms4[i], mode, opt, op);
    }

    std::vector<ncnn::Mat> tops(top_count);
    ret = op->forward(bottoms4, tops, opt);
    if (ret == 0)
        to_fp32_vanilla(tops[0], top, mode, opt);

    op->destroy_pipeline(opt);
    return ret;
}

static int run_layer(ncnn::Layer* op, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const std::vector<ncnn::Mat>& bottoms, int top_count, ncnn::Mat& top)
{
    return run_layer_mode(op, pd, weights, MODE_FP32, bottoms, top_count, top);
}

static float max_abs_diff(const ncnn::Mat& a, const ncnn::Mat& b)
{
    if (a.w != b.w || a.h != b.h || a.d != b.d || a.c != b.c || a.dims != b.dims)
        return 1e30f;
    float m = 0.f;
    for (int q = 0; q < a.c; q++)
    {
        const float* pa = a.channel(q);
        const float* pb = b.channel(q);
        for (int i = 0; i < a.w * a.h * a.d; i++)
        {
            float d = fabsf(pa[i] - pb[i]);
            if (!(d <= m))
                m = d; // also propagates NaN
        }
    }
    return m;
}

// ---- 1. naive vs optimized under test_layer's option matrix ----

// self-attention: a single 3-D blob is q, k and v
static int test_window_self(int embed_dim, int num_heads, int seqlen, int num_windows, int mask_type)
{
    ncnn::ParamDict pd;
    make_pd(pd, embed_dim, num_heads, embed_dim, embed_dim, embed_dim, mask_type ? 1 : 0, 1);

    std::vector<ncnn::Mat> weights;
    make_weights(weights, embed_dim, embed_dim, embed_dim, embed_dim);

    std::vector<ncnn::Mat> as(1);
    as[0] = RandomMat(embed_dim, seqlen, num_windows);
    if (mask_type)
        as.push_back(make_mask(mask_type, seqlen, seqlen, num_heads));

    int ret = test_layer("MultiHeadAttention", pd, weights, as, 1, 0.005f);
    if (ret != 0)
    {
        fprintf(stderr, "test_window_self failed embed_dim=%d num_heads=%d seqlen=%d num_windows=%d mask_type=%d\n", embed_dim, num_heads, seqlen, num_windows, mask_type);
    }
    return ret;
}

// cross-attention: qdim/kdim/vdim differ from embed_dim, dst_seqlen != src_seqlen
static int test_window_cross(int qdim, int kdim, int vdim, int embed_dim, int num_heads, int src_seqlen, int dst_seqlen, int num_windows, int mask_type)
{
    ncnn::ParamDict pd;
    make_pd(pd, embed_dim, num_heads, qdim, kdim, vdim, mask_type ? 1 : 0, 1);

    std::vector<ncnn::Mat> weights;
    make_weights(weights, embed_dim, qdim, kdim, vdim);

    std::vector<ncnn::Mat> as(3);
    as[0] = RandomMat(qdim, src_seqlen, num_windows);
    as[1] = RandomMat(kdim, dst_seqlen, num_windows);
    as[2] = RandomMat(vdim, dst_seqlen, num_windows);
    if (mask_type)
        as.push_back(make_mask(mask_type, dst_seqlen, src_seqlen, num_heads));

    int ret = test_layer("MultiHeadAttention", pd, weights, as, 1, 0.005f);
    if (ret != 0)
    {
        fprintf(stderr, "test_window_cross failed qdim=%d kdim=%d vdim=%d embed_dim=%d num_heads=%d src_seqlen=%d dst_seqlen=%d num_windows=%d mask_type=%d\n", qdim, kdim, vdim, embed_dim, num_heads, src_seqlen, dst_seqlen, num_windows, mask_type);
    }
    return ret;
}

static int test_window_layer_matrix()
{
    static const int windows[] = {1, 3, 4, 8};
    int ret = 0;
    for (int i = 0; i < 4; i++)
    {
        for (int mask_type = 0; mask_type <= 3; mask_type++)
        {
            if (mask_type == 1)
                continue;
            ret |= test_window_self(16, 2, 5, windows[i], mask_type);
            ret |= test_window_cross(12, 20, 9, 16, 4, 5, 7, windows[i], mask_type);
        }
    }
    return ret;
}

// ---- 2. per-window equivalence with plain 2-D MHA ----

// mode MODE_FP32: naive or cpu layer, fp32 options, CompareMat 0.001
// mode MODE_BF16 / MODE_FP16 (cpu layer only, packing on): same code path, max abs diff <= 1e-5
static int test_window_equivalence_one(bool cpu, int mode, bool self, int mask_type, int num_windows)
{
    const int embed_dim = 16;
    const int num_heads = 4;
    const int qdim = self ? embed_dim : 12;
    const int kdim = self ? embed_dim : 20;
    const int vdim = self ? embed_dim : 9;
    const int src_seqlen = 5;
    const int dst_seqlen = self ? 5 : 7;
    const int attn_mask = mask_type ? 1 : 0;
    const char* tag = cpu ? "cpu" : "naive";

    std::vector<ncnn::Mat> weights;
    make_weights(weights, embed_dim, qdim, kdim, vdim);

    ncnn::Mat q = RandomMat(qdim, src_seqlen, num_windows);
    ncnn::Mat k = self ? q : RandomMat(kdim, dst_seqlen, num_windows);
    ncnn::Mat v = self ? q : RandomMat(vdim, dst_seqlen, num_windows);
    ncnn::Mat mask = make_mask(mask_type, dst_seqlen, src_seqlen, num_heads);

    const int typeindex = ncnn::layer_to_index("MultiHeadAttention");

    // window result
    ncnn::ParamDict pd_w;
    make_pd(pd_w, embed_dim, num_heads, qdim, kdim, vdim, attn_mask, 1);

    std::vector<ncnn::Mat> bottoms_w(3);
    bottoms_w[0] = q;
    bottoms_w[1] = k;
    bottoms_w[2] = v;
    if (attn_mask)
        bottoms_w.push_back(mask);

    ncnn::Layer* op_w = cpu ? ncnn::create_layer_cpu(typeindex) : ncnn::create_layer_naive(typeindex);
    ncnn::Mat out_w;
    int ret = run_layer_mode(op_w, pd_w, weights, mode, bottoms_w, 1, out_w);
    delete op_w;
    if (ret == RUN_SKIPPED)
    {
        fprintf(stderr, "note: test_window_equivalence %s %s skipped, layer lacks %s storage on this cpu\n", tag, mode_name(mode), mode_name(mode));
        return 0;
    }
    if (ret != 0)
    {
        fprintf(stderr, "test_window_equivalence %s %s self=%d mask_type=%d num_windows=%d window forward ret=%d\n", tag, mode_name(mode), self, mask_type, num_windows, ret);
        return -1;
    }

    if (out_w.dims != 3 || out_w.w != qdim || out_w.h != src_seqlen || out_w.c != num_windows)
    {
        fprintf(stderr, "test_window_equivalence %s %s self=%d mask_type=%d num_windows=%d bad output shape dims=%d (%d %d %d), expect (%d %d %d)\n", tag, mode_name(mode), self, mask_type, num_windows, out_w.dims, out_w.w, out_w.h, out_w.c, qdim, src_seqlen, num_windows);
        return -1;
    }

    // plain 2-D reference, one window at a time, same layer kind and weights
    ncnn::ParamDict pd_p;
    make_pd(pd_p, embed_dim, num_heads, qdim, kdim, vdim, attn_mask, 0);

    for (int w = 0; w < num_windows; w++)
    {
        std::vector<ncnn::Mat> bottoms_p(3);
        bottoms_p[0] = q.channel(w).reshape(qdim, src_seqlen).clone();
        bottoms_p[1] = k.channel(w).reshape(kdim, dst_seqlen).clone();
        bottoms_p[2] = v.channel(w).reshape(vdim, dst_seqlen).clone();
        if (attn_mask)
            bottoms_p.push_back(mask);

        ncnn::Layer* op_p = cpu ? ncnn::create_layer_cpu(typeindex) : ncnn::create_layer_naive(typeindex);
        ncnn::Mat out_p;
        ret = run_layer_mode(op_p, pd_p, weights, mode, bottoms_p, 1, out_p);
        delete op_p;
        if (ret != 0)
        {
            fprintf(stderr, "test_window_equivalence %s %s self=%d mask_type=%d plain forward ret=%d\n", tag, mode_name(mode), self, mask_type, ret);
            return -1;
        }

        ncnn::Mat out_w_ch = out_w.channel(w).reshape(qdim, src_seqlen).clone();
        if (mode == MODE_FP32)
        {
            if (CompareMat(out_p, out_w_ch, 0.001f) != 0)
            {
                fprintf(stderr, "test_window_equivalence %s fp32 self=%d mask_type=%d num_windows=%d window %d differs from plain 2-D MHA\n", tag, self, mask_type, num_windows, w);
                return -1;
            }
        }
        else
        {
            const float diff = max_abs_diff(out_p, out_w_ch);
            if (!(diff <= 1e-5f))
            {
                fprintf(stderr, "test_window_equivalence %s %s self=%d mask_type=%d num_windows=%d window %d differs from plain 2-D MHA, max abs diff %g > 1e-5\n", tag, mode_name(mode), self, mask_type, num_windows, w, diff);
                return -1;
            }
        }
    }

    return 0;
}

static int test_window_equivalence()
{
    int ret = 0;

    // fp32, naive and cpu: masks none, 2-D, and 3-D per-head (plain 2-D MHA given the same 3-D mask)
    for (int cpu = 0; cpu < 2; cpu++)
    {
        for (int self = 0; self < 2; self++)
        {
            ret |= test_window_equivalence_one(cpu != 0, MODE_FP32, self != 0, 0, 3);
            ret |= test_window_equivalence_one(cpu != 0, MODE_FP32, self != 0, 2, 4);
            ret |= test_window_equivalence_one(cpu != 0, MODE_FP32, self != 0, 0, 8);
            ret |= test_window_equivalence_one(cpu != 0, MODE_FP32, self != 0, 3, 4);
        }
    }

    // bf16 storage and fp16 storage+arithmetic, cpu layer, packing on (ruling R11):
    // test_layer skips bf16 for MultiHeadAttention, so these are checked by equivalence only
    static const int windows[] = {3, 4, 8};
    for (int mode = MODE_BF16; mode <= MODE_FP16; mode++)
    {
#if !NCNN_BF16
        if (mode == MODE_BF16)
            continue;
#endif
        for (int i = 0; i < 3; i++)
        {
            for (int self = 0; self < 2; self++)
            {
                ret |= test_window_equivalence_one(true, mode, self != 0, 0, windows[i]);
                ret |= test_window_equivalence_one(true, mode, self != 0, 2, windows[i]);
            }
        }
    }

    return ret;
}

// ---- 3. negatives ----

static int expect_reject(const char* what, bool cpu, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const std::vector<ncnn::Mat>& bottoms, int top_count)
{
    const int typeindex = ncnn::layer_to_index("MultiHeadAttention");
    ncnn::Layer* op = cpu ? ncnn::create_layer_cpu(typeindex) : ncnn::create_layer_naive(typeindex);
    ncnn::Mat top;
    int ret = run_layer(op, pd, weights, bottoms, top_count, top);
    delete op;
    if (ret != -1)
    {
        fprintf(stderr, "test_window_negative %s %s: expect -1, got %d\n", cpu ? "cpu" : "naive", what, ret);
        return -1;
    }
    return 0;
}

static int test_window_negatives()
{
    const int embed_dim = 16;
    const int num_heads = 2;
    int ret = 0;

    std::vector<ncnn::Mat> weights;
    make_weights(weights, embed_dim, embed_dim, embed_dim, embed_dim);

    // kv_cache=1: bottoms q k v cache_k cache_v (empty caches), tops out cache_k cache_v
    {
        ncnn::ParamDict pd;
        make_pd(pd, embed_dim, num_heads, embed_dim, embed_dim, embed_dim, 0, 1);
        pd.set(7, 1);

        std::vector<ncnn::Mat> bottoms(5);
        bottoms[0] = RandomMat(embed_dim, 5, 3);
        bottoms[1] = RandomMat(embed_dim, 5, 3);
        bottoms[2] = RandomMat(embed_dim, 5, 3);

        ret |= expect_reject("kv_cache=1", false, pd, weights, bottoms, 3);
        ret |= expect_reject("kv_cache=1", true, pd, weights, bottoms, 3);
    }

    // q.c != k.c
    {
        ncnn::ParamDict pd;
        make_pd(pd, embed_dim, num_heads, embed_dim, embed_dim, embed_dim, 0, 1);

        std::vector<ncnn::Mat> bottoms(3);
        bottoms[0] = RandomMat(embed_dim, 5, 3);
        bottoms[1] = RandomMat(embed_dim, 7, 2);
        bottoms[2] = RandomMat(embed_dim, 7, 2);

        ret |= expect_reject("q.c!=k.c", false, pd, weights, bottoms, 1);
        ret |= expect_reject("q.c!=k.c", true, pd, weights, bottoms, 1);
    }

    // k.c != v.c
    {
        ncnn::ParamDict pd;
        make_pd(pd, embed_dim, num_heads, embed_dim, embed_dim, embed_dim, 0, 1);

        std::vector<ncnn::Mat> bottoms(3);
        bottoms[0] = RandomMat(embed_dim, 5, 3);
        bottoms[1] = RandomMat(embed_dim, 7, 3);
        bottoms[2] = RandomMat(embed_dim, 7, 4);

        ret |= expect_reject("k.c!=v.c", false, pd, weights, bottoms, 1);
        ret |= expect_reject("k.c!=v.c", true, pd, weights, bottoms, 1);
    }

    // k.h != v.h
    {
        ncnn::ParamDict pd;
        make_pd(pd, embed_dim, num_heads, embed_dim, embed_dim, embed_dim, 0, 1);

        std::vector<ncnn::Mat> bottoms(3);
        bottoms[0] = RandomMat(embed_dim, 5, 3);
        bottoms[1] = RandomMat(embed_dim, 7, 3);
        bottoms[2] = RandomMat(embed_dim, 6, 3);

        ret |= expect_reject("k.h!=v.h", false, pd, weights, bottoms, 1);
        ret |= expect_reject("k.h!=v.h", true, pd, weights, bottoms, 1);
    }

#if NCNN_INT8
    // int8_scale_term (18) != 0: naive layer only. The weights stay fp32 and the
    // int8 scale blobs are appended (q/k/v [embed_dim], out [1]); a cpu layer would
    // convert int8 weights in create_pipeline, so it is not exercised here.
    {
        ncnn::ParamDict pd;
        make_pd(pd, embed_dim, num_heads, embed_dim, embed_dim, embed_dim, 0, 1);
        pd.set(18, 2);

        std::vector<ncnn::Mat> weights_int8 = weights;
        weights_int8.push_back(RandomMat(embed_dim, 1.f, 10.f));
        weights_int8.push_back(RandomMat(embed_dim, 1.f, 10.f));
        weights_int8.push_back(RandomMat(embed_dim, 1.f, 10.f));
        weights_int8.push_back(RandomMat(1, 1.f, 10.f));
        // spare blobs so a loader that reads more entries stays in bounds
        for (int i = 0; i < 8; i++)
            weights_int8.push_back(RandomMat(embed_dim, 1.f, 10.f));

        std::vector<ncnn::Mat> bottoms(3);
        bottoms[0] = RandomMat(embed_dim, 5, 3);
        bottoms[1] = RandomMat(embed_dim, 5, 3);
        bottoms[2] = RandomMat(embed_dim, 5, 3);

        ret |= expect_reject("int8_scale_term=2", false, pd, weights_int8, bottoms, 1);
    }
#endif

    return ret;
}

int main()
{
    SRAND(7767517);

    int ret = 0;
    ret |= test_window_layer_matrix();
    ret |= test_window_equivalence();
    ret |= test_window_negatives();
    return ret ? -1 : 0;
}
