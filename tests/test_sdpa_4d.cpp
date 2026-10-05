// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

// Contract test: SDPA with a 4-D query (w=embed, h=src_seqlen, d=num_heads, c=batch)
// and 4-D key/value (d=num_group, GQA when num_group < num_heads) computes per batch
// item; output is (out_embed_dim, src_seqlen, num_heads, batch).
// The mask, when present, is 2-D (dst_seqlen, src_seqlen) and shared by all heads
// and batch items (the contract does not name a 4-D mask shape).

#include "testutil.h"

#include "layer.h"

#include <vector>

static int check_shape(bool cpu, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& as, int out_embed_dim)
{
    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_packing_layout = false;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_bf16_storage = false;
    opt.use_vulkan_compute = false;

    const int typeindex = ncnn::layer_to_index("SDPA");
    ncnn::Layer* op = cpu ? ncnn::create_layer_cpu(typeindex) : ncnn::create_layer_naive(typeindex);

    op->load_param(pd);

    int ret = op->create_pipeline(opt);
    std::vector<ncnn::Mat> tops(1);
    if (ret == 0)
        ret = op->forward(as, tops, opt);
    op->destroy_pipeline(opt);
    delete op;

    if (ret != 0)
    {
        fprintf(stderr, "test_sdpa_4d %s forward ret=%d\n", cpu ? "cpu" : "naive", ret);
        return -1;
    }

    const ncnn::Mat& q = as[0];
    const ncnn::Mat& b = tops[0];
    if (b.dims != 4 || b.w != out_embed_dim || b.h != q.h || b.d != q.d || b.c != q.c || b.elempack != 1)
    {
        fprintf(stderr, "test_sdpa_4d %s bad output shape dims=%d (%d %d %d %d) elempack=%d, expect (%d %d %d %d)\n", cpu ? "cpu" : "naive", b.dims, b.w, b.h, b.d, b.c, b.elempack, out_embed_dim, q.h, q.d, q.c);
        return -1;
    }

    return 0;
}

static int test_sdpa_4d(int embed_dim, int out_embed_dim, int src_seqlen, int dst_seqlen, int num_heads, int num_group, int batch, int attn_mask)
{
    ncnn::ParamDict pd;
    pd.set(5, attn_mask);
    pd.set(6, 0.f);

    std::vector<ncnn::Mat> weights(0);

    std::vector<ncnn::Mat> as(3);
    as[0] = RandomMat(embed_dim, src_seqlen, num_heads, batch);
    as[1] = RandomMat(embed_dim, dst_seqlen, num_group, batch);
    as[2] = RandomMat(out_embed_dim, dst_seqlen, num_group, batch);
    if (attn_mask)
        as.push_back(RandomMat(dst_seqlen, src_seqlen));

    int ret = 0;
    ret |= check_shape(false, pd, as, out_embed_dim);
    ret |= check_shape(true, pd, as, out_embed_dim);
    ret |= test_layer("SDPA", pd, weights, as, 1, 0.001f);
    if (ret != 0)
    {
        fprintf(stderr, "test_sdpa_4d failed embed_dim=%d out_embed_dim=%d src_seqlen=%d dst_seqlen=%d num_heads=%d num_group=%d batch=%d attn_mask=%d\n", embed_dim, out_embed_dim, src_seqlen, dst_seqlen, num_heads, num_group, batch, attn_mask);
    }
    return ret;
}

int main()
{
    SRAND(7767517);

    int ret = 0;
    for (int batch = 1; batch <= 3; batch++)
    {
        for (int attn_mask = 0; attn_mask < 2; attn_mask++)
        {
            // num_group == num_heads
            ret |= test_sdpa_4d(16, 16, 7, 7, 4, 4, batch, attn_mask);
            ret |= test_sdpa_4d(24, 12, 5, 9, 3, 3, batch, attn_mask);
            // GQA: num_group < num_heads
            ret |= test_sdpa_4d(16, 20, 6, 11, 4, 2, batch, attn_mask);
            ret |= test_sdpa_4d(8, 8, 9, 9, 6, 1, batch, attn_mask);
        }
    }
    return ret ? -1 : 0;
}
