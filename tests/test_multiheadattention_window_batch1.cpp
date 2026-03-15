// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#include "layer/multiheadattention.h"
#include "layer_type.h"
#include "modelbin.h"

#include <float.h>
#include <math.h>
#include <vector>

static ncnn::Mat make_identity_weight(int dim)
{
    ncnn::Mat weight(dim * dim);
    weight.fill(0.f);
    float* ptr = weight;
    for (int out = 0; out < dim; out++)
    {
        ptr[out * dim + out] = 1.f;
    }
    return weight;
}

static ncnn::Mat make_zero_bias(int dim)
{
    ncnn::Mat bias(dim);
    bias.fill(0.f);
    return bias;
}

static ncnn::Mat make_window_batch1_input(int embed_dim, int seq_len, int num_windows)
{
    ncnn::Mat input(embed_dim, seq_len, num_windows);

    for (int w = 0; w < num_windows; w++)
    {
        ncnn::Mat input_window = input.channel(w);
        for (int i = 0; i < seq_len; i++)
        {
            float* rowptr = input_window.row(i);
            for (int j = 0; j < embed_dim; j++)
            {
                rowptr[j] = (float)((w + 1) * 100 + i * 10 + j + 1) / 100.f;
            }
        }
    }

    return input;
}

static ncnn::Mat make_attention_mask(int seq_len, int num_heads)
{
    ncnn::Mat mask(seq_len, seq_len, num_heads);
    mask.fill(0.f);

    for (int h = 0; h < num_heads; h++)
    {
        ncnn::Mat mask_head = mask.channel(h);
        for (int i = 0; i < seq_len; i++)
        {
            float* rowptr = mask_head.row(i);
            for (int j = 0; j < seq_len; j++)
            {
                if (j > i)
                    rowptr[j] = -0.25f * (float)(h + 1);
            }
        }
    }

    return mask;
}

static ncnn::Mat reference_window_batch1_attention(const ncnn::Mat& input, const ncnn::Mat& attn_mask, int num_heads, float scale)
{
    const int embed_dim = input.w;
    const int seq_len = input.h;
    const int num_windows = input.c;
    const int head_dim = embed_dim / num_heads;

    ncnn::Mat output(embed_dim, seq_len, num_windows);

    for (int w = 0; w < num_windows; w++)
    {
        const ncnn::Mat inw = input.channel(w);
        ncnn::Mat outw = output.channel(w);

        for (int token = 0; token < seq_len; token++)
        {
            float* outptr = outw.row(token);
            for (int j = 0; j < embed_dim; j++)
                outptr[j] = 0.f;
        }

        for (int head = 0; head < num_heads; head++)
        {
            const ncnn::Mat maskh = attn_mask.channel(head);

            for (int i = 0; i < seq_len; i++)
            {
                float logits[16];
                float max_logit = -FLT_MAX;

                for (int j = 0; j < seq_len; j++)
                {
                    float sum = 0.f;
                    const float* qptr = inw.row(i) + head * head_dim;
                    const float* kptr = inw.row(j) + head * head_dim;

                    for (int k = 0; k < head_dim; k++)
                    {
                        sum += qptr[k] * kptr[k];
                    }

                    logits[j] = sum * scale + maskh.row(i)[j];
                    max_logit = std::max(max_logit, logits[j]);
                }

                float denom = 0.f;
                for (int j = 0; j < seq_len; j++)
                {
                    logits[j] = expf(logits[j] - max_logit);
                    denom += logits[j];
                }

                float* outptr = outw.row(i) + head * head_dim;
                for (int k = 0; k < head_dim; k++)
                {
                    outptr[k] = 0.f;
                }

                for (int j = 0; j < seq_len; j++)
                {
                    const float weight = logits[j] / denom;
                    const float* vptr = inw.row(j) + head * head_dim;
                    for (int k = 0; k < head_dim; k++)
                    {
                        outptr[k] += weight * vptr[k];
                    }
                }
            }
        }
    }

    return output;
}

static int run_multiheadattention_layer(ncnn::Layer* op, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const std::vector<ncnn::Mat>& bottoms, ncnn::Mat& top_blob)
{
    op->load_param(pd);
    ncnn::ModelBinFromMatArray mb(weights.data());
    op->load_model(mb);

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_packing_layout = false;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_bf16_storage = false;

    int ret = op->create_pipeline(opt);
    if (ret != 0)
        return ret;

    std::vector<ncnn::Mat> tops(1);
    ret = op->forward(bottoms, tops, opt);
    if (ret == 0)
        top_blob = tops[0];

    op->destroy_pipeline(opt);
    return ret;
}

static int test_multiheadattention_window_batch1_reference()
{
    const int embed_dim = 4;
    const int seq_len = 4;
    const int num_windows = 3;
    const int num_heads = 2;
    const float scale = 1.f / sqrtf((float)(embed_dim / num_heads));

    ncnn::ParamDict pd;
    pd.set(0, embed_dim);
    pd.set(1, num_heads);
    pd.set(2, embed_dim * embed_dim);
    pd.set(3, embed_dim);
    pd.set(4, embed_dim);
    pd.set(5, 1);
    pd.set(6, scale);
    pd.set(19, 1);

    std::vector<ncnn::Mat> weights(8);
    weights[0] = make_identity_weight(embed_dim);
    weights[1] = make_zero_bias(embed_dim);
    weights[2] = make_identity_weight(embed_dim);
    weights[3] = make_zero_bias(embed_dim);
    weights[4] = make_identity_weight(embed_dim);
    weights[5] = make_zero_bias(embed_dim);
    weights[6] = make_identity_weight(embed_dim);
    weights[7] = make_zero_bias(embed_dim);

    ncnn::Mat input = make_window_batch1_input(embed_dim, seq_len, num_windows);
    ncnn::Mat mask = make_attention_mask(seq_len, num_heads);
    ncnn::Mat expected = reference_window_batch1_attention(input, mask, num_heads, scale);

    std::vector<ncnn::Mat> bottoms(4);
    bottoms[0] = input;
    bottoms[1] = input;
    bottoms[2] = input;
    bottoms[3] = mask;

    ncnn::MultiHeadAttention generic_op;
    ncnn::Mat generic_output;
    int ret = run_multiheadattention_layer(&generic_op, pd, weights, bottoms, generic_output);
    if (ret != 0 || CompareMat(expected, generic_output, 0.001f) != 0)
    {
        fprintf(stderr, "generic window_batch1 MultiHeadAttention mismatch ret=%d\n", ret);
        return -1;
    }

    ncnn::Layer* cpu_op = ncnn::create_layer_cpu(ncnn::LayerType::MultiHeadAttention);
    ncnn::Mat cpu_output;
    ret = run_multiheadattention_layer(cpu_op, pd, weights, bottoms, cpu_output);
    delete cpu_op;
    if (ret != 0 || CompareMat(expected, cpu_output, 0.001f) != 0)
    {
        fprintf(stderr, "cpu window_batch1 MultiHeadAttention mismatch ret=%d\n", ret);
        return -1;
    }

    return 0;
}

int main()
{
    SRAND(7767517);

    return 0
           || test_multiheadattention_window_batch1_reference();
}
