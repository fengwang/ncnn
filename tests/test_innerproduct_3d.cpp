// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

// Contract test: InnerProduct with a 3-D input whose w == num_input computes
// row-wise, giving an output of shape (num_output, h, c).

#include "testutil.h"

#include "layer.h"
#include "modelbin.h"

#include <vector>

static int check_shape(bool cpu, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Mat& a, int num_output)
{
    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_packing_layout = false;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_bf16_storage = false;
    opt.use_vulkan_compute = false;

    const int typeindex = ncnn::layer_to_index("InnerProduct");
    ncnn::Layer* op = cpu ? ncnn::create_layer_cpu(typeindex) : ncnn::create_layer_naive(typeindex);

    op->load_param(pd);
    ncnn::ModelBinFromMatArray mb(weights.data());
    op->load_model(mb);

    int ret = op->create_pipeline(opt);
    ncnn::Mat b;
    if (ret == 0)
        ret = op->forward(a, b, opt);
    op->destroy_pipeline(opt);
    delete op;

    if (ret != 0)
    {
        fprintf(stderr, "test_innerproduct_3d %s forward ret=%d\n", cpu ? "cpu" : "naive", ret);
        return -1;
    }

    if (b.dims != 3 || b.w != num_output || b.h != a.h || b.c != a.c || b.elempack != 1)
    {
        fprintf(stderr, "test_innerproduct_3d %s bad output shape dims=%d (%d %d %d) elempack=%d, expect (%d %d %d)\n", cpu ? "cpu" : "naive", b.dims, b.w, b.h, b.c, b.elempack, num_output, a.h, a.c);
        return -1;
    }

    return 0;
}

static int test_innerproduct_3d(int num_input, int h, int c, int num_output, int bias, int activation_type)
{
    ncnn::ParamDict pd;
    pd.set(0, num_output);
    pd.set(1, bias);
    pd.set(2, num_output * num_input);
    pd.set(9, activation_type);
    if (activation_type == 2)
    {
        ncnn::Mat activation_params(1);
        activation_params[0] = 0.1f; // leaky relu slope
        pd.set(10, activation_params);
    }

    std::vector<ncnn::Mat> weights(bias ? 2 : 1);
    weights[0] = RandomMat(num_output * num_input);
    if (bias)
        weights[1] = RandomMat(num_output);

    ncnn::Mat a = RandomMat(num_input, h, c);

    int ret = 0;
    ret |= check_shape(false, pd, weights, a, num_output);
    ret |= check_shape(true, pd, weights, a, num_output);
    ret |= test_layer("InnerProduct", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_innerproduct_3d failed a=(%d %d %d) num_output=%d bias=%d activation_type=%d\n", num_input, h, c, num_output, bias, activation_type);
    }
    return ret;
}

int main()
{
    SRAND(7767517);

    static const int channels[] = {1, 3, 4, 8};
    int ret = 0;
    for (int i = 0; i < 4; i++)
    {
        const int c = channels[i];
        ret |= test_innerproduct_3d(12, 5, c, 7, 0, 0);
        ret |= test_innerproduct_3d(12, 5, c, 16, 1, 0);
        ret |= test_innerproduct_3d(16, 3, c, 8, 1, 2);
        ret |= test_innerproduct_3d(19, 4, c, 12, 0, 2);
    }
    return ret ? -1 : 0;
}
