/*
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 i*/
#include <catch2/catch_test_macros.hpp>
#include "../utils/helpers.h"

#include <cudnn_frontend.h>

TEST_CASE("MX Quant Test New", "[layernorm][graph][block_scale]") {
    namespace fe = cudnn_frontend;

#if (CUDNN_VERSION < 90700)
    SKIP("MXFP8 is not supported in cudnn versions prior to 9.7.0");
#endif

    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::HALF)
         .set_intermediate_data_type(fe::DataType_t::FLOAT)
         .set_compute_data_type(fe::DataType_t::FLOAT);

    auto m  = 1;
    auto n = 32;
    auto block_size  = 32;

    // dummy config for layernorm 这部分不重要
    auto X = graph.tensor(fe::graph::Tensor_attributes()
                          .set_name("X")
                          .set_dim({1, m, n, 1})
                          .set_stride({m * n, n, 1, n}));

    auto scale = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("scale")
                              .set_dim({1, 1, n, 1})
                              .set_stride({n, n, 1, n})
                              .set_data_type(fe::DataType_t::FLOAT));

    auto bias = graph.tensor(fe::graph::Tensor_attributes()
                             .set_name("bias")
                             .set_dim({1, 1, n, 1})
                             .set_stride({n, n, 1, n})
                             .set_data_type(fe::DataType_t::FLOAT));

    float epsilon_cpu = 1e-05f;
    auto epsilon = graph.tensor(epsilon_cpu);

    auto layernorm_options = fe::graph::Layernorm_attributes()
                                 .set_forward_phase(fe::NormFwdPhase_t::TRAINING)
                                 .set_epsilon(epsilon);

    auto [Y_ln, mean, inv_variance] = graph.layernorm(X, scale, bias, layernorm_options);
    mean->set_output(true).set_data_type(fe::DataType_t::FLOAT);
    inv_variance->set_output(true).set_data_type(fe::DataType_t::FLOAT);
    // dummy config for layernorm done

    // MX start 直接从X开始拿数据，忽略LN结果
    auto Y_ln_2d = graph.reshape(X, fe::graph::Reshape_attributes());
    Y_ln_2d->set_dim({m, n});
    Y_ln_2d->set_output(true).set_data_type(fe::DataType_t::FLOAT);  // reshape后的输出

    auto mxfp8_quantize_row_opts = fe::graph::Block_scale_quantize_attributes()
                                       .set_block_size(block_size)
                                       .set_axis(1)
                                       .set_transpose(false);
    auto [Y_row, mx_row] = graph.block_scale_quantize(Y_ln_2d, mxfp8_quantize_row_opts);
    Y_row->set_output(true).set_data_type(fe::DataType_t::FP8_E4M3);
    mx_row->set_output(true).set_data_type(fe::DataType_t::FP8_E8M0);

    if (check_device_arch_newer_than("blackwell") == false) {
        SKIP("MXFP8 requires Blackwell and up");
    }

    auto handle_ptr = create_cudnn_handle();
    auto handle = *handle_ptr;

    REQUIRE(graph.validate().is_good());
    REQUIRE(graph.build_operation_graph(handle).is_good());
    REQUIRE(graph.create_execution_plans({fe::HeurMode_t::FALLBACK}).is_good());
    REQUIRE(graph.check_support(handle).is_good());
    REQUIRE(graph.build_plans(handle).is_good());

    // Allocate tensors
    Surface<half> X_tensor(m * n, false);
    Surface<float> Scale_tensor(n, false);
    Surface<float> Bias_tensor(n, false);
    Surface<float> Mean_tensor(m, false);
    Surface<float> Var_tensor(m, false);
    Surface<float> Y_ln_2d_tensor(m * n, false);

    Surface<int8_t> Y_row_tensor(m * n, false);
    Surface<int8_t> mx_row_tensor(m * n / block_size, false);

    Surface<int8_t> workspace(graph.get_workspace_size(), false);
    
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {X, X_tensor.devPtr},
        {scale, Scale_tensor.devPtr},
        {bias, Bias_tensor.devPtr},
        {mean, Mean_tensor.devPtr},
        {inv_variance, Var_tensor.devPtr},
        {Y_ln_2d, Y_ln_2d_tensor.devPtr},
        {Y_row, Y_row_tensor.devPtr},
        {mx_row, mx_row_tensor.devPtr}
    };

    REQUIRE(graph.execute(handle, variant_pack, workspace.devPtr).is_good());

    // ===== Debug: 打印前 16 项 =====
    int total_elems = m * n;
    int row_blocks = total_elems / block_size;
    std::vector<float> host_Y_ln_2d(total_elems);
    std::vector<int8_t> host_Y_row(total_elems);
    std::vector<int8_t> host_mx_row(row_blocks);

    cudaMemcpy(host_Y_ln_2d.data(), Y_ln_2d_tensor.devPtr, sizeof(float) * total_elems, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_Y_row.data(),   Y_row_tensor.devPtr,   sizeof(int8_t) * total_elems, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_mx_row.data(),  mx_row_tensor.devPtr,  sizeof(int8_t) * row_blocks,  cudaMemcpyDeviceToHost);
    
    //     print.....
}

TEST_CASE("RmsNorm Inference NVFP4 print new", "[rmsnorm][graph][block_scale]") {
    namespace fe = cudnn_frontend;

#if (CUDNN_VERSION < 90700)  // TODO: v9.99 is new feature branch; switch to release branch when ready
    SKIP("NVFP4 is not supported in cudnn versions prior to 9.7.0");
#endif

    fe::graph::Graph graph;
    graph.set_intermediate_data_type(fe::DataType_t::FLOAT).set_compute_data_type(fe::DataType_t::FLOAT);

    auto batch_size  = 1;
    auto seq_length  = 32;
    auto hidden_size = 32;
    auto block_size  = 16;

    auto X     = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("X")
                              .set_data_type(fe::DataType_t::FLOAT)
                              .set_dim({batch_size, seq_length, hidden_size, 1})
                              .set_stride({seq_length * hidden_size, hidden_size, 1, hidden_size}));
    auto scale = graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("scale")
                                  .set_dim({1, 1, hidden_size, 1})
                                  .set_stride({hidden_size, hidden_size, 1, hidden_size})
                                  .set_data_type(fe::DataType_t::FLOAT));
    auto bias  = graph.tensor(fe::graph::Tensor_attributes()
                                 .set_name("bias")
                                 .set_dim({1, 1, hidden_size, 1})
                                 .set_stride({hidden_size, hidden_size, 1, hidden_size})
                                 .set_data_type(fe::DataType_t::FLOAT));

    float epsilon_cpu = 1e-05f;
    auto epsilon      = graph.tensor(epsilon_cpu);

    auto rmsnorm_options = fe::graph::Rmsnorm_attributes()
                               .set_forward_phase(fe::NormFwdPhase_t::INFERENCE)
                               .set_bias(bias)
                               .set_epsilon(epsilon);
    auto [Y_ln, inv_variance] = graph.rmsnorm(X, scale, rmsnorm_options);
    REQUIRE(inv_variance == nullptr);

    auto X_2d = graph.reshape(X, fe::graph::Reshape_attributes());
    X_2d->set_dim({batch_size * seq_length, hidden_size});
    X_2d->set_output(true).set_data_type(fe::DataType_t::FLOAT);

    auto nvfp4_quantize_opts =
        fe::graph::Block_scale_quantize_attributes().set_block_size(block_size).set_axis(1).set_transpose(false);
    auto [Y, mx_scale] = graph.block_scale_quantize(X_2d, nvfp4_quantize_opts);
    Y->set_output(true).set_data_type(fe::DataType_t::FP4_E2M1);
    mx_scale->set_output(true).set_data_type(fe::DataType_t::FP8_E4M3);

    if (check_device_arch_newer_than("blackwell") == false) {
        SKIP("NVFP4 requires Blackwell and up");
    }
    // Create a unique_ptr for the cuDNN handle
    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    REQUIRE(graph.validate().is_good());

    REQUIRE(graph.build_operation_graph(handle).is_good());

    REQUIRE(graph.create_execution_plans({fe::HeurMode_t::FALLBACK}).is_good());

    REQUIRE(graph.check_support(handle).is_good());

    REQUIRE(graph.build_plans(handle).is_good());

    Surface<float> X_tensor(batch_size * seq_length * hidden_size, false);
    Surface<float> X_2d_tensor(batch_size * seq_length * hidden_size, false);
    Surface<float> Scale_tensor(hidden_size, false);
    Surface<float> Bias_tensor(hidden_size, false);
    Surface<int8_t> Y_tensor(batch_size * seq_length * hidden_size / 2, false);
    Surface<int8_t> mx_scale_tensor(batch_size * seq_length * hidden_size / block_size, false);

    Surface<int8_t> workspace(graph.get_workspace_size(), false);
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {X, X_tensor.devPtr},
        {X_2d, X_2d_tensor.devPtr},
        {scale, Scale_tensor.devPtr},
        {bias, Bias_tensor.devPtr},
        {Y, Y_tensor.devPtr},
        {mx_scale, mx_scale_tensor.devPtr}};

    REQUIRE(graph.execute(handle, variant_pack, workspace.devPtr).is_good());

//    int print_count = 16;
    int total_elems = batch_size * seq_length * hidden_size;
    int row_blocks = total_elems / block_size;

    std::vector<float> host_Y_ln(total_elems);
    std::vector<int8_t> host_Y_row(batch_size * seq_length * hidden_size / 2);
    std::vector<int8_t> host_mx_row(row_blocks);

    printf("before:\n");
    for (size_t i=0;i<host_Y_ln.size();i++) printf("%f ", host_Y_ln[i]);
    printf("\n");
    for (size_t i=0;i<host_Y_row.size();i++) printf("%x ", host_Y_row[i]);
    printf("\n");
    for (size_t i=0;i<host_mx_row.size();i++) printf("%x ", host_mx_row[i]);

    cudaMemcpy(host_Y_ln.data(), X_2d_tensor.devPtr, sizeof(float) * total_elems, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_Y_row.data(),   Y_tensor.devPtr,   sizeof(int8_t) * batch_size * seq_length * hidden_size / 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_mx_row.data(),  mx_scale_tensor.devPtr,  sizeof(uint8_t) * row_blocks,  cudaMemcpyDeviceToHost);

    printf("\n\n\n\n\nafter:\n");
    for (size_t i=0;i<host_Y_ln.size();i++) printf("%f ", host_Y_ln[i]);
    printf("\n");
    for (size_t i=0;i<host_Y_row.size();i++) printf("%x ", host_Y_row[i]);
    printf("\n");
    for (size_t i=0;i<host_mx_row.size();i++) printf("%x ", host_mx_row[i]);
//    printf("\n===== Block-Scale Quantize Debug (Row vs Col) =====\n");
/*
    for (int block = 0; block < print_count; ++block) {
        printf("Block[%3d]:\n", block);
    
        printf("y_ln_2d: ");
        for (int i = 0; i < block_size; ++i) {
            int idx = block * block_size + i; 
            printf("%f ", host_Y_ln[idx]);
        }
        printf("\n");
    
        printf("y_row:  ");
        for (int i = 0; i < block_size / 2; ++i) {
            int idx = block * block_size / 2 + i; 
            printf("  0x%x  ", (host_Y_row[idx]));
        }
        printf("\n");
        printf("mx_rol:   0x%x\n", host_mx_row[block]);
    }

    printf("===================================================\n");
*/
}

TEST_CASE("RmsNorm Inference NVFP4", "[rmsnorm][graph][block_scale]") {
    namespace fe = cudnn_frontend;

#if (CUDNN_VERSION < 90700)  // TODO: v9.99 is new feature branch; switch to release branch when ready
    SKIP("NVFP4 is not supported in cudnn versions prior to 9.7.0");
#endif

    fe::graph::Graph graph;
    graph.set_intermediate_data_type(fe::DataType_t::FLOAT).set_compute_data_type(fe::DataType_t::FLOAT);

    auto batch_size  = 4;
    auto seq_length  = 1024;
    auto hidden_size = 128;
    auto block_size  = 16;

    auto X     = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("X")
                              .set_data_type(fe::DataType_t::FLOAT)
                              .set_dim({batch_size, seq_length, hidden_size, 1})
                              .set_stride({seq_length * hidden_size, hidden_size, 1, hidden_size}));
    auto scale = graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("scale")
                                  .set_dim({1, 1, hidden_size, 1})
                                  .set_stride({hidden_size, hidden_size, 1, hidden_size})
                                  .set_data_type(fe::DataType_t::FLOAT));
    auto bias  = graph.tensor(fe::graph::Tensor_attributes()
                                 .set_name("bias")
                                 .set_dim({1, 1, hidden_size, 1})
                                 .set_stride({hidden_size, hidden_size, 1, hidden_size})
                                 .set_data_type(fe::DataType_t::FLOAT));

    float epsilon_cpu = 1e-05f;
    auto epsilon      = graph.tensor(epsilon_cpu);

    auto rmsnorm_options = fe::graph::Rmsnorm_attributes()
                               .set_forward_phase(fe::NormFwdPhase_t::INFERENCE)
                               .set_bias(bias)
                               .set_epsilon(epsilon);
    auto [Y_ln, inv_variance] = graph.rmsnorm(X, scale, rmsnorm_options);
    REQUIRE(inv_variance == nullptr);

    auto nvfp4_quantize_opts =
        fe::graph::Block_scale_quantize_attributes().set_block_size(block_size).set_axis(2).set_transpose(false);
    auto [Y, mx_scale] = graph.block_scale_quantize(Y_ln, nvfp4_quantize_opts);
    Y->set_output(true).set_data_type(fe::DataType_t::FP4_E2M1);
    mx_scale->set_output(true).set_data_type(fe::DataType_t::FP8_E4M3);

    if (check_device_arch_newer_than("blackwell") == false) {
        SKIP("NVFP4 requires Blackwell and up");
    }
    // Create a unique_ptr for the cuDNN handle
    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    REQUIRE(graph.validate().is_good());

    REQUIRE(graph.build_operation_graph(handle).is_good());

    REQUIRE(graph.create_execution_plans({fe::HeurMode_t::FALLBACK}).is_good());

    REQUIRE(graph.check_support(handle).is_good());

    REQUIRE(graph.build_plans(handle).is_good());

    Surface<float> X_tensor(batch_size * seq_length * hidden_size, false);
    Surface<float> Scale_tensor(hidden_size, false);
    Surface<float> Bias_tensor(hidden_size, false);
    Surface<int8_t> Y_tensor(batch_size * seq_length * hidden_size / 2, false);
    Surface<int8_t> mx_scale_tensor(batch_size * seq_length * hidden_size / block_size, false);

    Surface<int8_t> workspace(graph.get_workspace_size(), false);
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {X, X_tensor.devPtr},
        {scale, Scale_tensor.devPtr},
        {bias, Bias_tensor.devPtr},
        {Y, Y_tensor.devPtr},
        {mx_scale, mx_scale_tensor.devPtr}};

    REQUIRE(graph.execute(handle, variant_pack, workspace.devPtr).is_good());
}
