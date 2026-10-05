# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

# Contract test: ncnn2int8 never quantizes MultiHeadAttention.
# It prints "skip_quantize_multiheadattention <name>" to stderr, leaves
# int8_scale_term at 0 (no " 18=" written) and ModelWriter keeps window_batch1 (" 19=1").
#
# usage: cmake -DNCNN2INT8=<path to ncnn2int8> -DWORK_DIR=<scratch dir> -P test_ncnn2int8_mha_skip.cmake

if(NOT NCNN2INT8)
    message(FATAL_ERROR "NCNN2INT8 not set")
endif()
if(NOT WORK_DIR)
    message(FATAL_ERROR "WORK_DIR not set")
endif()

file(REMOVE_RECURSE "${WORK_DIR}")
file(MAKE_DIRECTORY "${WORK_DIR}")

set(IN_PARAM "${WORK_DIR}/mha_window.param")
set(OUT_PARAM "${WORK_DIR}/mha_window-int8.param")
set(OUT_BIN "${WORK_DIR}/mha_window-int8.bin")

# Input (16, 5, 3) -> MultiHeadAttention self-attention, embed_dim=16 num_heads=2 window_batch1=1
file(WRITE "${IN_PARAM}" "7767517
2 2
Input                    in0                      0 1 in0 0=16 1=5 2=3
MultiHeadAttention       mha0                     1 1 in0 out0 0=16 1=2 2=256 3=16 4=16 19=1
")

# no calibration table: the tool accepts 4 arguments
execute_process(
    COMMAND $ENV{TESTS_EXECUTABLE_LOADER} $ENV{TESTS_EXECUTABLE_LOADER_ARGUMENTS} "${NCNN2INT8}" "${IN_PARAM}" null "${OUT_PARAM}" "${OUT_BIN}"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE out
    ERROR_VARIABLE err)

message(STATUS "ncnn2int8 exit: ${result}")
message(STATUS "ncnn2int8 stdout:\n${out}")
message(STATUS "ncnn2int8 stderr:\n${err}")

set(failures "")

if(NOT "${result}" STREQUAL "0")
    list(APPEND failures "exit code ${result}, expected 0")
endif()

if(NOT err MATCHES "skip_quantize_multiheadattention mha0")
    list(APPEND failures "stderr lacks 'skip_quantize_multiheadattention mha0'")
endif()

if(NOT EXISTS "${OUT_PARAM}")
    list(APPEND failures "output param ${OUT_PARAM} not written")
else()
    file(READ "${OUT_PARAM}" out_param)
    message(STATUS "output param:\n${out_param}")

    file(STRINGS "${OUT_PARAM}" mha_lines REGEX "^MultiHeadAttention[ \t]")
    list(LENGTH mha_lines mha_count)
    if(NOT mha_count EQUAL 1)
        list(APPEND failures "expected 1 MultiHeadAttention line in output param, found ${mha_count}")
    else()
        if(mha_lines MATCHES " 18=")
            list(APPEND failures "MultiHeadAttention line has a 18= (int8_scale_term) entry: ${mha_lines}")
        endif()
        if(NOT mha_lines MATCHES " 19=1( |$)")
            list(APPEND failures "MultiHeadAttention line lacks 19=1 (window_batch1): ${mha_lines}")
        endif()
    endif()
endif()

if(failures)
    foreach(f ${failures})
        message(SEND_ERROR "FAIL: ${f}")
    endforeach()
    message(FATAL_ERROR "test_ncnn2int8_mha_skip failed")
endif()

message(STATUS "test_ncnn2int8_mha_skip passed")
