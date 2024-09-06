/***************************************************************************************************
 * Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
    \brief Example of running an Ada FP8 GEMM.

    D = alpha * accumulator + beta * source
    
*/

#include <iostream>
#include <fstream>
#include <sstream>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/gemm_complex.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/host/gemm.h"

#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/thread/linear_combination_generic_with_scaling.h"
#include "cutlass/gemm/device/gemm_universal.h"

#include "cutlass/layout/matrix.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/layout/permute.h"

#include "device/dual_gemm.h"
#include "thread/left_silu_and_mul.h"

// Command line options parsing
struct Options {

  bool help;
  bool error;
  bool reference_check;
  cutlass::gemm::GemmCoord problem_size;

  int iterations;
  int warmup_iterations;

  float alpha0;
  float beta0;
  float alpha1;
  float beta1;

  int batch;

  Options():
    help(false),
    error(false),
    reference_check(false),
    iterations(20),
    warmup_iterations(5),
    alpha0(1.f),
    beta0(0.f),
    alpha1(1.f),
    beta1(0.f),
    batch(1)
  { }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("iterations", iterations, 20);
    cmd.get_cmd_line_argument("warmup_iterations", warmup_iterations, 5);
    cmd.get_cmd_line_argument("reference-check", reference_check, false);
    cmd.get_cmd_line_argument("alpha0", alpha0, 1.f);
    cmd.get_cmd_line_argument("beta0", beta0, 0.f);
    cmd.get_cmd_line_argument("alpha0", alpha1, 1.f);
    cmd.get_cmd_line_argument("beta0", beta1, 0.f);
    cmd.get_cmd_line_argument("batch", batch, 1);

    int m, n, k;
    cmd.get_cmd_line_argument("m", m, 1280);
    cmd.get_cmd_line_argument("n", n, 4096);
    cmd.get_cmd_line_argument("k", k, 4096);

    problem_size = cutlass::gemm::GemmCoord{m, n, k};
    
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "58_ada_fp8_gemm\n\n"
      << "  This example executes a GEMM using Ada FP8 Tensor Core operations. In addition to performing\n"
      << "  a normal GEMM, the kernel performs the following operations:\n"
      << "      D = alpha  * accumulator + beta * source \n"
      << "Options:\n\n"
      << "  --help                           If specified, displays this usage statement\n\n"
      << "  --m=<int>                        Sets the M dimension of the GEMM\n"
      << "  --n=<int>                        Sets the N dimension of the GEMM\n"
      << "  --k=<int>                        Sets the K dimension of the GEMM\n"
      << "  --iterations=<int>               Number of profiling iterations to perform\n"
      << "  --warmup-iterations=<int>        Number of warmup iterations to perform\n"
      << "  --reference-check=<bool>         If true, performs reference check\n";

    return out;
  }

  /// Compute performance in GFLOP/s
  float gflops(float runtime_s) const {
    // Two flops per multiply-add
    return 2.0f * float(problem_size.product()) / float(1.0e9) / runtime_s;
  }
};


using ElementA = cutlass::float_e4m3_t;
using ElementB = cutlass::float_e4m3_t;
using ElementC = cutlass::half_t;
using ElementOutput = cutlass::float_e4m3_t;
using ElementAccumulator = float;
using ElementCompute = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
static int const kStages = 3;
static int const kAlignmentA = 16;
static int const kAlignmentB = 16;

using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
    ElementC, 
    128 / cutlass::sizeof_bits<ElementC>::value,
    ElementAccumulator,
    ElementCompute,
    cutlass::epilogue::thread::ScaleType::NoBetaScaling>;

using EpilogueOutputOp2 = cutlass::epilogue::thread::LeftSiLUAndMul<
  ElementOutput,
  128 / cutlass::sizeof_bits<ElementC>::value,
  ElementC,
  ElementCompute
>;

constexpr bool kStoreD0 = true;
constexpr bool kStoreD1 = true;
constexpr bool kSplitKSerial = true;

template <typename MathOperator>
using DualGemm = cutlass::gemm::device::DualGemm<
    ElementA, LayoutA, 
    ElementB, LayoutB, LayoutB,
    ElementC, LayoutC,
    ElementOutput,
    ElementAccumulator, 
    cutlass::arch::OpClassTensorOp, 
    cutlass::arch::Sm89,
    cutlass::gemm::GemmShape<32, 128, 64>, 
    cutlass::gemm::GemmShape<32, 32, 64>, 
    cutlass::gemm::GemmShape<16, 8, 32>,
    EpilogueOutputOp, 
    EpilogueOutputOp, 
    EpilogueOutputOp2, 
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 
    kStages,
    kStoreD0,
    kStoreD1,
    kSplitKSerial,
    kAlignmentA, 
    kAlignmentB, 
    MathOperator
  >;

/// Helper class to run the kernel
template <typename Gemm>
struct TestbedRunner {

  using ElementAccumulator = typename Gemm::ElementAccumulator;
  using ElementCompute = ElementAccumulator;

  /// Initialization
  cutlass::Distribution::Kind init_A;
  cutlass::Distribution::Kind init_B;
  cutlass::Distribution::Kind init_C;
  uint64_t seed;

  cutlass::HostTensor<typename Gemm::ElementA, typename Gemm::LayoutA> tensor_A;
  cutlass::HostTensor<typename Gemm::ElementB, typename Gemm::LayoutB0> tensor_B0;
  cutlass::HostTensor<typename Gemm::ElementC, typename Gemm::LayoutC> tensor_C0;
  cutlass::HostTensor<typename Gemm::ElementB, typename Gemm::LayoutB1> tensor_B1;
  cutlass::HostTensor<typename Gemm::ElementC, typename Gemm::LayoutC> tensor_C1;
  cutlass::HostTensor<typename Gemm::EpilogueOutputOp0::ElementOutput, typename Gemm::LayoutC> tensor_D0;
  cutlass::HostTensor<typename Gemm::EpilogueOutputOp1::ElementOutput, typename Gemm::LayoutC> tensor_D1;
  cutlass::HostTensor<typename Gemm::EpilogueOutputOp2::ElementOutput, typename Gemm::LayoutC> tensor_D2;
  cutlass::HostTensor<ElementAccumulator, typename Gemm::LayoutC> tmp_D;
  cutlass::HostTensor<typename Gemm::EpilogueOutputOp0::ElementOutput, typename Gemm::LayoutC> reference_D;
  //
  // Methods
  //

  TestbedRunner(
    cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_C_ = cutlass::Distribution::Uniform,
    uint64_t seed_ = 2080
  ):
    init_A(init_A_), init_B(init_B_), init_C(init_C_), seed(seed_) { }

  /// Helper to initialize scaling factors
  template <typename Element, typename Layout>
  bool initialize_scale_factor(cutlass::TensorView<Element, Layout> view, uint64_t seed, int bits=0) {
    cutlass::reference::host::TensorFillRandomUniform(view, seed, double(1.), double(0.), bits);
    return true;
  }

  /// Helper to initialize a tensor view
  template <typename Element, typename Layout>
  bool initialize_tensor(
    cutlass::TensorView<Element, Layout> view,
    cutlass::Distribution::Kind dist_kind,
    uint64_t seed) {

    if (dist_kind == cutlass::Distribution::Uniform) {

      double scope_max, scope_min;
      int bits_input = cutlass::sizeof_bits<Element>::value;
      int bits_output = cutlass::sizeof_bits<typename Gemm::ElementC>::value;

      if (bits_input == 1) {
        scope_max = 2;
        scope_min = 0;
      } else if (bits_input <= 8) {
        scope_max = 2;
        scope_min = -2;
      } else if (bits_output == 16) {
        scope_max = 5;
        scope_min = -5;
      } else {
        scope_max = 8;
        scope_min = -8;
      }

      cutlass::reference::host::TensorFillRandomUniform(
        view, seed, scope_max, scope_min, 0);
    }
    else if (dist_kind == cutlass::Distribution::Identity) {

      cutlass::reference::host::TensorFillIdentity(view);
    }
    else if (dist_kind == cutlass::Distribution::Gaussian) {

      cutlass::reference::host::TensorFillRandomGaussian(view, seed, 0, 0.5);
    }
    else if (dist_kind == cutlass::Distribution::Sequential) {

      cutlass::reference::host::BlockFillSequential(
        view.data(), view.capacity());
    }
    else {
      std::cerr << "Not implemented";
      return false;
    }

    return true;
  }

  /// Initializes data structures
  void initialize(const Options& options) {
    //
    // Allocate the GEMM workspace
    //

    int batch_count = options.batch;
    cutlass::gemm::GemmCoord problem_size = options.problem_size;

    tensor_A.resize(
          cutlass::platform::is_same<typename Gemm::LayoutA, cutlass::layout::RowMajor>::value ?
          cutlass::MatrixCoord(batch_count * problem_size.m(), problem_size.k()) :
          cutlass::MatrixCoord(problem_size.m(), batch_count * problem_size.k()));
    tensor_B0.resize(
          cutlass::platform::is_same<typename Gemm::LayoutB0, cutlass::layout::RowMajor>::value ?
          cutlass::MatrixCoord(batch_count * problem_size.k(), problem_size.n()) :
          cutlass::MatrixCoord(problem_size.k(), batch_count * problem_size.n()));
    tensor_C0.resize({batch_count, options.problem_size.n()});
    tensor_B1.resize(cutlass::platform::is_same<typename Gemm::LayoutB0, cutlass::layout::RowMajor>::value ?
          cutlass::MatrixCoord(batch_count * problem_size.k(), problem_size.n()) :
          cutlass::MatrixCoord(problem_size.k(), batch_count * problem_size.n()));
    tensor_C1.resize({batch_count, options.problem_size.n()});
    tensor_D0.resize(
          cutlass::platform::is_same<typename Gemm::LayoutC, cutlass::layout::RowMajor>::value ?
          cutlass::MatrixCoord(batch_count * problem_size.m(), problem_size.n()) :
          cutlass::MatrixCoord(problem_size.m(), batch_count * problem_size.n()));
    tensor_D1.resize(
          cutlass::platform::is_same<typename Gemm::LayoutC, cutlass::layout::RowMajor>::value ?
          cutlass::MatrixCoord(batch_count * problem_size.m(), problem_size.n()) :
          cutlass::MatrixCoord(problem_size.m(), batch_count * problem_size.n()));
    tensor_D2.resize(
          cutlass::platform::is_same<typename Gemm::LayoutC, cutlass::layout::RowMajor>::value ?
          cutlass::MatrixCoord(batch_count * problem_size.m(), problem_size.n()) :
          cutlass::MatrixCoord(problem_size.m(), batch_count * problem_size.n()));
    reference_D.resize(
          cutlass::platform::is_same<typename Gemm::LayoutC, cutlass::layout::RowMajor>::value ?
          cutlass::MatrixCoord(batch_count * problem_size.m(), problem_size.n()) :
          cutlass::MatrixCoord(problem_size.m(), batch_count * problem_size.n()), false);
    tmp_D.resize(
          cutlass::platform::is_same<typename Gemm::LayoutC, cutlass::layout::RowMajor>::value ?
          cutlass::MatrixCoord(batch_count * problem_size.m(), problem_size.n()) :
          cutlass::MatrixCoord(problem_size.m(), batch_count * problem_size.n()), false);

    initialize_tensor(tensor_A.host_view(), init_A, seed + 2019);
    initialize_tensor(tensor_B0.host_view(), init_B, seed + 2018);
    initialize_tensor(tensor_C0.host_view(), init_C, seed + 2017);
    initialize_tensor(tensor_B1.host_view(), init_B, seed + 2018);
    initialize_tensor(tensor_C1.host_view(), init_C, seed + 2017);

    // It is possible to randomly initialize to all zeros, so override this with non-zeros
    // in the upper left corner of each operand.
    cutlass::Coord<2> origin(0);
    tensor_A.host_view().at(origin) = typename Gemm::ElementA(1);
    tensor_B0.host_view().at(origin) = typename Gemm::ElementB(1);
    tensor_C0.host_view().at(origin) = typename Gemm::ElementC(1);
    tensor_B1.host_view().at(origin) = typename Gemm::ElementB(1);
    tensor_C1.host_view().at(origin) = typename Gemm::ElementC(1);

    cutlass::reference::host::TensorFill(tensor_D0.host_view());
    cutlass::reference::host::TensorFill(tensor_D1.host_view());
    cutlass::reference::host::TensorFill(tensor_D2.host_view());
    cutlass::reference::host::TensorFill(reference_D.host_view());

    tensor_A.sync_device();
    tensor_B0.sync_device();
    tensor_C0.sync_device();
    tensor_B1.sync_device();
    tensor_C1.sync_device();
    tensor_D0.sync_device();
    tensor_D1.sync_device();
    tensor_D2.sync_device();
  }

  /// Compares computed reference with device reference and outputs to a file if incorrect
  bool compare_reference(const Options& options) {

    tensor_D0.sync_host();
    tensor_D1.sync_host();
    tensor_D2.sync_host();

    bool passed = cutlass::reference::host::TensorEquals(reference_D.host_view(), tensor_D0.host_view());

    if (!passed) {
      std::cerr << "Reference check failed" << std::endl;

      std::string output_file = "testbed_with_amax_errors.txt";
      std::ofstream file(output_file);

      file
        << "problem: " << options.problem_size
        << ", alpha: " << options.alpha0 << ", beta: " << options.beta0 << "\n\n";

      file
        << "A =\n" << tensor_A.host_view()
        << "\nB =\n" << tensor_B0.host_view()
        << "\nC =\n" << tensor_C0.host_view()
        << "\n\nReference D =\n" << reference_D.host_view()
        << "\nComputed D0 =\n" << tensor_D0.host_view()
        << "\nComputed D1 =\n" << tensor_D1.host_view()
        << "\nComputed D2 =\n" << tensor_D2.host_view();

      std::cerr << "Dumped results to " << output_file << std::endl;

    }

    return passed;
  }

  /// Verifies the result is a GEMM
  bool verify(const Options& options) {

    cutlass::Coord<2> origin(0);
    ElementCompute scaled_alpha = options.alpha0;
    ElementCompute scaled_beta = options.beta0;

    //
    // Verify
    //

    cutlass::reference::host::GemmComplex<
        typename Gemm::ElementA, typename Gemm::LayoutA,
        typename Gemm::ElementB, typename Gemm::LayoutB0,
        typename Gemm::ElementC, typename Gemm::LayoutC,
        ElementCompute, ElementAccumulator, ElementAccumulator
    >(
      options.problem_size,
      scaled_alpha,
      tensor_A.host_ref(),
      Gemm::kTransformA,
      tensor_B0.host_ref(),
      Gemm::kTransformB,
      scaled_beta,
      tensor_C0.host_ref(),
      tmp_D.host_ref(),
      ElementAccumulator(0)
    );

    cutlass::NumericConverter<ElementCompute, typename Gemm::ElementC> cvt_c_to_compute;
    cutlass::NumericConverter<ElementCompute, ElementAccumulator> cvt_accum_to_compute;
    cutlass::NumericConverter<ElementAccumulator, ElementCompute> cvt_compute_to_accum;
    cutlass::NumericConverter<typename Gemm::EpilogueOutputOp0::ElementOutput, ElementCompute> cvt_compute_to_d;

    for (int m = 0; m < options.problem_size.m(); ++m) {
      for (int n = 0; n < options.problem_size.n(); ++n) {
        ElementCompute d = cvt_accum_to_compute(tmp_D.host_view().at({m, n}));
        reference_D.host_view().at({m, n}) = cvt_compute_to_d(d);
      }
    }

    return compare_reference(options);
  }

  /// Returns true if the CUDA device is sufficient to execute the kernel.
  bool sufficient() const {

    if (__CUDACC_VER_MAJOR__ < 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 4)) {
      std::cerr << "This example requires CUDA 12.4 or greater." << std::endl;
      return false;
    }

    size_t smem_size = sizeof(typename Gemm::DualGemmKernel::SharedStorage);

    cudaDeviceProp properties;
    int device_idx;
    cudaError_t result = cudaGetDevice(&device_idx);

    if (result != cudaSuccess) {
      std::cerr << "cudaGetDevice() failed with error: " << cudaGetErrorString(result) << std::endl;
      return false;
    }

    result = cudaGetDeviceProperties(&properties, device_idx);

    if (result != cudaSuccess) {
      std::cerr << "cudaGetDeviceProperties() failed with error: " << cudaGetErrorString(result) << std::endl;
      return false;
    }

    if (properties.major < 8 || (properties.major == 8 && properties.minor < 9)) {
      std::cerr << "CUTLASS's Ada FP8 GEMM example requires a device of compute capability 89 or higher.\n" << std::endl;
      return false;
    }

    if (properties.sharedMemPerBlockOptin < smem_size) {
      std::cerr << "Insufficient shared memory. Need " << smem_size
                << ", but device only has " << properties.sharedMemPerBlockOptin << std::endl;
      return false;
    }

    return true;
  }

  /// Executes one test
  bool run(Options& options)
  {

    // Waive test if insufficient CUDA device
    if (!sufficient()) {
      std::cerr << "Insufficient resources to run the kernel." << std::endl;
      return false;
    }

    this->initialize(options);

    typename cutlass::TensorRef<typename Gemm::ElementC, typename Gemm::LayoutC> nullptr_ref{};

    int batch_count = options.batch;
    int split_k_slices = Gemm::kSplitKSerial ? 2 : 1;
    int lda = tensor_A.device_ref().stride(0);
    int ldb = tensor_B0.device_ref().stride(0);
    int ldd = tensor_D2.device_ref().stride(0);
    printf("lda: %d, ldb: %d, ldd: %d\n", lda, ldb, ldd);

    float alpha_out = 2.0f;

    bool hasbias = true;
    typename Gemm::Arguments arguments{
      (batch_count > 1 ?
        cutlass::gemm::DualGemmMode::kBatched :
        cutlass::gemm::DualGemmMode::kGemm),
      {options.problem_size.m(), options.problem_size.n(), options.problem_size.k()},
      // tensor_A.device_ref(),
      // tensor_B0.device_ref(),
      {tensor_A.device_data(), lda},
      {tensor_B0.device_data(), ldb},
      hasbias? typename cutlass::TensorRef<typename Gemm::ElementC, typename Gemm::LayoutC>{tensor_C0.device_data(), 0} : nullptr_ref,
      Gemm::kStoreD0 ? tensor_D0.device_ref() : nullptr_ref,
      // tensor_B1.device_ref(),
      {tensor_B1.device_data(), ldb},
      {tensor_C1.device_data(), 0},
      Gemm::kStoreD1 ? tensor_D1.device_ref() : nullptr_ref,
      // tensor_D2.device_ref(),
      {tensor_D2.device_data(), ldd},
      {options.alpha0, options.beta0},
      {options.alpha1, options.beta1},
      {alpha_out},
      split_k_slices,
      batch_count,
      options.problem_size.m() * options.problem_size.k(),
      options.problem_size.n() * options.problem_size.k(),
      options.problem_size.n() * options.problem_size.k(),
      options.problem_size.n(),
      options.problem_size.m() * options.problem_size.n()
      // batch_stride_A,
      // batch_stride_B0,
      // batch_stride_B1,
      // batch_stride_Bias,
      // batch_stride_D,
    };

    Gemm gemm_op;

    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Gemm::can_implement() failed" << std::endl;
      return false;
    }

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    status = gemm_op.initialize(arguments, workspace.get());
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Gemm::initialize() failed" << std::endl;
      return false;
    }

    //
    // Run the GEMM
    //

    status = gemm_op();

    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Gemm::run() failed" << std::endl;
      return false;
    }

    cudaError_t cuda_error = cudaDeviceSynchronize();
    if (cuda_error != cudaSuccess) {
      std::cerr << "CUDA error: " << cudaGetErrorString(cuda_error) << std::endl;
      return false;
    }

    //
    // Verify
    //

    bool passed = true;
    if (options.reference_check) {
      passed &= this->verify(options);
    } else {
      std::cout << "Skipped reference check" << std::endl;
    }

    //
    // Warm up
    //

    for (int i = 0; i < options.warmup_iterations; ++i) {
      gemm_op();
    }

    //
    // Profile
    //

    cudaEvent_t events[2];
    cudaError_t error;
    for (auto & event : events) {
      error = cudaEventCreate(&event);
      if (error != cudaSuccess) {
        std::cerr << "cudaEventCreate() failed: " << cudaGetErrorString(error) << std::endl;
        return false;
      }
    }

    // Record an event at the start of a series of GEMM operations
    error = cudaEventRecord(events[0]);
    if (error != cudaSuccess) {
      std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(error) << std::endl;
      return false;
    }

    // Run profiling loop
    for (int iter = 0; iter < options.iterations; ++iter) {
      gemm_op();
    }

    // Record an event when the GEMM operations have been launched.
    error = cudaEventRecord(events[1]);
    if (error != cudaSuccess) {
      std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(error) << std::endl;
      return false;
    }

    // Wait for work on the device to complete.
    error = cudaEventSynchronize(events[1]);
    if (error != cudaSuccess) {
      std::cerr << "cudaEventSynchronize() failed: " << cudaGetErrorString(error) << std::endl;
      return false;
    }

    // Measure elapsed runtime
    float runtime_ms = 0;
    error = cudaEventElapsedTime(&runtime_ms, events[0], events[1]);
    if (error != cudaSuccess) {
      std::cerr << "cudaEventElapsed() failed: " << cudaGetErrorString(error) << std::endl;
      return false;
    }

    // Compute average runtime and GFLOPs.
    runtime_ms = runtime_ms / float(options.iterations);
    float gflops = options.gflops(runtime_ms / 1000.0f);

    std::cout << "Problem size: " << options.problem_size.m() << 'x' << options.problem_size.n() << 'x' << options.problem_size.k() << std::endl;
    std::cout << "Runtime (ms): " << runtime_ms << std::endl;
    std::cout << "GFLOPs/sec:   " << gflops << std::endl;

    // Cleanup
    for (auto event : events) {
      (void)cudaEventDestroy(event);
    }

    return passed;
  }

};

/////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const** argv) {

  cudaDeviceProp props;

  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
    return -1;
  }

  if (__CUDACC_VER_MAJOR__ < 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 4) ||
      (props.major != 8 && props.minor != 9)) {

    //
    // This example requires an NVIDIA Ada-architecture GPU.
    //

    std::cout
      << "CUTLASS's FP8 SM89 example requires a GPU of NVIDIA's Ada architecture "
      << "and CUDA toolkit version 12.4 or later.\n";

    return 0;
  }

  //
  // Parse options
  //

  Options options;

  options.parse(argc, argv);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  if (options.error) {
    std::cerr << "Aborting execution." << std::endl;
    return -1;
  }

  std::cout << "Running GEMM with staged accumulation (OpMultiplyAdd)" << std::endl;
  std::cout << "=====================================================" << std::endl;
  TestbedRunner<DualGemm<cutlass::arch::OpMultiplyAdd>> testbed_staged_accum;
  bool passed = testbed_staged_accum.run(options);

  if (passed) {
    std::cout << "Passed" << std::endl;
  } else {
    std::cout << "Failed" << std::endl;
  }

  std::cout << "\nRunning GEMM with fast accumulation (OpMultiplyAddFastAccum)" << std::endl;
  std::cout << "============================================================" << std::endl;
  TestbedRunner<DualGemm<cutlass::arch::OpMultiplyAddFastAccum>> testbed_fast_accum;
  passed = testbed_fast_accum.run(options);

  if (passed) {
    std::cout << "Passed" << std::endl;
  } else {
    std::cout << "Failed" << std::endl;
  }

  return 0;
}
