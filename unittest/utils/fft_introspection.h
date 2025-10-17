/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   FFT Introspection API - Runtime detection of FFT configuration

   Purpose: Provide compile-time FFT configuration detection for testing

   Documentation: See doc/src/Developer_unittest.rst
                  (Section: FFT Testing Infrastructure)

   This header-only utility provides functions to detect and report the
   current FFT library, precision, threading support, and KOKKOS backend
   configuration at runtime. All detection is performed at compile-time
   using preprocessor macros, resulting in zero runtime overhead.

   Usage Example:

       #include "fft_introspection.h"

       TEST_F(FFT3DTest, ConfigurationDetection) {
           using namespace FFTIntrospection;

           // Print comprehensive configuration
           std::cout << get_fft_configuration() << std::endl;

           // Query specific settings
           EXPECT_FALSE(get_fft_library().empty());
           EXPECT_EQ(get_fft_precision(), "double");

           // Conditionally skip tests
           if (!is_fft_fftw3()) {
               GTEST_SKIP() << "Test requires FFTW3, current: "
                           << get_fft_library();
           }
       }

   Architecture:
   - Namespace: FFTIntrospection (all functions in this namespace)
   - Header-only: All functions are inline
   - Zero overhead: Compile-time string constants
   - No dependencies: Only requires lmpfftsettings.h and standard library

   Supported FFT Libraries:
   - Standard: KISS FFT, FFTW3, MKL, NVPL, cuFFT, hipFFT, MKL_GPU, HeFFTe
   - KOKKOS: KISS FFT, FFTW3, MKL, NVPL, cuFFT, hipFFT, MKL_GPU

   Supported KOKKOS Backends:
   - GPU: CUDA, HIP, SYCL, OpenMPTarget
   - CPU: OpenMP, Threads, Serial

   Note: This API is intended for unit testing only
------------------------------------------------------------------------- */

#ifndef FFT_INTROSPECTION_H
#define FFT_INTROSPECTION_H

#include "lmpfftsettings.h"

#include <map>
#include <string>

namespace FFTIntrospection {

// ============================================================================
// Standard FFT Library Detection
// ============================================================================

/**
 * Get the name of the standard FFT library in use
 * @return Library name: "KISS FFT", "FFTW3", "MKL FFT", "cuFFT", "hipFFT",
 *         "MKL GPU FFT", "HeFFTe(FFTW3)", "HeFFTe(MKL)", or "HeFFTe(builtin)"
 */
inline std::string get_fft_library()
{
#if defined(LMP_FFT_LIB)
    return LMP_FFT_LIB;
#else
    return "Unknown";
#endif
}

/**
 * Get FFT precision as a string
 * @return "single" or "double"
 */
inline std::string get_fft_precision()
{
#if defined(LMP_FFT_PREC)
    return LMP_FFT_PREC;
#else
    return "unknown";
#endif
}

/**
 * Check if FFT threading support is available
 * @return true if FFTW3 or MKL threading is enabled
 */
inline bool has_fft_threading()
{
#if defined(FFT_FFTW_THREADS) || defined(FFT_MKL_THREADS)
    return true;
#else
    return false;
#endif
}

/**
 * Get FFT threading information
 * @return Threading mode: "FFTW3 threads", "MKL threads", or "none"
 */
inline std::string get_fft_threading_info()
{
#if defined(FFT_FFTW_THREADS)
    return "FFTW3 threads";
#elif defined(FFT_MKL_THREADS)
    return "MKL threads";
#else
    return "none";
#endif
}

// ============================================================================
// KOKKOS FFT Library Detection
// ============================================================================

/**
 * Get the name of the KOKKOS FFT library in use
 * @return Library name or "N/A" if KOKKOS is not enabled
 */
inline std::string get_kokkos_fft_library()
{
#ifdef LMP_KOKKOS
#if defined(LMP_FFT_KOKKOS_LIB)
    return LMP_FFT_KOKKOS_LIB;
#else
    return "Unknown";
#endif
#else
    return "N/A";
#endif
}

/**
 * Get the KOKKOS backend name
 * @return Backend name: "CUDA", "HIP", "SYCL", "OpenMPTarget", "OpenMP",
 *         "Threads", "Serial", or "N/A" if KOKKOS is not enabled
 */
inline std::string get_kokkos_backend()
{
#ifdef LMP_KOKKOS
#if defined(KOKKOS_ENABLE_CUDA)
    return "CUDA";
#elif defined(KOKKOS_ENABLE_HIP)
    return "HIP";
#elif defined(KOKKOS_ENABLE_SYCL)
    return "SYCL";
#elif defined(KOKKOS_ENABLE_OPENMPTARGET)
    return "OpenMPTarget";
#elif defined(KOKKOS_ENABLE_OPENMP)
    return "OpenMP";
#elif defined(KOKKOS_ENABLE_THREADS)
    return "Threads";
#else
    return "Serial";
#endif
#else
    return "N/A";
#endif
}

/**
 * Get human-readable KOKKOS device type
 * @return Device type: "GPU (CUDA)", "GPU (HIP)", "GPU (SYCL)",
 *         "GPU (OpenMPTarget)", "CPU (OpenMP)", "CPU (Threads)",
 *         "CPU (Serial)", or "N/A" if KOKKOS is not enabled
 */
inline std::string get_kokkos_device_type()
{
#ifdef LMP_KOKKOS
#if defined(KOKKOS_ENABLE_CUDA)
    return "GPU (CUDA)";
#elif defined(KOKKOS_ENABLE_HIP)
    return "GPU (HIP)";
#elif defined(KOKKOS_ENABLE_SYCL)
    return "GPU (SYCL)";
#elif defined(KOKKOS_ENABLE_OPENMPTARGET)
    return "GPU (OpenMPTarget)";
#elif defined(KOKKOS_ENABLE_OPENMP)
    return "CPU (OpenMP)";
#elif defined(KOKKOS_ENABLE_THREADS)
    return "CPU (Threads)";
#else
    return "CPU (Serial)";
#endif
#else
    return "N/A";
#endif
}

/**
 * Check if KOKKOS is using a GPU backend
 * @return true if KOKKOS backend is CUDA, HIP, SYCL, or OpenMPTarget
 */
inline bool is_kokkos_gpu_backend()
{
#ifdef LMP_KOKKOS
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) ||     \
    defined(KOKKOS_ENABLE_SYCL) || defined(KOKKOS_ENABLE_OPENMPTARGET)
    return true;
#endif
#endif
    return false;
}

/**
 * Check if KOKKOS FFT threading is available
 * @return true if FFTW3 threading is enabled for KOKKOS
 */
inline bool has_kokkos_fft_threading()
{
#ifdef LMP_KOKKOS
#if defined(FFT_KOKKOS_FFTW_THREADS)
    return true;
#endif
#endif
    return false;
}

/**
 * Get KOKKOS FFT threading information
 * @return Threading mode: "FFTW3 threads" or "none" or "N/A"
 */
inline std::string get_kokkos_fft_threading_info()
{
#ifdef LMP_KOKKOS
#if defined(FFT_KOKKOS_FFTW_THREADS)
    return "FFTW3 threads";
#else
    return "none";
#endif
#else
    return "N/A";
#endif
}

// ============================================================================
// Boolean Query Helpers - Standard FFT
// ============================================================================

inline bool is_fft_kiss()
{
#if defined(FFT_KISS)
    return true;
#elif !defined(FFT_FFTW3) && !defined(FFT_MKL) && !defined(FFT_MKL_GPU) && \
    !defined(FFT_CUFFT) && !defined(FFT_HIPFFT) && !defined(FFT_HEFFTE)
    return true;    // Default is KISS FFT
#else
    return false;
#endif
}

inline bool is_fft_fftw3()
{
#if defined(FFT_FFTW3)
    return true;
#else
    return false;
#endif
}

inline bool is_fft_mkl()
{
#if defined(FFT_MKL)
    return true;
#else
    return false;
#endif
}

inline bool is_fft_mkl_gpu()
{
#if defined(FFT_MKL_GPU)
    return true;
#else
    return false;
#endif
}

inline bool is_fft_nvpl()
{
#if defined(FFT_NVPL)
    return true;
#else
    return false;
#endif
}

inline bool is_fft_cufft()
{
#if defined(FFT_CUFFT)
    return true;
#else
    return false;
#endif
}

inline bool is_fft_hipfft()
{
#if defined(FFT_HIPFFT)
    return true;
#else
    return false;
#endif
}

inline bool is_fft_heffte()
{
#if defined(FFT_HEFFTE)
    return true;
#else
    return false;
#endif
}

inline bool is_fft_single_precision()
{
#if defined(FFT_SINGLE)
    return true;
#else
    return false;
#endif
}

inline bool is_fft_double_precision()
{
    return !is_fft_single_precision();
}

// ============================================================================
// Boolean Query Helpers - KOKKOS FFT
// ============================================================================

inline bool is_kokkos_fft_kiss()
{
#ifdef LMP_KOKKOS
#if defined(FFT_KOKKOS_KISS)
    return true;
#elif !defined(FFT_KOKKOS_FFTW3) && !defined(FFT_KOKKOS_MKL) &&            \
    !defined(FFT_KOKKOS_MKL_GPU) && !defined(FFT_KOKKOS_CUFFT) &&          \
    !defined(FFT_KOKKOS_HIPFFT) && !defined(FFT_KOKKOS_NVPL)
    return true;    // Default is KISS FFT
#endif
#endif
    return false;
}

inline bool is_kokkos_fft_fftw3()
{
#ifdef LMP_KOKKOS
#if defined(FFT_KOKKOS_FFTW3)
    return true;
#endif
#endif
    return false;
}

inline bool is_kokkos_fft_mkl()
{
#ifdef LMP_KOKKOS
#if defined(FFT_KOKKOS_MKL)
    return true;
#endif
#endif
    return false;
}

inline bool is_kokkos_fft_mkl_gpu()
{
#ifdef LMP_KOKKOS
#if defined(FFT_KOKKOS_MKL_GPU)
    return true;
#endif
#endif
    return false;
}

inline bool is_kokkos_fft_nvpl()
{
#ifdef LMP_KOKKOS
#if defined(FFT_KOKKOS_NVPL)
    return true;
#endif
#endif
    return false;
}

inline bool is_kokkos_fft_cufft()
{
#ifdef LMP_KOKKOS
#if defined(FFT_KOKKOS_CUFFT)
    return true;
#endif
#endif
    return false;
}

inline bool is_kokkos_fft_hipfft()
{
#ifdef LMP_KOKKOS
#if defined(FFT_KOKKOS_HIPFFT)
    return true;
#endif
#endif
    return false;
}

// ============================================================================
// Boolean Query Helpers - KOKKOS Backend
// ============================================================================

inline bool is_kokkos_cuda()
{
#if defined(KOKKOS_ENABLE_CUDA)
    return true;
#else
    return false;
#endif
}

inline bool is_kokkos_hip()
{
#if defined(KOKKOS_ENABLE_HIP)
    return true;
#else
    return false;
#endif
}

inline bool is_kokkos_sycl()
{
#if defined(KOKKOS_ENABLE_SYCL)
    return true;
#else
    return false;
#endif
}

inline bool is_kokkos_openmptarget()
{
#if defined(KOKKOS_ENABLE_OPENMPTARGET)
    return true;
#else
    return false;
#endif
}

inline bool is_kokkos_openmp()
{
#if defined(KOKKOS_ENABLE_OPENMP)
    return true;
#else
    return false;
#endif
}

inline bool is_kokkos_threads()
{
#if defined(KOKKOS_ENABLE_THREADS)
    return true;
#else
    return false;
#endif
}

inline bool is_kokkos_serial()
{
#ifdef LMP_KOKKOS
#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP) &&          \
    !defined(KOKKOS_ENABLE_SYCL) && !defined(KOKKOS_ENABLE_OPENMPTARGET) && \
    !defined(KOKKOS_ENABLE_OPENMP) && !defined(KOKKOS_ENABLE_THREADS)
    return true;
#endif
#endif
    return false;
}

inline bool is_kokkos_enabled()
{
#ifdef LMP_KOKKOS
    return true;
#else
    return false;
#endif
}

// ============================================================================
// Comprehensive Reporting
// ============================================================================

/**
 * Get comprehensive FFT configuration as a map
 * @return Map with keys: fft_library, fft_precision, fft_threading,
 *         kokkos_fft_library, kokkos_backend, kokkos_device_type,
 *         kokkos_fft_threading
 */
inline std::map<std::string, std::string> get_fft_config_map()
{
    std::map<std::string, std::string> config;

    // Standard FFT configuration
    config["fft_library"] = get_fft_library();
    config["fft_precision"] = get_fft_precision();
    config["fft_threading"] = get_fft_threading_info();

    // KOKKOS FFT configuration
    config["kokkos_enabled"] = is_kokkos_enabled() ? "yes" : "no";
    config["kokkos_fft_library"] = get_kokkos_fft_library();
    config["kokkos_backend"] = get_kokkos_backend();
    config["kokkos_device_type"] = get_kokkos_device_type();
    config["kokkos_fft_threading"] = get_kokkos_fft_threading_info();

    return config;
}

/**
 * Get comprehensive FFT configuration as a formatted multi-line string
 * @return Human-readable configuration report
 */
inline std::string get_fft_configuration()
{
    std::string config;

    config += "FFT Configuration:\n";
    config += "==================\n";
    config += "Standard FFT:\n";
    config += "  Library:    " + get_fft_library() + "\n";
    config += "  Precision:  " + get_fft_precision() + "\n";
    config += "  Threading:  " + get_fft_threading_info() + "\n";

#ifdef LMP_KOKKOS
    config += "\nKOKKOS FFT:\n";
    config += "  Library:    " + get_kokkos_fft_library() + "\n";
    config += "  Backend:    " + get_kokkos_backend() + "\n";
    config += "  Device:     " + get_kokkos_device_type() + "\n";
    config += "  Threading:  " + get_kokkos_fft_threading_info() + "\n";
#else
    config += "\nKOKKOS: Not enabled\n";
#endif

    return config;
}

/**
 * Get a short one-line configuration summary
 * @return Compact configuration string, e.g., "FFTW3/double/threads"
 */
inline std::string get_fft_config_summary()
{
    std::string summary = get_fft_library() + "/" + get_fft_precision();

    if (has_fft_threading()) {
        summary += "/threads";
    }

#ifdef LMP_KOKKOS
    summary += " | KOKKOS:" + get_kokkos_fft_library() + ":" + get_kokkos_backend();
#endif

    return summary;
}

}    // namespace FFTIntrospection

#endif    // FFT_INTROSPECTION_H
