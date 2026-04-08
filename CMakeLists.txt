cmake_minimum_required(VERSION 3.20)
project(NeuralNetHack VERSION 0.9.5 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(NNH_USE_BLAS "Use BLAS for matrix operations" ON)

# ---------------------------------------------------------------------------
# Compiler flags
# ---------------------------------------------------------------------------
add_compile_options(
    -Wall -pedantic -O3
    -march=native -ffast-math -funroll-loops -ftree-vectorize -fno-math-errno
)
add_compile_definitions(NDEBUG)

# ---------------------------------------------------------------------------
# BLAS detection
# ---------------------------------------------------------------------------
set(USE_BLAS 0)
if(NNH_USE_BLAS)
    find_package(BLAS QUIET)
    if(BLAS_FOUND)
        # Check for cblas.h
        include(CheckIncludeFileCXX)
        check_include_file_cxx(cblas.h HAVE_CBLAS_H)
        if(HAVE_CBLAS_H)
            set(USE_BLAS 1)
            message(STATUS "BLAS support enabled (${BLAS_LIBRARIES})")
        else()
            message(STATUS "BLAS found but cblas.h missing — disabled")
        endif()
    else()
        message(STATUS "BLAS not found — disabled")
    endif()
endif()

# ---------------------------------------------------------------------------
# config.h
# ---------------------------------------------------------------------------
configure_file(
    ${CMAKE_SOURCE_DIR}/config.h.cmake
    ${CMAKE_BINARY_DIR}/config.h
)
add_compile_definitions(HAVE_CONFIG_H)

# ---------------------------------------------------------------------------
# Library sources
# ---------------------------------------------------------------------------
set(LIB_SOURCES
    # mlp
    neuralnethack/mlp/MultiLayerPerceptron.cc
    neuralnethack/mlp/Trainer.cc
    neuralnethack/mlp/GradientDescent.cc
    neuralnethack/mlp/QuasiNewton.cc
    neuralnethack/mlp/Adam.cc
    neuralnethack/mlp/Error.cc
    neuralnethack/mlp/SummedSquare.cc
    neuralnethack/mlp/CrossEntropy.cc
    neuralnethack/mlp/Mlp.cc
    neuralnethack/mlp/Layer.cc
    neuralnethack/mlp/SigmoidLayer.cc
    neuralnethack/mlp/TanHypLayer.cc
    neuralnethack/mlp/LinearLayer.cc
    neuralnethack/mlp/ReLULayer.cc
    neuralnethack/mlp/LeakyReLULayer.cc
    neuralnethack/mlp/ELULayer.cc
    neuralnethack/mlp/Weights.cc
    neuralnethack/mlp/Serialization.cc
    # datatools
    neuralnethack/datatools/DataTools.cc
    neuralnethack/datatools/DataSet.cc
    neuralnethack/datatools/CoreDataSet.cc
    neuralnethack/datatools/Pattern.cc
    neuralnethack/datatools/DataManager.cc
    neuralnethack/datatools/Sampler.cc
    neuralnethack/datatools/BootstrapSampler.cc
    neuralnethack/datatools/CrossSplitSampler.cc
    neuralnethack/datatools/HoldOutSampler.cc
    neuralnethack/datatools/DummySampler.cc
    neuralnethack/datatools/Normaliser.cc
    # evaltools
    neuralnethack/evaltools/EvalTools.cc
    neuralnethack/evaltools/Evaluator.cc
    neuralnethack/evaltools/Gof.cc
    neuralnethack/evaltools/Roc.cc
    # matrixtools
    neuralnethack/matrixtools/MatrixTools.cc
    # parser
    neuralnethack/parser/Parser.cc
    neuralnethack/parser/NetworkParser.cc
    # top-level
    neuralnethack/NeuralNetHack.cc
    neuralnethack/Config.cc
    neuralnethack/Factory.cc
    neuralnethack/Ensemble.cc
    neuralnethack/EnsembleBuilder.cc
    neuralnethack/ModelEstimator.cc
    neuralnethack/ModelSelector.cc
    neuralnethack/FeatureSelector.cc
    neuralnethack/PrintUtils.cc
    neuralnethack/OddsRatio.cc
    neuralnethack/Saliency.cc
)

add_library(neuralnethack ${LIB_SOURCES})

target_include_directories(neuralnethack
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/neuralnethack>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
        $<INSTALL_INTERFACE:include>
)

if(USE_BLAS)
    target_link_libraries(neuralnethack PUBLIC ${BLAS_LIBRARIES})
endif()

# ---------------------------------------------------------------------------
# Executables
# ---------------------------------------------------------------------------
set(EXECUTABLES neuralnethack_bin ann modelselector featureselector featureselector2 saliency auc)
set(neuralnethack_bin_SOURCE src/neuralnethack.cc)
set(ann_SOURCE src/ann.cc)
set(modelselector_SOURCE src/modelselector.cc)
set(featureselector_SOURCE src/featureselector.cc)
set(featureselector2_SOURCE src/featureselector2.cc)
set(saliency_SOURCE src/saliency.cc)
set(auc_SOURCE src/auc.cc)

foreach(exe ${EXECUTABLES})
    add_executable(${exe} ${${exe}_SOURCE})
    target_link_libraries(${exe} PRIVATE neuralnethack)
endforeach()

# Rename neuralnethack_bin -> neuralnethack (avoid target name clash with library)
set_target_properties(neuralnethack_bin PROPERTIES OUTPUT_NAME neuralnethack)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
enable_testing()

set(TESTS testing testSampler testNormaliser testGof testMatrix testSaliency testParser)

foreach(test ${TESTS})
    add_executable(${test} test/${test}.cc)
    target_link_libraries(${test} PRIVATE neuralnethack)
    add_test(
        NAME ${test}
        COMMAND ${test}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test
    )
endforeach()
