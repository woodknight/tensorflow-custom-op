list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules) 
find_package(TensorFlow REQUIRED)
find_package(CUDA 9 REQUIRED)

# set necessary flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SSE_FLAGS} -march=native -fopenmp -D_GLIBCXX_USE_CXX11_ABI=${TensorFlow_ABI}")
set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -fPIC --shared -D_GLIBCXX_USE_CXX11_ABI=${TensorFlow_ABI}" )
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11  --expt-relaxed-constexpr -D GOOGLE_CUDA=1 --gpu-architecture=sm_52 -D_GLIBCXX_USE_CXX11_ABI=${TensorFlow_ABI}" )

get_filename_component(TF_CUDA ${CUDA_INCLUDE_DIRS} DIRECTORY)
get_filename_component(TF_CUDA ${TF_CUDA} DIRECTORY)
include_directories(SYSTEM ${TF_CUDA})
include_directories(SYSTEM ${TensorFlow_INCLUDE_DIR})

# quick fix for drone-ci
include_directories(SYSTEM "/usr/local/")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_tensorflow_gpu_operation(example)