#ifndef __INNER_PRODUCT_H__
#define __INNER_PRODUCT_H__

namespace tensorflow
{
namespace functor
{

template <typename Device, typename T>
struct InnerProductFunctor
{
    void operator()(const Device &d, int weight_height, int weight_width, const T *data, const T *weight, T *out);
};

template<typename Device, typename T>
struct InnerProductGradFunctor
{
    void operator()(const Device &d, int weight_height, int weight_width, 
                    const T *data, const T *weight, const T *grad_output,
                    T *grad_data, T *grad_weight);
};


    
} // namespace functor

    
} // namespace tensorflow


#endif // __INNER_PRODUCT_H__
