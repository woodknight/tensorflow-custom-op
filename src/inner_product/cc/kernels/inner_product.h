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

    
} // namespace functor

    
} // namespace tensorflow


#endif // __INNER_PRODUCT_H__
