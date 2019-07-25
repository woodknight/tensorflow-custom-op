#ifndef __TIME_TWO_H__
#define __TIME_TWO_H__

namespace tensorflow
{
namespace functor
{

template <typename Device, typename T>
struct TimeTwoFunctor
{
    void operator()(const Device &d, int size, const T *in, T *out);
};
    
} // namespace functor

    
} // namespace tensorflow


#endif // __TIME_TWO_H__
