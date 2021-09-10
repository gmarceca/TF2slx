#ifndef LEARNING_MY_EXPORT_POLICY_POLICY_EXPORT_H_
#define LEARNING_MY_EXPORT_POLICY_POLICY_EXPORT_H_

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// C bindings for interacting with matlab/simulink. C++ users should use the
// C++ library `policy_wrapper` directly.
//
// These functions store a single C++ instance in a global variable. Given that
// it wraps a stateful RNN, it shouldn't be used from multiple threads. Even
// with proper locking interleaved NetworkOutput calls will break the RNN state.
// Failing to call Delete will lead to a memory leak, and calling Create or
// Delete out of order will assert to make catching this easier.
void CreateNetwork(void);
void DeleteNetwork(void);
void run(const float* input, float* output_states,
    bool reset);
#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // LEARNING_MY_EXPORT_POLICY_POLICY_EXPORT_H_
