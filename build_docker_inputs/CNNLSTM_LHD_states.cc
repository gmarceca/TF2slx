#include "aot_model.h"


extern "C" {

AotModel* model = nullptr;

void CreateNetwork(void) {
	assert(model == nullptr);
	model = new AotModel();
	// allocate an instance of the Graph, along with all its private buffers
	//Eigen::ThreadPool tp(std::thread::hardware_concurrency());
	//Eigen::ThreadPoolDevice device(&tp, tp.NumThreads());
	//aot_model.set_thread_pool(&device);
}

void DeleteNetwork(void) {
	assert(model != nullptr);
	delete model;
	model = nullptr;
}

int run(
		const float *input_signal,
		float *output_prediction,
		bool reset
    ){
	// copy over the input buffer
	std::copy_n(input_signal, model->arg_InputMeasurements_count(), model->arg_InputMeasurements_data());

	// For the first time run() is called reset the internal states to zero
	if (reset) {
	  // Hidden state h
          std::fill_n(model->arg_InputHidden_data(),
                      model->arg_InputHidden_count(), 0);
	  // Hidden state c
          std::fill_n(model->arg_InputCell_data(),
                      model->arg_InputCell_count(), 0);
        }

	// execute the inference model->
	auto ok = model->Run();
	if (not ok) return -1;

	// Now that the model has ran and produced the output result0_data().
	// Copy the result into the output buffer.
	std::copy_n(model->result_PredictionStates_data(),
                      model->result_PredictionStates_count(),
                      output_prediction);

	// Update the hidden states to compute the next time step (now with reset = false)
	// Hidden state h
        std::copy_n(model->result_OutputHidden_data(),
                      model->arg_InputHidden_count(),
                      model->arg_InputHidden_data());
	// Hidden state c
        std::copy_n(model->result_OutputCell_data(),
                      model->arg_InputCell_count(),
                      model->arg_InputCell_data());

	return 0;
}

}
