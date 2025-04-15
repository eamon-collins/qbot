#include <torch/torch.h>
#include "Tree.h"

class ModelInference {
private:
    torch::jit::script::Module model;
    torch::Device device;
    int batch_size;
    std::deque<StateNode*> evaluation_queue;


	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> state_to_tensors(const StateNode* node);
    
public:
	ModelInference(const std::string& model_path, int batch_size = 16);
	ModelInference();

	void queue_for_evaluation(StateNode* node);
	void flush_queue();
	void process_batch();
	float evaluate_node(StateNode* node);

};
