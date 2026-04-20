// corvid_torch_smoke — Week 2 libtorch plumbing verification.
// Verifies: library links, CUDA available, minimal RavenNet stub forward pass.

#include <torch/torch.h>
#include <iostream>

// Minimal RavenNet trunk stub: matches the spec's medium MLP profile.
// Input: obs vector (64-d), output: action logits + value head (32+1).
struct RavenNetStub : torch::nn::Module {
  torch::nn::Linear fc1{nullptr}, fc2{nullptr}, action_head{nullptr}, value_head{nullptr};

  RavenNetStub() {
    fc1         = register_module("fc1",         torch::nn::Linear(64, 128));
    fc2         = register_module("fc2",         torch::nn::Linear(128, 64));
    action_head = register_module("action_head", torch::nn::Linear(64, 8));
    value_head  = register_module("value_head",  torch::nn::Linear(64, 1));
  }

  std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
    x = torch::relu(fc1->forward(x));
    x = torch::relu(fc2->forward(x));
    return {action_head->forward(x), value_head->forward(x)};
  }
};

int main() {
  std::cout << "libtorch " << TORCH_VERSION << "\n";
  std::cout << "CUDA available : " << (torch::cuda::is_available() ? "yes" : "no") << "\n";
  if (torch::cuda::is_available())
    std::cout << "CUDA devices   : " << torch::cuda::device_count() << "\n";

  auto device = torch::cuda::is_available() ? torch::Device(torch::kCUDA, 0)
                                             : torch::Device(torch::kCPU);
  std::cout << "running on     : " << device << "\n\n";

  // --- Forward pass ---
  RavenNetStub net;
  net.to(device);
  net.eval();

  const int batch = 8;
  auto obs = torch::randn({batch, 64}, torch::TensorOptions().device(device));
  auto [logits, value] = net.forward(obs);

  std::cout << "obs    : " << obs.sizes()    << " on " << obs.device()    << "\n";
  std::cout << "logits : " << logits.sizes() << " on " << logits.device() << "\n";
  std::cout << "value  : " << value.sizes()  << " on " << value.device()  << "\n\n";

  // --- LoRA stub: rank-8 adapter on fc1 weight ---
  const int rank = 8;
  auto lora_A = torch::randn({rank, 64},  torch::TensorOptions().device(device)).requires_grad_(true);
  auto lora_B = torch::zeros({128, rank}, torch::TensorOptions().device(device)).requires_grad_(true);
  auto adapted_w = net.fc1->weight + lora_B.mm(lora_A);  // W + B @ A  ([128,r]@[r,64]=[128,64])
  std::cout << "LoRA adapted fc1 weight: " << adapted_w.sizes() << "\n\n";

  // --- Gradient check ---
  auto loss = logits.mean() + value.mean();
  loss.backward();
  std::cout << "fc1.weight.grad norm   : "
            << net.fc1->weight.grad().norm().item<float>() << "\n";

  std::cout << "\n[torch smoke PASS]\n";
  return 0;
}
