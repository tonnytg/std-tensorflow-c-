#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/framework/tensor.h>

int main() {
    using namespace tensorflow;
    using namespace tensorflow::ops;

    // Criação do escopo
    Scope root = Scope::NewRootScope();

    // Criação de um placeholder para a entrada
    auto input = Placeholder(root, DT_FLOAT, Placeholder::Shape({1, 3}));

    // Criação de variáveis para os pesos e bias
    auto weights = Variable(root, {3, 2}, DT_FLOAT);
    auto biases = Variable(root, {2}, DT_FLOAT);

    // Inicialização dos pesos e bias
    auto assign_weights = Assign(root, weights, Const(root, {{0.1f, 0.2f}, {0.3f, 0.4f}, {0.5f, 0.6f}}));
    auto assign_biases = Assign(root, biases, Const(root, {0.1f, 0.2f}));

    // Cálculo da camada densa
    auto dense_layer = Add(root, MatMul(root, input, weights), biases);

    // Criação de uma sessão para executar as operações
    ClientSession session(root);

    // Executar a inicialização das variáveis
    session.Run({assign_weights, assign_biases}, nullptr);

    // Criação do tensor de entrada
    Tensor input_data(DT_FLOAT, TensorShape({1, 3}));
    auto input_matrix = input_data.matrix<float>();
    input_matrix(0, 0) = 1.0f;
    input_matrix(0, 1) = 2.0f;
    input_matrix(0, 2) = 3.0f;

    // Executar a camada densa
    std::vector<Tensor> outputs;
    session.Run({{input, input_data}}, {dense_layer}, &outputs);

    // Exibir o resultado
    std::cout << outputs[0].matrix<float>() << std::endl;

    return 0;
}
