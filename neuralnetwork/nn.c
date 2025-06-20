#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

float  or_gate_model[][3] = {
	{0,0,0},
	{1,0,1},
	{0,1,1},
	{1,1,1},
};

float  and_gate_model[][3] = {
	{0,0,0},
	{1,0,0},
	{0,1,0},
	{1,1,1},
};

float nand_gate_model[][3] = {
	{0, 0, 1},
	{0, 1, 1},
	{1, 0, 1},
	{1, 1, 0},
};

float xor_gate_model[][3] = {
	{0, 0, 0},
	{0, 1, 1},
	{1, 0, 1},
	{1, 1, 0},
};

#define ITERATIONS 500
#define MODEL_SIZE 4

typedef struct {
	float w1_hidden_1, w2_hidden_1, b_hidden_1;
	float w1_hidden_2, w2_hidden_2, b_hidden_2;
	float w_h_output_1, w_h_output_2, b_output;
} NetworkParams;

typedef struct {
	float h1_input, h1_output;
	float h2_input, h2_output;
	float out_input, y_pred;
} ForwardPass_Intermediate_Values;

float rand_float(void) {
	return (float)rand() / (float)RAND_MAX;
}

float sigmoid_f(float x) {
	return 1.f / (1.f + expf(-x));
}

float sigmoid_derivative(float x) {
	float s = sigmoid_f(x);
	return s * (1 - s);
}

/// <summary>
///	Calculates each layer’s intermediate values and stores them in the ForwardPass_Intermediate_Values struct (cache)
///	so they can be used later, such as during backpropagation.
/// </summary>
void forward_pass(float x1, float x2, NetworkParams* p, ForwardPass_Intermediate_Values* cache) {
	cache->h1_input = x1 * p->w1_hidden_1 + x2 * p->w2_hidden_1 + p->b_hidden_1;
	cache->h1_output = sigmoid_f(cache->h1_input);
	cache->h2_input = x1 * p->w1_hidden_2 + x2 * p->w2_hidden_2 + p->b_hidden_2;
	cache->h2_output = sigmoid_f(cache->h2_input);
	cache->out_input = cache->h1_output * p->w_h_output_1 + cache->h2_output * p->w_h_output_2 + p->b_output;
	cache->y_pred = sigmoid_f(cache->out_input);
}

/// <summary>
/// Updates the network’s weights and biases to reduce the error between the predicted output and the expected output
/// works backward through the network, calculating how much each weight and bias contributed to the error
/// It uses the derivative of the sigmoid function and the chain rule from calculus to do this
/// The gradients show the direction and size of the adjustment needed for each parameter
/// </summary>
void backward_pass(float x1, float x2, float y_true, NetworkParams* p, float magnitude, ForwardPass_Intermediate_Values* cache) {
	// Output layer error
	float d_loss_d_ypred = 2 * (cache->y_pred - y_true);
	float d_ypred_d_out_input = sigmoid_derivative(cache->out_input);
	float d_loss_d_out_input = d_loss_d_ypred * d_ypred_d_out_input;

	// Gradients for output layer weights and bias
	float grad_w_h_output_1 = d_loss_d_out_input * cache->h1_output;
	float grad_w_h_output_2 = d_loss_d_out_input * cache->h2_output;
	float grad_b_output = d_loss_d_out_input;

	// Hidden layer error
	float d_loss_d_h1_output = d_loss_d_out_input * p->w_h_output_1;
	float d_loss_d_h2_output = d_loss_d_out_input * p->w_h_output_2;
	float d_h1_output_d_h1_input = sigmoid_derivative(cache->h1_input);
	float d_h2_output_d_h2_input = sigmoid_derivative(cache->h2_input);
	float d_loss_d_h1_input = d_loss_d_h1_output * d_h1_output_d_h1_input;
	float d_loss_d_h2_input = d_loss_d_h2_output * d_h2_output_d_h2_input;

	// Gradients for hidden layer weights and biases
	float grad_w1_hidden_1 = d_loss_d_h1_input * x1;
	float grad_w2_hidden_1 = d_loss_d_h1_input * x2;
	float grad_b_hidden_1 = d_loss_d_h1_input;
	float grad_w1_hidden_2 = d_loss_d_h2_input * x1;
	float grad_w2_hidden_2 = d_loss_d_h2_input * x2;
	float grad_b_hidden_2 = d_loss_d_h2_input;

	// Update the weights and biases
	p->w1_hidden_1 -= magnitude * grad_w1_hidden_1;
	p->w2_hidden_1 -= magnitude * grad_w2_hidden_1;
	p->b_hidden_1 -= magnitude * grad_b_hidden_1;
	p->w1_hidden_2 -= magnitude * grad_w1_hidden_2;
	p->w2_hidden_2 -= magnitude * grad_w2_hidden_2;
	p->b_hidden_2 -= magnitude * grad_b_hidden_2;
	p->w_h_output_1 -= magnitude * grad_w_h_output_1;
	p->w_h_output_2 -= magnitude * grad_w_h_output_2;
	p->b_output -= magnitude * grad_b_output;
}

void dump_results(const char* gate_name, NetworkParams* p) {
	printf("\n------ Results for %s gate ------\n", gate_name);
	for (size_t i = 0; i < 2; i++) {
		for (size_t j = 0; j < 2; j++) {
			ForwardPass_Intermediate_Values cache;
			forward_pass((float)i, (float)j, p, &cache);
			printf("%s(%zu, %zu) = %.4f\n", gate_name, i, j, cache.y_pred);
		}
	}
	printf("\n");
}

/// <summary>
/// Train the model with weights and bias
/// </summary>
void train_logic_gate(const char* gate_name, float (*model)[3], NetworkParams* p,
	float eps, float magnitude, size_t iterations) {
	ForwardPass_Intermediate_Values cache;
	for (size_t i = 0; i < iterations; i++) {
		size_t sample = i % MODEL_SIZE;
		float x1 = model[sample][0];
		float x2 = model[sample][1];
		float y_expected = model[sample][2];
		forward_pass(x1, x2, p, &cache);
		backward_pass(x1, x2, y_expected, p, magnitude, &cache);
	}

	// check accuracy after training
	int correct_count = 0;
	for (size_t i = 0; i < MODEL_SIZE; i++)
	{
		float x1 = model[i][0];
		float x2 = model[i][1];
		float y_expected = model[i][2];
		int y_pred_label;
		
		forward_pass(x1, x2, p, &cache);
		
		if (cache.y_pred >= 0.95) {
			y_pred_label = 1;
		}
		else {
			y_pred_label = 0;
		}
		
		if (y_pred_label == (int)y_expected) correct_count++;
	}
	dump_results(gate_name, p);
	printf("Accuracy: %.2f%%\n", 100.0 * correct_count / MODEL_SIZE);
}



int main() {
	srand(1);

	float eps = 1e-1;
	float magnitude = 1e-1;

	NetworkParams p = {
		rand_float(), rand_float(), rand_float(),
		rand_float(), rand_float(), rand_float(),
		rand_float(), rand_float(), rand_float()
	};
	train_logic_gate("OR", or_gate_model, &p, eps, magnitude, ITERATIONS);

	p = (NetworkParams){ rand_float(), rand_float(), rand_float(),
						rand_float(), rand_float(), rand_float(),
						rand_float(), rand_float(), rand_float() };
	train_logic_gate("AND", and_gate_model, &p, eps, magnitude, ITERATIONS);

	p = (NetworkParams){ rand_float(), rand_float(), rand_float(),
						rand_float(), rand_float(), rand_float(),
						rand_float(), rand_float(), rand_float() };
	train_logic_gate("NAND", nand_gate_model, &p, eps, magnitude, ITERATIONS);

	p = (NetworkParams){ rand_float(), rand_float(), rand_float(),
						rand_float(), rand_float(), rand_float(),
						rand_float(), rand_float(), rand_float() };
	train_logic_gate("XOR", xor_gate_model, &p, eps, magnitude, ITERATIONS);

	return 0;
}
