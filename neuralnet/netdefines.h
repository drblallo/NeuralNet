#pragma once
#include <vector>

namespace neuralNet
{
	class Neuron;
	typedef double net_type;
	typedef std::vector<Neuron> Layer;

	struct Connection
	{
		net_type weight;
		net_type deltaWeight;
	};
};
