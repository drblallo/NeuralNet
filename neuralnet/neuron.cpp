#include "neuron.h"
#include <random>
#include <cmath>

using namespace neuralNet;

double Neuron::eta(0.3);
double Neuron::alpha(0.5);

Neuron::Neuron(unsigned numOutputs, unsigned index) : m_myIndex(index)
{
	std::default_random_engine engine;
	std::uniform_real_distribution<net_type> dist(0, 1);
	for (unsigned c = 0; c < numOutputs; c++)
	{
		m_outputWeights.push_back(Connection());
		m_outputWeights.back().weight = dist(engine);
	}
}


void Neuron::feedForward(const Layer* prevLayer)
{
	net_type sum(0);

	for (unsigned n = 0; n < prevLayer->size(); n++)	
	{
		sum += prevLayer->at(n).m_outputValue 
			* prevLayer->at(n).m_outputWeights[m_myIndex].weight;
	}
	
	m_outputValue = transferFunction(sum);
}

net_type Neuron::transferFunction(net_type x)
{
	return std::tanh(x);
}

net_type Neuron::transferFunctionDerivative(net_type x)
{
	net_type t(std::tanh(x));
	t = t*t;
	return 1-t;
}

void Neuron::calcOutputGradients(net_type targetVal)
{
	double delta(targetVal - m_outputValue);
	m_gradient = delta * Neuron::transferFunctionDerivative(m_outputValue);
}

void Neuron::calcHiddenGradients(const Layer& nextLayer)
{
	m_gradient = sumDOW(nextLayer) * Neuron::transferFunctionDerivative(m_outputValue);
}

double Neuron::sumDOW(const Layer &nextLayer) const
{
	double sum(0);

	for (unsigned n = 0; n < nextLayer.size() - 1; n++)
	{
		sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
	}

	return sum;

}

void Neuron::updateInputWeights(Layer& prevLayer)
{
	Neuron* neuron;
	double oldDeltaWeight;
	double newDeltaWeight;
	for (unsigned n = 0; n < prevLayer.size(); n++)
	{
		neuron = &prevLayer[n];
		oldDeltaWeight = neuron->m_outputWeights[m_myIndex].deltaWeight;
		newDeltaWeight =
			(eta
			* neuron->getOutputVal()
			* m_gradient)
			+ (alpha
			* oldDeltaWeight);
		
		neuron->m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
		neuron->m_outputWeights[m_myIndex].weight += newDeltaWeight;


	}
}
