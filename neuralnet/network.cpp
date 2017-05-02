#include "network.h"
#include "neuron.h"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace neuralNet;

Network::Network(const std::vector<unsigned>& topology)
	: m_recentAverageSmoothingFactor(0)
{
	unsigned numLayers = topology.size();
	
	for (unsigned layerNum = 0; layerNum < numLayers; layerNum++)
	{
		m_layers.push_back(Layer());
		unsigned numOutputs(0);
		if (layerNum != topology.size() - 1)
			numOutputs = topology[layerNum + 1];


		for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; neuronNum++)
		{
			m_layers.back().push_back(Neuron(numOutputs, neuronNum));
			std::cout << "created a neuron" << std::endl;
		}

		m_layers.back().back().setOutputVal(1);
		std::cout << m_layers.size() << " neurons" << std::endl;
	}
}

void Network::feedForward(const std::vector<net_type> &inputVals)
{
	assert(inputVals.size() == m_layers.size() - 1);

	for (unsigned i = 0; i < inputVals.size(); i++)
		m_layers[0][i].setOutputVal(inputVals[i]);

	Layer* prevLayer;
	for (unsigned layerNum = 1; layerNum < m_layers.size(); layerNum++)
	{
		prevLayer = &m_layers[layerNum - 1];
		for (unsigned n = 0; n < m_layers[layerNum].size() - 1; n++)	
			m_layers[layerNum][n].feedForward(prevLayer);
	}

}

void Network::backProp(const std::vector<net_type> &targetVales)
{
	//calculate overall net error(RMS)
	Layer &outputLayer(m_layers.back());	
	m_error = 0;
	
	double delta;
	for (unsigned n = 0; n < outputLayer.size() - 1; n++)
	{
		delta = targetVales[n] - outputLayer[n].getOutputVal();
		m_error += delta * delta;
	}
	m_error /= outputLayer.size()-1;
	m_error = std::sqrt(m_error);

	//impelement a recent average mesurment
	
	m_recentAverageError = 
		(m_recentAverageError * m_recentAverageSmoothingFactor + m_error)
		/ (m_recentAverageSmoothingFactor + 1.0);

	//calculate output layer gradients
	
	for (unsigned n = 0; n < outputLayer.size() - 1; n++)
	{
		outputLayer[n].calcOutputGradients(targetVales[n]);
	}
	
	//calculate gradients on hidden layer
	
	for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; layerNum--)
	{
		Layer &hiddenLayer(m_layers[layerNum]);
		Layer &nextLayer(m_layers[layerNum + 1]);

		for (unsigned n = 0; n < hiddenLayer.size(); n++)
		{
			hiddenLayer[n].calcHiddenGradients(nextLayer);
		}
	}
	
	//update connections weight
	
	for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; layerNum--)
	{	
		Layer &layer(m_layers[layerNum]);
		Layer &prevLayer(m_layers[layerNum - 1]);

		for (unsigned n = 0; n < layer.size() - 1; n++)
		{
			layer[n].updateInputWeights(prevLayer);
		}
	}
}

void Network::getResoults(std::vector<net_type>& resultsVals) const
{
		resultsVals.clear();

		for (unsigned n  = 0; n < m_layers.back().size() - 1; n++)
		{
			resultsVals.push_back(m_layers.back()[n].getOutputVal());
		}
}
