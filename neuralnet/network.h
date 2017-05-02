#pragma once
#include <vector>
#include "neuron.h"
#include "netdefines.h"

namespace neuralNet
{
	class Network
	{
		public:
			Network(const std::vector<unsigned>& topology);
			void feedForward(const std::vector<net_type> &inputVals);
			void backProp(const std::vector<net_type> &targetVals);
			void getResoults(std::vector<net_type>& resultsVals) const;
			double getRecentAverageError() const { return m_recentAverageError; }

		public:
			std::vector<Layer> m_layers;
			double m_error;
			double m_recentAverageSmoothingFactor;
			double m_recentAverageError;
	};
}
