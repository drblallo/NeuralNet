#pragma once
#include "netdefines.h"
#include <vector>

namespace neuralNet
{

	class Neuron
	{
		public:
			Neuron(unsigned numOutputs, unsigned index);
			void feedForward(const Layer* prevLayer);
			inline void setOutputVal(net_type val) { m_outputValue = val; }
			inline net_type getOutputVal() const { return m_outputValue; }
			void calcOutputGradients(net_type targetVal);
			void calcHiddenGradients(const Layer& nextLayer);
			void updateInputWeights(Layer& prevLayer);

		private:
			static double eta;
			static double alpha;
			static net_type transferFunction(net_type x);
			static net_type transferFunctionDerivative(net_type x);
			double sumDOW(const Layer &nextLayer) const;
			net_type m_outputValue;
			unsigned m_myIndex;
			std::vector<Connection> m_outputWeights;
			double m_gradient;			
	};
}
