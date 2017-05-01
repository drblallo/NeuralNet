#include <iostream>
#include "neuralnet/network.h" 
#include <vector> 
#include <random>
#include <iomanip>

using namespace neuralNet;

void train(Network &net)
{
	std::default_random_engine engine;
	std::uniform_int_distribution<int> dist(0,1);
	std::cout.precision(4);
	for (int a = 0; a < 2000; a++)
	{
			
		std::vector<net_type> inputVals;
		inputVals.push_back(dist(engine));
		inputVals.push_back(dist(engine));
		net.feedForward(inputVals);

		std::vector<net_type> resultsVals;
		net.getResoults(resultsVals);
		
		std::vector<net_type> targetVales;
		targetVales.push_back(inputVals[0] * inputVals[1]);
		net.backProp(targetVales);


		std::cout << "iteration: " << a << std::endl;
		std::cout << "feed: " << inputVals[0] << ", " << inputVals[1] << std::endl;
		std::cout << "expected: " <<  (targetVales[0]) << ", got: " << resultsVals[0] << std::endl;
	}

}

int main()
{
	std::vector<unsigned> topology;
	topology.push_back(2);
	topology.push_back(4);
	topology.push_back(1);
	Network net(topology);

	train(net);
		
	std::cout << "done. Exiting" << std::endl;
	return 0;
}
