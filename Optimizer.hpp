#pragma once

#include "Evaluator.hpp"
#include "CIndividual.hpp" 
#include <vector>
#include <random>

using namespace std;

namespace LcVRPContest {
	class Optimizer {
	public:
		//DO NOT CHANGE THIS
		Optimizer(Evaluator& evaluator);

		//this helps me 
		~Optimizer();


		// THEESE METHODS HAVE TO EXIST
		void Initialize();
		void RunIteration();

		// DO NOT CHANGE THIS
		vector<int>* GetCurrentBest() { return &current_best_; }


		double GetCurrentBestFitness() const { return current_best_fitness_; }

	private:
		// REQUIRED FIELDS DO NOT CHANGE
		Evaluator& evaluator_;
		vector<int> current_best_;
		double current_best_fitness_;



		// my stuff
		mt19937 rng_;

		//counters
		long long iteration_counter_;
		int stagnation_counter_;

		//constants for ga
		const int POPULATION_SIZE = 300;
		const double CROSSOVER_PROB = 0.8;
		const double MUTATION_PROB = 0.01;
		const int TOURNAMENT_SIZE = 2;
		const int STAGNATION_THRESHOLD = 80;
		const double HYPER_MUTATION_PROB = 0.6;
		const int TO_OPTIMIZE = 4;
		const int LS_LIMIT = 5;

		const double MAX_FITNESS_VAL = 1e15;
		const double MUTATION_RATIO = 0.5;
		const int RANDOM_CANDIDATES = 3;
		const int CATACLYSM_DIVISOR = 3;
		const double FRESH_MUT_PROB = 0.1;


		vector<CIndividual> population_;

		void vEvaluatePopulationParallel();
		const CIndividual& cSelectParentTournament(int tournament_size);


		void vRunLocalSearch(CIndividual& individual);
		void vRunCataclysm();
		void vUpdateGlobalBest();

		void vSaveToFile(const std::string& sFileName);


	};
}