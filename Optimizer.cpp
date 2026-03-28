#include "Optimizer.hpp"
#include <iostream>
#include <limits>
#include <algorithm>
#include <thread>
#include <future>
#include <fstream>
#include <numeric>
//#define DEBUG
using namespace LcVRPContest;

Optimizer::Optimizer(Evaluator& evaluator)
    : evaluator_(evaluator),
    rng_(random_device{}()),
    current_best_fitness_(numeric_limits<double>::max()) {
}

Optimizer::~Optimizer() {
    vSaveToFile("solution.txt");
}

void Optimizer::Initialize() {
#ifdef DEBUG
    cout << ">>> SMART INITIALIZATION <<<" << endl;

#endif // DEBUG


    population_.clear();
    population_.reserve(POPULATION_SIZE);

    int genotype_size = evaluator_.GetSolutionSize(); //client count
    int num_groups = evaluator_.GetNumGroups(); //truck count


    //first half of the pop
    for (int i = 0; i < POPULATION_SIZE / 2; ++i) {
        vector<int> genotype(genotype_size);
        for (int k = 0; k < genotype_size; ++k) {
            genotype[k] = (k * num_groups) / genotype_size;
        }
        //thess are likely next to each other physically? so i assign with block like -> 0,0,0,0,1,1,1,1, etc.
        population_.emplace_back(genotype);
    }
    //secomd half - what if physical neighbors are actually far away -> 0,1,2,0,1,2 sth like that
    for (int i = POPULATION_SIZE / 2; i < POPULATION_SIZE; ++i) {
        vector<int> genotype(genotype_size);
        for (int k = 0; k < genotype_size; ++k) {
            genotype[k] = k % num_groups;
        }
        population_.emplace_back(genotype);
    }

    vEvaluatePopulationParallel();
    vUpdateGlobalBest();

    iteration_counter_ = 0;
    stagnation_counter_ = 0;

#ifdef DEBUG
    cout << ">>> INIT BEST: " << current_best_fitness_ << endl;
#endif // DEBUG

}

void Optimizer::RunIteration() {
    iteration_counter_++;
#ifdef DEBUG
    if (iteration_counter_ % 10 == 0) {
        cout << "Iter: " << iteration_counter_
            << " | Best: " << current_best_fitness_
            << " | Stagnation: " << stagnation_counter_ << endl;
    }
#endif // DEBUG


    //if no progress after k iterations -> run cataclysm
    if (stagnation_counter_ > STAGNATION_THRESHOLD) {

#ifdef DEBUG
        cout << ">>> CATACLYSM (Smart Restart) <<<" << endl;
#endif // DEBUG

        vRunCataclysm();
    }

    vector<CIndividual> next_population;
    next_population.reserve(POPULATION_SIZE);


    //elitism 
    if (!population_.empty()) {

        //find the iterator showim nte best one 
        auto it = std::min_element(population_.begin(), population_.end(),
            [this](const CIndividual& a, const CIndividual& b) {
                double fA = a.dGetFitness() > 0 ? a.dGetFitness() : MAX_FITNESS_VAL;
                double fB = b.dGetFitness() > 0 ? b.dGetFitness() : MAX_FITNESS_VAL;
                return fA < fB;
            });
        next_population.push_back(*it);
    }

    //until the next_population is full cross bread
    while (next_population.size() < POPULATION_SIZE) {
        //call tournament sellection of given size
        const CIndividual& p1 = cSelectParentTournament(TOURNAMENT_SIZE); //Lebron
        const CIndividual& p2 = cSelectParentTournament(TOURNAMENT_SIZE); //Savannah  

        pair<CIndividual, CIndividual> children = p1.pCrossover(p2, CROSSOVER_PROB, rng_);  //Bronny and Bryce 


        //two mutations
        if (std::bernoulli_distribution(MUTATION_RATIO)(rng_)) {
            //classsic
            children.first.vMutate(MUTATION_PROB, evaluator_.GetNumGroups(), rng_);
            children.second.vMutate(MUTATION_PROB, evaluator_.GetNumGroups(), rng_);
        }
        else {
            //heuristic
            //assuming that ID 5 is next to ID 4 and ID 6 - if client 5 is in a
            // different truck than 4 than maybe it's worth to move im to the group of 4

            auto SmartMutate = [&](CIndividual& ind) {
                std::uniform_int_distribution<int> geneDist(1, ind.iGetGenotypeSize() - 2);
                std::uniform_real_distribution<double> probDist(0.0, 1.0);

                for (int i = 0; i < ind.iGetGenotypeSize(); ++i) {
                    if (probDist(rng_) < MUTATION_PROB) {
                        int neighbor = (i > 0) ? i - 1 : i + 1;
                        if (neighbor < ind.iGetGenotypeSize()) {
                            ind.vSetGene(i, ind.viGetGenotype()[neighbor]);
                        }
                    }
                }
                };
            SmartMutate(children.first);
            SmartMutate(children.second);
        }

        next_population.push_back(std::move(children.first));
        if (next_population.size() < POPULATION_SIZE) {
            next_population.push_back(std::move(children.second));
        }
    }

    population_ = std::move(next_population);
    vEvaluatePopulationParallel();

    if (iteration_counter_ > 0 && iteration_counter_ % 10 == 0) {

        std::sort(population_.begin(), population_.end(),
            [this](const CIndividual& a, const CIndividual& b) {
                double fA = a.dGetFitness() > 0 ? a.dGetFitness() : MAX_FITNESS_VAL;
                double fB = b.dGetFitness() > 0 ? b.dGetFitness() : MAX_FITNESS_VAL;
                return fA < fB;
            });


        // run e.g 4 local searches at the same time
        int num_to_optimize = std::min((int)population_.size(), TO_OPTIMIZE);
        vector<thread> ls_threads;

        for (int i = 0; i < num_to_optimize; ++i) {
            ls_threads.emplace_back([this, i]() {
                this->vRunLocalSearch(population_[i]);
                });
        }

        for (auto& t : ls_threads) t.join();
    }

    vUpdateGlobalBest();
}

void Optimizer::vRunLocalSearch(CIndividual& individual) {

    std::mt19937 local_rng(std::random_device{}());
    int iGenotypeSize = individual.iGetGenotypeSize();
    double dCurrentFit = individual.dGetFitness();
    if (dCurrentFit < 0) dCurrentFit = MAX_FITNESS_VAL;


    bool bImprovement = true;
    int iLoops = 0;
    vector<int> indices(iGenotypeSize);
    iota(indices.begin(), indices.end(), 0);


    //work till improvement but no more than 
    while (bImprovement && iLoops < LS_LIMIT) {
        bImprovement = false;
        iLoops++;
        shuffle(indices.begin(), indices.end(), local_rng); //this gives us a chacne to find the improvement quicker

        for (int i : indices) {
            int currentGroup = individual.viGetGenotype()[i];

            vector<int> candidateGroups;

            if (i > 0) candidateGroups.push_back(individual.viGetGenotype()[i - 1]);
            if (i < iGenotypeSize - 1) candidateGroups.push_back(individual.viGetGenotype()[i + 1]);
            for (int k = 0; k < RANDOM_CANDIDATES; ++k) candidateGroups.push_back(std::uniform_int_distribution<int>(0, evaluator_.GetNumGroups() - 1)(local_rng));

            for (int targetGroup : candidateGroups) {
                if (targetGroup != currentGroup) {

                    individual.vSetGene(i, targetGroup);
                    double dMoveFit = individual.dCalculateFitness(evaluator_);
                    //move and recalculate
                    if (dMoveFit > 0 && dMoveFit < dCurrentFit) {
                        dCurrentFit = dMoveFit;
                        bImprovement = true;
                        currentGroup = targetGroup;  //the client is in a new group!
                    }
                    else {
                        //backtrack
                        individual.vSetGene(i, currentGroup);
                    }
                }

            }
        }
    }
}

void Optimizer::vRunCataclysm() {
    //save the LEBRON JAMES
    CIndividual best_indiv;
    if (!population_.empty() && !current_best_.empty()) {
        best_indiv = CIndividual(current_best_);
        best_indiv.vSetFitness(current_best_fitness_);
    }

    //kill everybody 
    population_.clear();
    population_.push_back(best_indiv); // king james makes it thru becasue he's the goat

    int genotype_size = evaluator_.GetSolutionSize();
    int num_groups = evaluator_.GetNumGroups();

    //repopulate
    while (population_.size() < POPULATION_SIZE) {
        //1/3 is a mutated lebron - maybe bronny will be better?
        if (population_.size() < POPULATION_SIZE / CATACLYSM_DIVISOR) {
            CIndividual clone = best_indiv;
            clone.vMutate(HYPER_MUTATION_PROB, num_groups, rng_);
            population_.push_back(std::move(clone));
        }
        else {
            //2/3 are rookies - brand new
            vector<int> genotype(genotype_size);
            for (int k = 0; k < genotype_size; ++k) {
                genotype[k] = (k * num_groups) / genotype_size; // Reset do bloków
            }
            CIndividual fresh(genotype);
            fresh.vMutate(FRESH_MUT_PROB, num_groups, rng_); //slightlu mutate
            population_.push_back(std::move(fresh));
        }
    }

    stagnation_counter_ = 0;
    vEvaluatePopulationParallel();
}

void Optimizer::vUpdateGlobalBest() {

    bool improvement = false;
    for (const auto& indiv : population_) {
        double fit = indiv.dGetFitness();
        if (fit > 0 && fit < current_best_fitness_) {
            current_best_fitness_ = fit;
            current_best_ = indiv.viGetGenotype();
            improvement = true;
        }
    }

    if (improvement) stagnation_counter_ = 0;
    else stagnation_counter_++;
}

void Optimizer::vEvaluatePopulationParallel() {

    //check for count of proccessors
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 2;

    //no larger than individuals
    num_threads = std::min((unsigned int)population_.size(), num_threads);

    vector<thread> threads;

    //what to do for the worker
    auto worker = [&](int start_idx, int end_idx) {
        for (int i = start_idx; i < end_idx; ++i) {
            population_[i].dCalculateFitness(evaluator_);
        }
        };

    //chunking - even chunks
    int chunk = population_.size() / num_threads;
    int rem = population_.size() % num_threads;
    int start = 0;

    //start the threads
    for (int i = 0; i < num_threads; ++i) {

        int end = start + chunk + (i < rem ? 1 : 0); //add the rem for the firts threads
        //create and start
        threads.emplace_back(worker, start, end);
        start = end;
    }

    //join
    for (auto& t : threads) if (t.joinable()) t.join();
}
//play offs - who gets further?
const CIndividual& Optimizer::cSelectParentTournament(int tournament_size) {

    std::uniform_int_distribution<int> dist(0, population_.size() - 1);
    //we assume a mvp but we dont know
    int best_idx = dist(rng_);
    double best_fit = population_[best_idx].dGetFitness();

    for (int i = 1; i < tournament_size; ++i) {
        //fresh blood - might be better than the current mvp
        int idx = dist(rng_);
        double fit = population_[idx].dGetFitness();
        if (fit > 0 && (best_fit < 0 || fit < best_fit)) {
            best_idx = idx;
            best_fit = fit; //new mvp
        }
    }
    return population_[best_idx];
}

void Optimizer::vSaveToFile(const std::string& sFileName) {
    if (current_best_.empty()) return;
    std::ofstream f(sFileName);
    if (f.is_open()) {
        for (size_t i = 0; i < current_best_.size(); i++) {
            f << current_best_[i] << (i == current_best_.size() - 1 ? "" : " ");
        }
        f << "\n";
        f.close();
        std::cout << "Saved best solution to " << sFileName << std::endl;
    }
}