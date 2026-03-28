#pragma once
#include <vector>
#include <utility>
#include <string>
#include <random>
#include "Evaluator.hpp"

namespace LcVRPContest {

    class CIndividual
    {
    public:
        static constexpr double INF_FITNESS = 1e15;

        CIndividual() : d_fitness(INF_FITNESS) {}
        CIndividual(int iGenotypeSize, int iNumberOfGroups, std::mt19937& cRng);
        CIndividual(const std::vector<int>& vNewGenotype);
        CIndividual(const CIndividual& cOther);
        CIndividual(CIndividual&& cOther);

        CIndividual& operator=(const CIndividual& cOther);
        CIndividual& operator=(CIndividual&& cOther);

        std::pair<CIndividual, CIndividual> pCrossover(const CIndividual& cOtherParent, double dCrossProb, std::mt19937& cRng) const;
        void vMutate(double dMutProb, int iNumberOfGroups, std::mt19937& cRng);

        double dGetFitness() const;
        void vSetFitness(double dNewFitness);


        double dCalculateFitness(const Evaluator& evaluator);

        const std::vector<int>& viGetGenotype() const;
        std::string sToString() const;

        int iGetGenotypeSize() const { return (int)v_genotype.size(); }
        void vSetGene(int iIndex, int iValue);

    private:
        std::vector<int> v_genotype;
        double d_fitness;
    };

}