#include "CIndividual.hpp"
#include "Evaluator.hpp"
#include <iostream>
#include <sstream>
#include <string>
#include <utility>

using namespace LcVRPContest;



CIndividual::CIndividual(int iGenotypeSize, int iNumberOfGroups, std::mt19937& cRng) {

    d_fitness = INF_FITNESS;
    v_genotype.reserve(iGenotypeSize);

    std::uniform_int_distribution<int> dist(0, iNumberOfGroups - 1);

    for (int i = 0; i < iGenotypeSize; i++) {
        v_genotype.push_back(dist(cRng));
    }
}

CIndividual::CIndividual(const std::vector<int>& vNewGenotype) {
    d_fitness = INF_FITNESS;
    v_genotype = vNewGenotype;
}

CIndividual::CIndividual(const CIndividual& cOther) {
    d_fitness = cOther.d_fitness;
    v_genotype = cOther.v_genotype;
}


CIndividual::CIndividual(CIndividual&& cOther) {
    v_genotype = std::move(cOther.v_genotype);
    d_fitness = cOther.d_fitness;
    cOther.d_fitness = INF_FITNESS;
}

CIndividual& CIndividual::operator=(const CIndividual& cOther) {
    if (this != &cOther) {
        v_genotype = cOther.v_genotype;
        d_fitness = cOther.d_fitness;
    }
    return *this;
}

CIndividual& CIndividual::operator=(CIndividual&& cOther) {
    if (this != &cOther) {
        v_genotype = std::move(cOther.v_genotype);
        d_fitness = cOther.d_fitness;
        cOther.d_fitness = INF_FITNESS;
    }
    return *this;
}

std::pair<CIndividual, CIndividual> CIndividual::pCrossover(const CIndividual& cOtherParent, double dCrossProb, std::mt19937& cRng) const {
    std::uniform_real_distribution<double> distProb(0.0, 1.0);

    if (distProb(cRng) > dCrossProb) {

        return std::make_pair(*this, cOtherParent);
    }

    int iSize = v_genotype.size();
    if (iSize < 2) return std::make_pair(*this, cOtherParent);

    std::uniform_int_distribution<int> distSplit(1, iSize - 1);
    int iSplitPoint = distSplit(cRng);

    std::vector<int> v_child1Proto;
    std::vector<int> v_child2Proto;
    v_child1Proto.reserve(iSize);
    v_child2Proto.reserve(iSize);

    for (int i = 0; i < iSplitPoint; i++) {
        v_child1Proto.push_back(this->v_genotype[i]);
        v_child2Proto.push_back(cOtherParent.v_genotype[i]);
    }
    for (int i = iSplitPoint; i < iSize; i++) {
        v_child1Proto.push_back(cOtherParent.v_genotype[i]);
        v_child2Proto.push_back(this->v_genotype[i]);
    }

    return std::make_pair(CIndividual(v_child1Proto), CIndividual(v_child2Proto));
}

void CIndividual::vMutate(double dMutProb, int iNumberOfGroups, std::mt19937& cRng) {
    std::uniform_real_distribution<double> distProb(0.0, 1.0);
    std::uniform_int_distribution<int> distGroup(0, iNumberOfGroups - 1);

    for (size_t i = 0; i < v_genotype.size(); i++) {
        if (distProb(cRng) < dMutProb) {
            v_genotype[i] = distGroup(cRng);
        }
    }
}

double CIndividual::dGetFitness() const {
    return d_fitness;
}

void CIndividual::vSetFitness(double dNewFitness) {
    d_fitness = dNewFitness;
}

const std::vector<int>& CIndividual::viGetGenotype() const {
    return v_genotype;
}

std::string CIndividual::sToString() const {
    std::stringstream ss;
    ss << "Fitness: " << d_fitness;
    return ss.str();
}

double CIndividual::dCalculateFitness(const Evaluator& evaluator) {
    d_fitness = evaluator.Evaluate(v_genotype);
    return d_fitness;
}

void CIndividual::vSetGene(int iIndex, int iValue) {
    if (iIndex >= 0 && iIndex < (int)v_genotype.size()) {
        v_genotype[iIndex] = iValue;
    }
}