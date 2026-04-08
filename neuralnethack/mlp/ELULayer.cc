#include "ELULayer.hh"
using namespace MultiLayerPerceptron;
ELULayer::ELULayer(uint nc, uint np) : Layer(nc, np, ELU_ACT) {}
ELULayer::~ELULayer() {}
