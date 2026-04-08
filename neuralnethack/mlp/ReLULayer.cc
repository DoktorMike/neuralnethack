#include "ReLULayer.hh"
using namespace MultiLayerPerceptron;
ReLULayer::ReLULayer(uint nc, uint np) : Layer(nc, np, RELU) {}
ReLULayer::~ReLULayer() {}
