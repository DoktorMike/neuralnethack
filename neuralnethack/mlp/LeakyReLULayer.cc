#include "LeakyReLULayer.hh"
using namespace MultiLayerPerceptron;
LeakyReLULayer::LeakyReLULayer(uint nc, uint np) : Layer(nc, np, LEAKYRELU) {}
LeakyReLULayer::~LeakyReLULayer() {}
