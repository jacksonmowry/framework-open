#include "crisp.hpp"
#include "framework.hpp"

neuro::Processor* neuro::Processor::make(const string& name, json& params) {
    string error_string;

    if (name != "vrisp" && name != "risp" && name != "crisp") {
        error_string = (string) "Processor::make() called with a name (" +
                       name + (string) ") not equal to crisp|risp|vrisp";
        throw std::runtime_error(error_string);
    }

    return new crisp::Processor(params);
}
