#pragma once

#include "framework.hpp"
#include "nlohmann/json.hpp"
#include "utils/MOA.hpp"
#include <cstdint>
#include <vector>

using namespace neuro;
using namespace std;
namespace crisp {
class Network;
class Processor;

struct compiled_vtable {
    void (*compiled_apply_spike)(size_t input_id, size_t time, double value);
    size_t (*compiled_output_count)(size_t output_id);
    ssize_t (*compiled_output_last_fire)(size_t output_id);
    double (*compiled_neuron_charge)(size_t neuron_id);
    void (*compiled_run)(size_t timesteps);
    void (*compiled_clear_output_tracking)();
    void (*compiled_clear_activity)();
};

class Network {
  public:
    /** Convert network in framework format to an internal crisp network */

    Network(neuro::Network* net, double _min_potential, char leak,
            size_t tracked_timesteps, double spike_value_factor, bool discrete);
    ~Network();

    /* Mirror calls from the Processor API */
    void apply_spike(const Spike& s, bool normalized = true);
    void run(size_t duration);
    double get_time();

    double output_last_fire(int output_id);
    vector<double> output_last_fires();

    int output_count(int output_id);
    vector<int> output_counts();

    vector<double> output_vector(int output_id);
    vector<vector<double>> output_vectors();

    long long total_neuron_counts();
    long long total_neuron_accumulates();
    vector<int> neuron_counts();
    vector<double> neuron_last_fires();
    vector<vector<double>> neuron_vectors();

    vector<double> neuron_charges();
    void synapse_weights(vector<uint32_t>& pres, vector<uint32_t>& posts,
                         vector<double>& vals);

    void clear_activity();
    void clear_output_tracking();

  protected:
    vector<bool> inputs;
    vector<bool> outputs;

    vector<size_t> input_mappings;
    vector<size_t> output_mappings;
    vector<size_t> neuron_mappings;

    size_t neuron_count;
    double spike_value_factor;

    size_t num_inputs;
    size_t num_outputs;
    size_t tracked_timesteps;
    size_t* timestep;

    compiled_vtable vtable;
    void* dlhandle = nullptr;

    string c_file;
    string so_file;
};

class Processor : public neuro::Processor {
  public:
    Processor(json& params);
    ~Processor();

    bool load_network(neuro::Network* n, int network_id = 0);
    bool load_networks(std::vector<neuro::Network*>& n);
    void clear(int network_id = 0);

    /* Queue spike(s) as input to a network or to multiple networks */

    void apply_spike(const Spike& s, bool normalized = true,
                     int network_id = 0);
    void apply_spike(const Spike& s, const vector<int>& network_ids,
                     bool normalized = true);

    void apply_spikes(const vector<Spike>& s, bool normalized = true,
                      int network_id = 0);
    void apply_spikes(const vector<Spike>& s, const vector<int>& network_ids,
                      bool normalized = true);

    /* Run the network(s) for the desired time with queued inputs */

    void run(double duration, int network_id = 0);
    void run(double duration, const vector<int>& network_ids);

    /* Get processor time based on specified network */
    double get_time(int network_id = 0);

    /* Output tracking. */
    bool track_output_events(int output_id, bool track = true,
                             int network_id = 0);
    bool track_neuron_events(uint32_t node_id, bool track = true,
                             int network_id = 0);

    /* Access output spike data */
    double output_last_fire(int output_id, int network_id = 0);
    vector<double> output_last_fires(int network_id = 0);

    int output_count(int output_id, int network_id = 0);
    vector<int> output_counts(int network_id = 0);

    vector<double> output_vector(int output_id, int network_id = 0);
    vector<vector<double>> output_vectors(int network_id = 0);

    /* Spike data from all neurons. */

    long long total_neuron_counts(int network_id = 0);
    long long total_neuron_accumulates(int network_id = 0);
    vector<int> neuron_counts(int network_id = 0);
    vector<double> neuron_last_fires(int network_id = 0);
    vector<vector<double>> neuron_vectors(int network_id = 0);

    vector<double> neuron_charges(int network_id = 0);

    void synapse_weights(vector<uint32_t>& pres, vector<uint32_t>& posts,
                         vector<double>& vals, int network_id = 0);

    /* Remove state, keep network loaded */
    void clear_activity(int network_id = 0);

    /* Network/Processor Properties */
    PropertyPack get_network_properties() const;
    json get_processor_properties() const;
    json get_params() const;
    string get_name() const;

  protected:
    crisp::Network* get_crisp_network(int network_id);
    map<int, crisp::Network*> networks;

    double min_weight;
    double max_weight;
    double min_threshold;
    double max_threshold;
    double min_potential;
    string leak_mode;
    double spike_value_factor;
    size_t tracked_timesteps_count;
    bool discrete;

    uint32_t min_delay;
    uint32_t max_delay;

    json saved_params;
};
} // namespace crisp
