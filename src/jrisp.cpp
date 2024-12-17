#include "jrisp.hpp"
#include "framework.hpp"
#include "utils/alignment_helpers.hpp"
#include "utils/json_helpers.hpp"
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#define RISCVV
#ifdef AVX
#include <immintrin.h>
#endif
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
#ifdef RISCVV
#include <riscv_vector.h>
#endif

typedef std::runtime_error SRE;
using namespace std;

namespace jrisp {
/** Configurable settings for jrisp */
static json jrisp_spec = {
    {"min_weight", "I"},
    {"max_weight", "I"},
    {"max_delay", "I"},
    {"min_threshold", "I"},
    {"max_threshold", "I"},
    {"min_potential", "I"},
    {"tracked_timesteps", "I"},
    {"leak_mode", "S"},
    {"spike_value_factor", "D"},
    {"Necessary",
     {"max_delay", "min_threshold", "max_threshold", "min_potential",
      "tracked_timesteps"}},
};

static bool is_integer(double v) {
    int iv;

    iv = v;
    return (iv == v);
}

Network::Network(neuro::Network* net, double _min_potential, char leak,
                 size_t tracked_timesteps, double _spike_value_factor) {
    leak_mode = leak;
    spike_value_factor = _spike_value_factor;
    tracked_timesteps_count = tracked_timesteps;

    min_potential = _min_potential;
    current_timestep = 0;

    net->make_sorted_node_vector();

    neuron_count = net->sorted_node_vector.back()->id + 1;
    size_t allocation_size =
        ((neuron_count + 31) / 32) *
        32; // JDM Instead of messing with masking load/stores we can just round
            // up to a multiple of 32

    inputs.resize(allocation_size);
    outputs.resize(allocation_size);

    neuron_fired.resize(allocation_size);
    output_fire_count.resize(allocation_size, 0);
    output_last_fire_timestep.resize(allocation_size, -1);
    outgoing_synapse_count.resize(allocation_size);
    neuron_threshold.resize(allocation_size, INT8_MAX);
    synapse_to.resize(allocation_size);
    synapse_delay.resize(allocation_size);
    synapse_weight.resize(allocation_size);
    neuron_charge_buffer = (int8_t*)aligned_alloc(
        32, (tracked_timesteps_count * neuron_count + 31) / 32 * 32);
    memset(neuron_charge_buffer, 0,
           (tracked_timesteps_count * neuron_count + 31) / 32 * 32);
    neuron_leak.resize(allocation_size / 8);

    /* Add neurons */
    for (size_t i = 0; i < net->sorted_node_vector.size(); i++) {
        neuro::Node* node = net->sorted_node_vector[i];

        neuron_mappings.push_back(node->id);

        if (leak_mode == 'c') {
            neuron_leak[node->id / 8] =
                (1 << (node->id % 8)) & (node->get("Leak") != 0);
        } else {
            neuron_leak[node->id / 8] =
                (1 << (node->id % 8)) & (leak_mode == 'a');
        }

        neuron_threshold[node->id] = node->get("Threshold");

        if (node->is_input()) {
            inputs[node->id] = true;
        }

        if (node->is_output()) {
            outputs[node->id] = true;
        }
    }

    for (int i = 0; i < net->num_inputs(); i++) {
        input_mappings.push_back(net->get_input(i)->id);
    }

    for (int i = 0; i < net->num_outputs(); i++) {
        output_mappings.push_back(net->get_output(i)->id);
    }

    /* Add synapses */
    for (EdgeMap::iterator eit = net->edges_begin(); eit != net->edges_end();
         ++eit) {
        neuro::Edge* edge = eit->second.get();

        outgoing_synapse_count[edge->from->id]++;
        synapse_to[edge->from->id].push_back(edge->to->id);
        synapse_delay[edge->from->id].push_back(edge->get("Delay"));
        synapse_weight[edge->from->id].push_back(edge->get("Weight"));
    }
}

Network::~Network() { free(neuron_charge_buffer); }

void Network::apply_spike(const Spike& s, bool normalized) {
    if (!normalized && !is_integer(s.value)) {
        throw SRE("jrisp::Network::apply_spike() only supports integer spike "
                  "values - value (" +
                  to_string(s.value) + ") is not valid.");
    }

    if (normalized && (s.value < -1 || s.value > 1)) {
        throw SRE("jrisp::Network::apply_spike() - value (" +
                  to_string(s.value) + ") must be in [-1,1].");
    }

    if ((size_t)s.time >= tracked_timesteps_count) {
        throw SRE("jrisp::Network::apply_spike - time (" +
                  to_string((size_t)s.time) +
                  ") must be < tracked_timesteps_count (" +
                  to_string(tracked_timesteps_count) + ")");
    }

    uint8_t spike_value = (normalized) ? s.value * spike_value_factor : s.value;
    neuron_charge_buffer[((current_timestep + (size_t)s.time) %
                          tracked_timesteps_count) *
                             neuron_count +
                         input_mappings[s.id]] += spike_value;
}

void Network::run(size_t duration) {
    if (current_timestep != 0) {
        clear_output_tracking();
    }

    for (size_t i = 0; i < duration; i++) {
        process_events(i);
    }

    current_timestep += duration;

    for (size_t i = 0; i < neuron_count; i++) {
        if (neuron_charge_buffer[(current_timestep % tracked_timesteps_count) *
                                     neuron_count +
                                 +i] < min_potential) {
            neuron_charge_buffer[(current_timestep % tracked_timesteps_count) *
                                     neuron_count +
                                 i] = min_potential;
        }
    }
}

void Network::process_events(uint32_t time) {
    size_t internal_timestep =
        (current_timestep + time) % tracked_timesteps_count;

    fill(neuron_fired.begin(), neuron_fired.end(), false);

#ifdef NO_SIMD
    for (size_t i = 0; i < neuron_charge_buffer[internal_timestep].size();
         i++) {
        if (neuron_charge_buffer[internal_timestep][i] < min_potential) {
            neuron_charge_buffer[internal_timestep][i] = min_potential;
        }
        if (neuron_charge_buffer[internal_timestep][i] >= neuron_threshold[i]) {
            for (size_t j = 0; j < synapse_to[i].size(); j++) {
                neuron_charge_buffer[(internal_timestep + synapse_delay[i][j]) %
                                     tracked_timesteps_count]
                                    [synapse_to[i][j]] += synapse_weight[i][j];
            }

            neuron_fired[i] = true;
            // If a neuron fires it always clears to zero
            neuron_charge_buffer[internal_timestep][i] = 0;

            // Track output count and last fire time
            if (outputs[i]) {
                output_last_fire_timestep[i] = time;
                output_fire_count[i] =
                    output_fire_count[i] == -1 ? 1 : output_fire_count[i] + 1;
            }
        } else {
            if (!(neuron_leak[i / 8]) >> (i % 8) & 1) {
                // If we don't leak we carry this charge over into the next
                // timestep
                // printf("Not leaking, so I add %d charge to the next timestep,
                // resulting in")
                neuron_charge_buffer[(internal_timestep + 1) %
                                     tracked_timesteps_count][i] +=
                    neuron_charge_buffer[internal_timestep][i];
                neuron_charge_buffer[internal_timestep][i] = 0;
            } else {
                // Otherwise reset charge to zero
                neuron_charge_buffer[internal_timestep][i] = 0;
            }
        }
    }
#endif // NO_SIMD
#ifdef AVX
    __m256i ones = _mm256_set1_epi8(1);
    for (size_t i = 0; i < neuron_count; i += 32) {
        __m256i charges = _mm256_load_si256(
            (__m256i*)&neuron_charge_buffer[internal_timestep][i]);
        // Bring up any charges that are less than min_potential
        __m256i min_potential_vec = _mm256_set1_epi8(min_potential);
        charges = _mm256_max_epi8(charges, min_potential_vec);

        __m256i threshold = _mm256_load_si256((__m256i*)&neuron_threshold[i]);
        threshold =
            _mm256_sub_epi8(threshold,
                            ones); // There isn't a cmpge instruction on
                                   // most CPUs (Only suported on AVX-512)

        __m256i fired = _mm256_cmpgt_epi8(charges, threshold);

        bool values[32] __attribute__((aligned(32)));
        _mm256_store_si256((__m256i*)values, fired);

        for (int k = 0; k < 32 && (i + k) < neuron_count; k++) {
            if (values[k]) {
                for (size_t j = 0; j < synapse_to[i + k].size(); j++) {
                    neuron_charge_buffer[(internal_timestep +
                                          synapse_delay[i + k][j]) %
                                         tracked_timesteps_count]
                                        [synapse_to[i + k][j]] +=
                        synapse_weight[i + k][j];
                }
                neuron_fired[i + k] = true;
                if (outputs[i + k]) {
                    output_last_fire_timestep[i + k] = time;
                    output_fire_count[i + k] =
                        output_fire_count[i + k] == -1
                            ? 1
                            : output_fire_count[i + k] + 1;
                }
            } else {
                if (!neuron_leak[(i + k) / 8] >> ((i + k) % 8) & 1) {
                    neuron_charge_buffer[(internal_timestep + 1) %
                                         tracked_timesteps_count][i + k] +=
                        neuron_charge_buffer[internal_timestep][i + k];
                }
            }
        }
    }

    fill(neuron_charge_buffer[internal_timestep].begin(),
         neuron_charge_buffer[internal_timestep].end(), 0);
#endif // AVX
#ifdef __ARM_NEON
    int8x16_t zero_vector = vdupq_n_s8(0);

    for (size_t i = 0; i < neuron_charge_buffer[internal_timestep].size();
         i += 16) {
        int8x16_t charges =
            vld1q_s8(&neuron_charge_buffer[internal_timestep][i]);
        int8x16_t min_potential_vec = vdupq_n_s8(min_potential);

        charges = vmaxq_s8(charges, min_potential_vec);

        int8x16_t thresholds = vld1q_s8(&neuron_threshold[i]);

        uint8_t fired_arr[16];
        uint8x16_t fired = vcgeq_s8(charges, thresholds);
        vst1q_u8(fired_arr, fired);

        for (size_t j = 0; j < 16; j++) {
            if (!fired_arr[j]) {
                continue;
            }
        }
    }
    fill(neuron_charge_buffer[internal_timestep].begin(),
         neuron_charge_buffer[internal_timestep].end(), 0);
#endif
#ifdef RISCVV
    for (size_t i = 0; i < neuron_count; i += 16) {
        size_t vector_length = min((size_t)16, neuron_count - i);
        vint8m1_t zero_vector = __riscv_vmv_v_x_i8m1(0, vector_length);

        vint8m1_t charges = __riscv_vle8_v_i8m1(
            &neuron_charge_buffer[(internal_timestep * neuron_count) + i],
            vector_length);
        vint8m1_t min_potential_vec =
            __riscv_vmv_v_x_i8m1(min_potential, vector_length);
        charges =
            __riscv_vmax_vv_i8m1(charges, min_potential_vec, vector_length);
        vint8m1_t thresholds =
            __riscv_vle8_v_i8m1(&neuron_threshold[i], vector_length);

        // for (int z = 0; z < vector_length; z++) {
        //     printf("Neuron %d %sfire because %d/%d\n", i,
        //            neuron_charge_buffer[(internal_timestep * neuron_count) +
        //            i +
        //                                 z] >= neuron_threshold[i + z]
        //                ? "does "
        //                : "does not ",
        //            neuron_charge_buffer[(internal_timestep * neuron_count) +
        //            i +
        //                                 z],
        //            neuron_threshold[i + z]);
        // }

        vbool8_t fired =
            __riscv_vmsge_vv_i8m1_b8(charges, thresholds, vector_length);

        if (leak_mode != 'n') {
            vbool8_t not_fired = __riscv_vmnot_m_b8(fired, vector_length);
            vbool8_t leak = __riscv_vlm_v_b8(&neuron_leak[i], vector_length);
        }

        uint8_t fired_arr[16];
        __riscv_vsm_v_b8(fired_arr, fired, vector_length);
        // printf("0b%d%d%d%d_%d%d%d%d\n", fired_arr[0] & 128, fired_arr[0] &
        // 64,
        //        fired_arr[0] & 32, fired_arr[0] & 16, fired_arr[0] & 8,
        //        fired_arr[0] & 4, fired_arr[0] & 2, fired_arr[0] & 1);

        // printf("Neurons fired: ");
        for (size_t j = 0; j < vector_length; j++) {
            if (!(fired_arr[j / 8] & (1 << (j % 8)))) {
                continue;
            }
            // printf("%zu ", i + j);
            neuron_fired[i + j] = true;
            if (outputs[i + j]) {
                output_last_fire_timestep[i + j] = time;
                output_fire_count[i + j] = output_fire_count[i + j] == -1
                                               ? 1
                                               : output_fire_count[i + j] + 1;
            }

            size_t num_outgoing = synapse_to[i + j].size();
            for (size_t k = 0; k < num_outgoing; k += 16) {
                size_t vector_length = min((size_t)16, num_outgoing - k);

                vint8m1_t weights = __riscv_vle8_v_i8m1(
                    &synapse_weight[i + j][k], vector_length);
                vuint8m1_t delays = __riscv_vle8_v_u8m1(
                    &synapse_delay[i + j][k], vector_length);
                vuint16m2_t destinations =
                    __riscv_vle16_v_u16m2(&synapse_to[i + j][k], vector_length);

                vuint16m2_t indexes = __riscv_vwaddu_vx_u16m2(
                    delays, (int8_t)internal_timestep, vector_length);
                indexes = __riscv_vremu_vx_u16m2(
                    indexes, tracked_timesteps_count, vector_length);

                // vmadd.vx vd, rs1, vs2, vm | vd[i] = (x[rs1] * vd[i]) + vs2[i]
                indexes =
                    __riscv_vmadd_vx_u16m2(indexes, (uint16_t)neuron_count,
                                           destinations, vector_length);

                vint8m1_t downstream_charges = __riscv_vloxei16_v_i8m1(
                    neuron_charge_buffer, indexes, vector_length);

                downstream_charges = __riscv_vadd_vv_i8m1(
                    downstream_charges, weights, vector_length);

                __riscv_vsoxei16_v_i8m1(neuron_charge_buffer, indexes,
                                        downstream_charges, vector_length);
            }
        }
        // putchar('\n');
        for (size_t z = 0; z < 16 && i + z < neuron_count; z++) {
            neuron_charge_buffer[internal_timestep * neuron_count + i + z] = 0;
        }
    }
    // memset(&neuron_charge_buffer[(internal_timestep * neuron_count)], 0,
    //        sizeof(*neuron_charge_buffer) * tracked_timesteps_count *
    //            neuron_count);
#endif
}

double Network::get_time() { return (double)current_timestep; }

double Network::output_last_fire(int output_id) {
    return output_last_fire_timestep[output_mappings[output_id]];
}
vector<double> Network::output_last_fires() {
    vector<double> return_vector;

    for (size_t i = 0; i < output_mappings.size(); i++) {
        return_vector.push_back(output_last_fire(i));
    }

    return return_vector;
}

int Network::output_count(int output_id) {
    return output_fire_count[output_mappings[output_id]];
}

vector<int> Network::output_counts() {
    vector<int> return_vector;

    for (size_t i = 0; i < output_mappings.size(); i++) {
        return_vector.push_back(output_count(i));
    }

    return return_vector;
}

vector<double> Network::output_vector(int output_id) {
    (void)output_id;
    return vector<double>{};
}
vector<vector<double>> Network::output_vectors() {
    return vector<vector<double>>{};
}

long long Network::total_neuron_counts() { return -1; }
long long Network::total_neuron_accumulates() { return -1; }
vector<int> Network::neuron_counts() { return vector<int>{}; }
vector<double> Network::neuron_last_fires() { return vector<double>{}; }
vector<vector<double>> Network::neuron_vectors() {
    return vector<vector<double>>{};
}

vector<double> Network::neuron_charges() {
    vector<double> return_vector;

    for (size_t i = 0; i < neuron_mappings.size(); i++) {
        return_vector.push_back(
            neuron_charge_buffer[(current_timestep % tracked_timesteps_count) *
                                     neuron_count +
                                 neuron_mappings[i]]);
    }

    return return_vector;
}
/** synapse_weights() returns three vectors, pres, posts and vals. Each entry
 * represents a synapse weight -- pres[i] is the id of the pre-neuron, posts[i]
 * is the id of the post-neuron, and vas[i] is the weight of the synapse.*/
void Network::synapse_weights(vector<uint32_t>& pres, vector<uint32_t>& posts,
                              vector<double>& vals) {
    pres.clear();
    posts.clear();
    vals.clear();

    for (size_t i = 0; i < neuron_count; i++) {
        for (size_t j = 0;
             j < synapse_to[i].size() && j < synapse_weight[i].size(); j++) {
            pres.push_back(i);
            posts.push_back(synapse_to[i][j]);
            vals.push_back(synapse_weight[i][j]);
        }
    }
}

void Network::clear_activity() {
    memset(neuron_charge_buffer, 0,
           sizeof(*neuron_charge_buffer) * tracked_timesteps_count *
               neuron_count);

    fill(neuron_fired.begin(), neuron_fired.end(), false);
    fill(output_last_fire_timestep.begin(), output_last_fire_timestep.end(),
         -1);
    fill(output_fire_count.begin(), output_fire_count.end(), 0);

    current_timestep = 0;
}

void Network::clear_output_tracking() {
    fill(output_last_fire_timestep.begin(), output_last_fire_timestep.end(),
         -1);
    fill(output_fire_count.begin(), output_fire_count.end(), 0);
}

Processor::Processor(json& params) {
    Parameter_Check_Json_T(params, jrisp_spec);

    /* Default params */

    min_delay = 1;
    leak_mode = "none";

    /* You don't have to check for these, because they are required in the
     * JSON
     */
    max_delay = params["max_delay"];
    min_threshold = params["min_threshold"];
    max_threshold = params["max_threshold"];
    min_potential = params["min_potential"];
    tracked_timesteps_count = params["tracked_timesteps"];

    if (!params.contains("min_weight"))
        throw SRE("JRISP: Need parameter min_weight.");
    if (!params.contains("max_weight"))
        throw SRE("JRISP: Need parameter max_weight.");
    if (params.contains("inputs_from_weights")) {
        throw SRE("JRISP: If you don't specify weights, you cannot specify "
                  "inputs_from_weights.");
    }

    min_weight = params["min_weight"];
    max_weight = params["max_weight"];

    if (params.contains("spike_value_factor")) {
        spike_value_factor = params["spike_value_factor"];
    } else {
        spike_value_factor = max_weight;
        if (max_weight < max_threshold) {
            fprintf(stderr, "Warning: max_weight < max_threshold and "
                            "spike_value_factor unset.\n");
            fprintf(stderr, "Spike_value_factor set to %lg.\n",
                    spike_value_factor);
        }
    }

    if (params.contains("leak_mode")) {
        leak_mode = params["leak_mode"];
    }

    if (leak_mode != "all" && leak_mode != "none" &&
        leak_mode != "configurable") {
        throw SRE("Reading processor json - bad leak_mode. Must be all, none "
                  "or configurable");
    }

    /* General Error Checking */
    if (!is_integer(max_delay)) {
        throw SRE("max_delay must be an integer.");
    }
    if (!is_integer(min_weight)) {
        throw SRE("min_weight must be an integer.");
    }
    if (!is_integer(max_weight)) {
        throw SRE("max_weight must be an integer.");
    }
    if (!is_integer(min_potential)) {
        throw SRE("min_potential must be an integer.");
    }
    if (!is_integer(min_threshold)) {
        throw SRE("min_threshold must be an integer.");
    }
    if (!is_integer(max_threshold)) {
        throw SRE("max_threshold must be an integer.");
    }
    if (max_delay >= tracked_timesteps_count) {
        throw SRE("max_delay (" + to_string(max_delay) +
                  ") must be < tracked_timesteps(" +
                  to_string(tracked_timesteps_count) + ").");
    }

    if (min_potential > 0) {
        throw SRE("Reading processor json - min_potential must be <= 0.");
    }

    /* Have the saved parameters include all of the default information. The
     * reason is that this way, if defaults change, you can still have this
     * information stored. */
    saved_params["min_weight"] = min_weight;
    saved_params["max_weight"] = max_weight;
    saved_params["spike_value_factor"] = spike_value_factor;

    saved_params["max_delay"] = max_delay;
    saved_params["min_threshold"] = min_threshold;
    saved_params["max_threshold"] = max_threshold;
    saved_params["min_potential"] = min_potential;
    saved_params["tracked_timesteps"] = tracked_timesteps_count;

    saved_params["leak_mode"] = leak_mode;
}

Processor::~Processor() {
    map<int, jrisp::Network*>::const_iterator it;
    for (it = networks.begin(); it != networks.end(); ++it)
        delete it->second;
}

bool Processor::load_network(neuro::Network* net, int network_id) {
    jrisp::Network* jrisp_net;
    string error = "";
    string rln = "jrisp::load_network() - ";

    /* Error Check properties */
    if (!net->is_node_property("Threshold")) {
        error = rln + "Missing node' Threshold property\n";
    }
    if (!net->is_edge_property("Weight")) {
        error += (rln + "Missing edge Weight property\n");
    }
    if (!net->is_edge_property("Delay")) {
        error += (rln + "Missing edge Delay propery\n");
    }
    if (leak_mode[0] == 'c' && !net->is_node_property("Leak")) {
        error += (rln + "Missing node' Leak propery\n");
    }

    if (net->get_properties().as_json() != get_network_properties().as_json()) {
        error += (rln + "neuro::Network's properties are different than "
                        "processor's network properties\n");
    }

    if (error != "") {
        cerr << error;
        return false;
    }

    if (networks.find(network_id) != networks.end())
        delete networks[network_id];

    jrisp_net = new jrisp::Network(net, min_potential, leak_mode[0],
                                   tracked_timesteps_count, spike_value_factor);

    networks[network_id] = jrisp_net;

    return true;
}

bool Processor::load_networks(std::vector<neuro::Network*>& n) {
    for (size_t i = 0; i < n.size(); i++) {
        if (load_network(n[i], i) == false) {
            for (size_t j = 0; j <= i; j++) {
                delete networks[j];
                networks.erase(j);
            }

            return false;
        }
    }

    return true;
}

void Processor::clear(int network_id) {
    jrisp::Network* jrisp_net = get_jrisp_network(network_id);
    networks.erase(network_id);
    delete jrisp_net;
}

void Processor::apply_spike(const Spike& s, bool normalize, int network_id) {
    get_jrisp_network(network_id)->apply_spike(s, normalize);
}

void Processor::apply_spike(const Spike& s, const vector<int>& network_ids,
                            bool normalize) {
    for (size_t i = 0; i < network_ids.size(); i++) {
        apply_spike(s, normalize, network_ids[i]);
    }
}

void Processor::apply_spikes(const vector<Spike>& s, bool normalize,
                             int network_id) {
    for (size_t i = 0; i < s.size(); i++) {
        apply_spike(s[i], normalize, network_id);
    }
}

void Processor::apply_spikes(const vector<Spike>& s,
                             const vector<int>& network_ids, bool normalize) {
    for (size_t i = 0; i < network_ids.size(); i++) {
        apply_spikes(s, normalize, network_ids[i]);
    }
}

void Processor::run(double duration, int network_id) {
    if (duration < 0) {
        throw SRE("jrisp::Processor::run called with a negative duration (" +
                  to_string(duration) + ").");
    }

    get_jrisp_network(network_id)->run(static_cast<size_t>(duration));
}

void Processor::run(double duration, const vector<int>& network_ids) {
    for (size_t i = 0; i < network_ids.size(); i++) {
        run(duration, network_ids[i]);
    }
}

long long Processor::total_neuron_counts(int network_id) {
    return get_jrisp_network(network_id)->total_neuron_counts();
}

long long Processor::total_neuron_accumulates(int network_id) {
    return get_jrisp_network(network_id)->total_neuron_accumulates();
}

double Processor::get_time(int network_id) {
    return get_jrisp_network(network_id)->get_time();
}

bool Processor::track_output_events(int output_id, bool track, int network_id) {
    (void)output_id;
    (void)track;
    (void)network_id;
    return false;
}

bool Processor::track_neuron_events(uint32_t node_id, bool track,
                                    int network_id) {
    (void)node_id;
    (void)track;
    (void)network_id;
    return false;
}

double Processor::output_last_fire(int output_id, int network_id) {
    return get_jrisp_network(network_id)->output_last_fire(output_id);
}

vector<double> Processor::output_last_fires(int network_id) {
    return get_jrisp_network(network_id)->output_last_fires();
}

int Processor::output_count(int output_id, int network_id) {
    return get_jrisp_network(network_id)->output_count(output_id);
}

vector<int> Processor::output_counts(int network_id) {
    return get_jrisp_network(network_id)->output_counts();
}

vector<double> Processor::output_vector(int output_id, int network_id) {
    return get_jrisp_network(network_id)->output_vector(output_id);
}

vector<vector<double>> Processor::output_vectors(int network_id) {
    return get_jrisp_network(network_id)->output_vectors();
}

vector<int> Processor::neuron_counts(int network_id) {
    return get_jrisp_network(network_id)->neuron_counts();
}

vector<vector<double>> Processor::neuron_vectors(int network_id) {
    return get_jrisp_network(network_id)->neuron_vectors();
}

vector<double> Processor::neuron_charges(int network_id) {
    return get_jrisp_network(network_id)->neuron_charges();
}

vector<double> Processor::neuron_last_fires(int network_id) {
    return get_jrisp_network(network_id)->neuron_last_fires();
}

void Processor::synapse_weights(vector<uint32_t>& pre, vector<uint32_t>& posts,
                                vector<double>& vals, int network_id) {
    return get_jrisp_network(network_id)->synapse_weights(pre, posts, vals);
}

void Processor::clear_activity(int network_id) {
    get_jrisp_network(network_id)->clear_activity();
}

PropertyPack Processor::get_network_properties() const {
    PropertyPack pp;

    pp.add_node_property("Threshold", min_threshold, max_threshold,
                         Property::Type::INTEGER);

    if (leak_mode[0] == 'c') {
        pp.add_node_property("Leak", 0, 1, Property::Type::BOOLEAN);
    }

    pp.add_edge_property("Weight", min_weight, max_weight,
                         Property::Type::INTEGER);

    pp.add_edge_property("Delay", min_delay, max_delay,
                         Property::Type::INTEGER);

    return pp;
}

json Processor::get_processor_properties() const {
    json j = json::object();

    j["binary_input"] = true;
    j["spike_raster_info"] = false;
    j["plasticity"] = "none";
    j["integration_delay"] = false;

    return j;
}

json Processor::get_params() const { return saved_params; }

string Processor::get_name() const { return "jrisp"; }

Network* Processor::get_jrisp_network(int network_id) {
    map<int, jrisp::Network*>::const_iterator it;
    char buf[200];
    it = networks.find(network_id);
    if (it == networks.end()) {
        snprintf(buf, 200,
                 "jrisp::Processor::get_jrisp_network() network_id %d does not "
                 "exist",
                 network_id);
        throw SRE((string)buf);
    }

    return it->second;
}

} // namespace jrisp