#include "crisp.hpp"
#include "utils/json_helpers.hpp"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <dlfcn.h>
#include <stdexcept>
#include <string>
#include <unistd.h>

typedef std::runtime_error SRE;
using namespace std;

namespace crisp {
/** Configurable settings for crisp */
static json crisp_spec = {
    {"min_weight", "I"},
    {"fire_like_ravens", "B"},
    {"run_time_inclusive", "B"},
    {"threshold_inclusive", "B"},
    {"max_weight", "I"},
    {"max_delay", "I"},
    {"min_threshold", "I"},
    {"max_threshold", "I"},
    {"min_potential", "I"},
    {"tracked_timesteps", "I"},
    {"leak_mode", "S"},
    {"spike_value_factor", "D"},
    {"discrete", "B"},
    {"Necessary",
     {"min_weight", "max_weight", "max_delay", "min_threshold", "max_threshold",
      "min_potential"}},
};

static inline bool is_integer(double v) {
    int iv;

    iv = v;
    return (iv == v);
}

Network::Network(neuro::Network* net, double _min_potential, char leak,
                 size_t _tracked_timesteps, double _spike_value_factor,
                 bool discrete) {
    // assert(false);
    spike_value_factor = _spike_value_factor;

    net->make_sorted_node_vector();
    unordered_map<int, int> neuron_id_to_ind;
    for (size_t i = 0; i < net->sorted_node_vector.size(); i++) {
        neuron_id_to_ind[net->sorted_node_vector[i]->id] = i;
    }

    for (int i = 0; i < net->num_inputs(); i++) {
        input_mappings.push_back(neuron_id_to_ind[net->get_input(i)->id]);
    }

    for (int i = 0; i < net->num_outputs(); i++) {
        output_mappings.push_back(neuron_id_to_ind[net->get_output(i)->id]);
    }

    neuron_count = net->num_nodes();

    size_t max_delay = 0;
    net->make_sorted_node_vector();
    for (Node* node : net->sorted_node_vector) {
        for (Edge* edge : node->outgoing) {
            if (edge->get("Delay") > max_delay) {
                max_delay = edge->get("Delay");
            }
        }
    }

    string data_type = discrete ? "int16_t" : "double";

    if (max_delay < _tracked_timesteps) {
        max_delay = _tracked_timesteps - 1;
    }

    size_t table_rows = 1 << (int)ceil(log2(max_delay + 1));

    FILE* code = nullptr;
    int rn = 0;

    while (code == nullptr) {
        rn = rand();
        string fname = "/tmp/" + to_string(rn) + "_out.c";
        c_file = fname;

        code = fopen(fname.c_str(), "wx");
    }

    fprintf(code, "#include <stdint.h>\n");
    fprintf(code, "#include <stddef.h>\n");
    fprintf(code, "#include <string.h>\n");
    fprintf(code, "#include <stdbool.h>\n");
    fprintf(code, "#include <stdbool.h>\n");
    fprintf(code, "#include <sys/types.h>\n");
    fprintf(code, "\n");
    fprintf(code, "typedef struct export_vtable {\n");
    fprintf(code,
            "\tvoid (*compiled_apply_spike)(size_t input_id, size_t time, "
            "double value);\n");
    fprintf(code, "\tsize_t (*compiled_output_count)(size_t output_id);\n");
    fprintf(code,
            "\tssize_t (*compiled_output_last_fire)(size_t output_id);\n");
    fprintf(code, "\tdouble (*compiled_neuron_charge)(size_t neuron_id);\n");
    fprintf(code, "\tvoid (*compiled_run)(size_t timesteps);\n");
    fprintf(code, "\tvoid (*compiled_clear_output_tracking)();\n");
    fprintf(code, "\tvoid (*compiled_clear_activity)();\n");
    fprintf(code, "} export_vtable;\n");
    fprintf(code, "\n");
    fprintf(code, "const size_t num_inputs = %d;\n", net->num_inputs());
    fprintf(code, "const size_t num_outputs = %d;\n", net->num_outputs());
    fprintf(code, "const size_t tracked_timesteps = %zu;\n", table_rows);
    fprintf(code, "\n");
    fprintf(code, "%s charge_table[%zu][%zu] = {0};\n", data_type.c_str(),
            table_rows, net->num_nodes());
    fprintf(code, "size_t fire_count[%d] = {%c};\n", net->num_outputs(),
            (net->num_outputs() != 0) ? '0' : ' ');
    fprintf(code, "int16_t last_fire[%d] = {%c};\n", net->num_outputs(),
            (net->num_outputs() != 0) ? '0' : ' ');
    fprintf(code, "size_t timestep = 0;\n");
    fprintf(code, "\n");
    fprintf(
        code,
        "void apply_spike(size_t neuron_id, size_t time, double value) {\n");
    fprintf(
        code,
        "\tcharge_table[(timestep + time) %% %zu][neuron_id] += (%s)value;\n",
        table_rows, data_type.c_str());
    fprintf(code, "}\n");
    fprintf(code, "\n");
    fprintf(code, "size_t output_count(size_t output_id) {\n");
    fprintf(code, "\treturn fire_count[output_id];\n");
    fprintf(code, "}\n");
    fprintf(code, "\n");
    fprintf(code, "ssize_t output_last_fire(size_t output_id) {\n");
    fprintf(code, "\treturn last_fire[output_id];\n");
    fprintf(code, "}\n");
    fprintf(code, "\n");
    fprintf(code, "double neuron_charge(size_t neuron_id) {\n");
    fprintf(code, "\treturn charge_table[timestep %% %zu][neuron_id];\n",
            table_rows);
    fprintf(code, "}\n");
    fprintf(code, "\n");
    fprintf(code, "void clear_output_tracking() {\n");
    fprintf(code, "\tmemset(fire_count, 0, sizeof(fire_count));\n");
    fprintf(code, "\tmemset(last_fire, -1, sizeof(last_fire));\n");
    fprintf(code, "}\n");
    fprintf(code, "\n");
    fprintf(code, "void clear_activity() {\n");
    fprintf(code, "\tmemset(charge_table, 0, sizeof(charge_table));\n");
    fprintf(code, "}\n");
    fprintf(code, "\n");
    fprintf(code, "void run(size_t timesteps) {\n");
    fprintf(code, "\tfor (size_t step = 0; step < timesteps; step++) {\n");
    fprintf(code, "\t\tsize_t internal_ts = (timestep + step) %% %zu;\n",
            table_rows);
    fprintf(code, "\t\tbool fired;\n");
    fprintf(code, "\n");
    fprintf(code, "\t\tfor (size_t i = 0; i < %zu; i++) {\n", net->num_nodes());
    fprintf(code,
            "\t\t\tcharge_table[internal_ts][i] = "
            "\n\t\t\t\t(charge_table[internal_ts][i] < %f) ?\n\t\t\t\t\t "
            "%f :\n\t\t\t\t\tcharge_table[internal_ts][i];\n",
            _min_potential, _min_potential);
    fprintf(code, "\t\t}\n");
    fprintf(code, "\n");
    for (Node* node : net->sorted_node_vector) {
        double neuron_threshold = node->get("Threshold");
        fprintf(code, "\t\t// Neuron %u\n", node->id);

        if (node->is_output() || node->outgoing.size() > 0) {
            fprintf(code, "\t\tfired = charge_table[internal_ts][%d] >= ",
                    neuron_id_to_ind[node->id]);
            if (discrete) {
                fprintf(code, "%hd", (int16_t)neuron_threshold);
            } else {
                fprintf(code, "%f", neuron_threshold);
            }
            fprintf(code, ";\n");

            if (node->is_output()) {
                fprintf(code, "\t\tfire_count[%d] += fired;\n",
                        node->output_id);
                fprintf(code,
                        "\t\tlast_fire[%d] = fired ? (step) : last_fire[%d];\n",
                        node->output_id, node->output_id);
            }
            if (node->is_output() && node->outgoing.size() > 0) {
                fprintf(code, "\n");
            }
            if (node->outgoing.size() > 0) {
                for (Edge* edge : node->outgoing) {
                    fprintf(code,
                            "\t\tcharge_table[(internal_ts + %d) %% %zu][%d] "
                            "+= fired ? ",
                            (int)edge->get("Delay"), table_rows,
                            neuron_id_to_ind[edge->to->id]);
                    if (discrete) {
                        fprintf(code, "%hd", (int16_t)edge->get("Weight"));
                    } else {
                        fprintf(code, "%f", edge->get("Weight"));
                    }

                    fprintf(code, " : 0;\n");
                }
            }
        }
        if (leak != 'a') {
            fprintf(code,
                    "\t\tcharge_table[(internal_ts + 1) %% %zu][%d] += fired ? "
                    "0 : charge_table[internal_ts][%d];\n",
                    table_rows, neuron_id_to_ind[node->id],
                    neuron_id_to_ind[node->id]);
        }
    }
    fprintf(code, "\n");
    fprintf(
        code,
        "\t\tmemset(&charge_table[internal_ts], 0, sizeof(int16_t) * %zu);\n",
        net->num_nodes());
    fprintf(code, "\t}\n");
    fprintf(code, "\n");
    fprintf(code, "\ttimestep += timesteps;\n");
    fprintf(code, "\t for (size_t i = 0; i < %zu; i++) {\n", net->num_nodes());
    fprintf(
        code,
        "\t\tcharge_table[timestep %% %zu][i] = \n\t\t\t(charge_table[timestep "
        "%% %zu][i] < %f) ?\n\t\t\t\t %f :\n\t\t\t\t"
        "charge_table[timestep %% %zu][i];\n",
        table_rows, table_rows, _min_potential, _min_potential, table_rows);
    fprintf(code, "\t}\n");
    fprintf(code, "\n");
    fprintf(code, "}\n");
    fprintf(code, "\n");
    fprintf(
        code,
        "export_vtable vtable = {apply_spike, output_count, output_last_fire, "
        "neuron_charge, run, clear_output_tracking, clear_activity};\n");

    fclose(code);

    so_file = "/tmp/libcrisp_" + to_string(rn) + ".so";

    string command = "gcc -march=native -O3 -g3 -shared -fPIC " + c_file +
                     " -o /tmp/libcrisp_" + to_string(rn) + ".so";
    system(command.c_str());

    // cerr << c_file << endl;
    dlhandle = dlopen(so_file.c_str(), RTLD_NOW);
    if (!dlhandle) {
        puts(dlerror());
        exit(1);
    }

    num_inputs = *(size_t*)dlsym(dlhandle, "num_inputs");
    num_outputs = *(size_t*)dlsym(dlhandle, "num_outputs");
    tracked_timesteps = *(size_t*)dlsym(dlhandle, "tracked_timesteps");
    timestep = (size_t*)dlsym(dlhandle, "timestep");
    vtable = *(compiled_vtable*)dlsym(dlhandle, "vtable");
    vtable.compiled_clear_output_tracking();
}

Network::~Network() {
    if (dlhandle) {
        dlclose(dlhandle);
    }

    unlink(c_file.c_str());
    unlink(so_file.c_str());
}

void Network::apply_spike(const Spike& s, bool normalized) {
    if (normalized && (s.value < -1 || s.value > 1)) {
        throw SRE("crisp::Network::apply_spike() - value (" +
                  to_string(s.value) + ") must be in [-1,1].");
    }

    if ((size_t)s.time >= tracked_timesteps) {
        throw SRE(
            "crisp::Network::apply_spike() - Cannot spike past max_delay+1 " +
            to_string(tracked_timesteps));
    }

    int32_t spike_value = (normalized) ? s.value * spike_value_factor : s.value;

    vtable.compiled_apply_spike(input_mappings[s.id], s.time, spike_value);
}

void Network::run(size_t duration) {
    if (*timestep != 0) {
        vtable.compiled_clear_output_tracking();
    }

    vtable.compiled_run(duration);
}

double Network::get_time() { return *timestep; }

double Network::output_last_fire(int output_id) {
    return vtable.compiled_output_last_fire(output_id);
}
vector<double> Network::output_last_fires() {
    vector<double> return_vector;

    for (size_t i = 0; i < output_mappings.size(); i++) {
        return_vector.push_back(output_last_fire(i));
    }

    return return_vector;
}

int Network::output_count(int output_id) {
    return vtable.compiled_output_count(output_id);
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

    for (size_t i = 0; i < neuron_count; i++) {
        return_vector.push_back(vtable.compiled_neuron_charge(i));
    }

    return return_vector;
}

/** synapse_weights() returns three vectors, pres, posts and vals. Each entry
 * represents a synapse weight -- pres[i] is the id of the pre-neuron,
 * posts[i] is the id of the post-neuron, and vas[i] is the weight of the
 * synapse.*/
void Network::synapse_weights(vector<uint32_t>& pres, vector<uint32_t>& posts,
                              vector<double>& vals) {
    // pres.clear();
    // posts.clear();
    // vals.clear();

    // for (size_t i = 0; i < neuron_count; i++) {
    //   for (size_t j = 0; j < synapse_to[i].size() && j <
    //   synapse_weight[i].size();
    //        j++) {
    //     pres.push_back(i);
    //     posts.push_back(synapse_to[i][j]);
    //     vals.push_back(synapse_weight[i][j]);
    //   }
    // }
}

void Network::clear_activity() { vtable.compiled_clear_activity(); }

void Network::clear_output_tracking() {
    vtable.compiled_clear_output_tracking();
}

Processor::Processor(json& params) {
    Parameter_Check_Json_T(params, crisp_spec);

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

    if (!params.contains("min_weight"))
        throw SRE("CRISP: Need parameter min_weight.");
    if (!params.contains("max_weight"))
        throw SRE("CRISP: Need parameter max_weight.");
    if (params.contains("inputs_from_weights")) {
        throw SRE("CRISP: inputs_from_weights not supported");
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

    if (params.contains("tracked_timesteps")) {
        size_t tt = params["tracked_timesteps"];
        if (tt <= max_delay) {
            throw SRE("CRISP: tracked_timesteps cannot be <= max_delay");
        }

        tracked_timesteps_count = tt;
    } else {
        tracked_timesteps_count = max_delay + 1;
    }

    if (params.contains("leak_mode")) {
        leak_mode = params["leak_mode"];
    }

    if (leak_mode != "all" && leak_mode != "none" &&
        leak_mode != "configurable") {
        throw SRE("Reading processor json - bad leak_mode. Must be all, none "
                  "or configurable");
    }

    // if (params.contains("discrete") && params["discrete"] == false) {
    //   throw SRE("CRISP: discrete == false is not supported for CRISP");
    // }
    discrete = true;
    if (params.contains("discrete")) {
        discrete = params["discrete"];
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
    map<int, crisp::Network*>::const_iterator it;
    for (it = networks.begin(); it != networks.end(); ++it)
        delete it->second;
}

bool Processor::load_network(neuro::Network* net, int network_id) {
    crisp::Network* crisp_net;
    string error = "";
    string rln = "crisp::load_network() - ";

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

    crisp_net = new crisp::Network(net, min_potential, leak_mode[0],
                                   tracked_timesteps_count, spike_value_factor,
                                   discrete);

    networks[network_id] = crisp_net;

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
    crisp::Network* crisp_net = get_crisp_network(network_id);
    networks.erase(network_id);
    delete crisp_net;
}

void Processor::apply_spike(const Spike& s, bool normalized, int network_id) {
    get_crisp_network(network_id)->apply_spike(s, normalized);
}

void Processor::apply_spike(const Spike& s, const vector<int>& network_ids,
                            bool normalized) {
    for (size_t i = 0; i < network_ids.size(); i++) {
        apply_spike(s, normalized, network_ids[i]);
    }
}

void Processor::apply_spikes(const vector<Spike>& s, bool normalized,
                             int network_id) {
    for (size_t i = 0; i < s.size(); i++) {
        apply_spike(s[i], normalized, network_id);
    }
}

void Processor::apply_spikes(const vector<Spike>& s,
                             const vector<int>& network_ids, bool normalized) {
    for (size_t i = 0; i < network_ids.size(); i++) {
        apply_spikes(s, network_ids[i], normalized);
    }
}

void Processor::run(double duration, int network_id) {
    if (duration < 0) {
        throw SRE("crisp::Processor::run called with a negative duration (" +
                  to_string(duration) + ").");
    }

    get_crisp_network(network_id)->run(static_cast<size_t>(duration));
}

void Processor::run(double duration, const vector<int>& network_ids) {
    for (size_t i = 0; i < network_ids.size(); i++) {
        run(duration, network_ids[i]);
    }
}

long long Processor::total_neuron_counts(int network_id) {
    return get_crisp_network(network_id)->total_neuron_counts();
}

long long Processor::total_neuron_accumulates(int network_id) {
    return get_crisp_network(network_id)->total_neuron_accumulates();
}

double Processor::get_time(int network_id) {
    return get_crisp_network(network_id)->get_time();
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
    return get_crisp_network(network_id)->output_last_fire(output_id);
}

vector<double> Processor::output_last_fires(int network_id) {
    return get_crisp_network(network_id)->output_last_fires();
}

int Processor::output_count(int output_id, int network_id) {
    return get_crisp_network(network_id)->output_count(output_id);
}

vector<int> Processor::output_counts(int network_id) {
    return get_crisp_network(network_id)->output_counts();
}

vector<double> Processor::output_vector(int output_id, int network_id) {
    return get_crisp_network(network_id)->output_vector(output_id);
}

vector<vector<double>> Processor::output_vectors(int network_id) {
    return get_crisp_network(network_id)->output_vectors();
}

vector<int> Processor::neuron_counts(int network_id) {
    return get_crisp_network(network_id)->neuron_counts();
}

vector<vector<double>> Processor::neuron_vectors(int network_id) {
    return get_crisp_network(network_id)->neuron_vectors();
}

vector<double> Processor::neuron_charges(int network_id) {
    return get_crisp_network(network_id)->neuron_charges();
}

vector<double> Processor::neuron_last_fires(int network_id) {
    return get_crisp_network(network_id)->neuron_last_fires();
}

void Processor::synapse_weights(vector<uint32_t>& pre, vector<uint32_t>& posts,
                                vector<double>& vals, int network_id) {
    return get_crisp_network(network_id)->synapse_weights(pre, posts, vals);
}

void Processor::clear_activity(int network_id) {
    get_crisp_network(network_id)->clear_activity();
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

string Processor::get_name() const { return "crisp"; }

Network* Processor::get_crisp_network(int network_id) {
    map<int, crisp::Network*>::const_iterator it;
    char buf[200];
    it = networks.find(network_id);
    if (it == networks.end()) {
        snprintf(buf, 200,
                 "crisp::Processor::get_crisp_network() network_id %d does not "
                 "exist",
                 network_id);
        throw SRE((string)buf);
    }

    return it->second;
}

} // namespace crisp
