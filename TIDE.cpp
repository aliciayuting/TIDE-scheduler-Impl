#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdlib>
#include <iostream>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

using node_id_t = uint32_t;

struct model {
    int id;
    int size;
};

struct worker {
  node_id_t id;
  std::vector<uint32_t> cached_model_ids;
  uint64_t wait_time;
};

struct task {
    std::vector<std::string> dependencies;
    std::vector<model> models;
    uint32_t output_size;
    uint64_t exec_time;
};

using task_profile_t = std::unordered_map<std::string, task>;

uint64_t time_since_epoch();
uint64_t host_to_GPU_delay(uint64_t object_size);
uint64_t round_double(double d);
uint64_t CPU_to_CPU_delay(uint64_t object_size);
uint64_t GPU_to_host_delay(uint64_t object_size);
uint64_t GPU_to_GPU_delay(uint64_t object_size);
std::string tide_scheduler(std::string entry_prefix, task_profile_t task_profile, std::vector<std::string> sorted_tasks, const std::vector<worker>& workers);


int main(int argc, char** argv) {
    int num_workers = 5;
    if(argc > 1) {
        num_workers = std::atoi(argv[1]);
    }
    int num_tasks = 5;
    if(argc > 2) {
        num_tasks = std::atoi(argv[2]);
    }
    bool is_horizontal = true;
    if(argc > 2) {
        is_horizontal = std::atoi(argv[2]);
    }
    // rank of a node is its position in the workers
    std::vector<worker> workers(num_workers);
    for(uint32_t i = 0; i < num_workers; ++i) {
        workers[i] = {
                .id = i,
                .cached_model_ids = {0, 1},
                .wait_time = 10};
    }

    std::vector<std::string> sorted_tasks(num_tasks);
    std::generate(sorted_tasks.begin(), sorted_tasks.end(),
                  [i = 0]() mutable {
                      return "task" + std::to_string(i++);
                  });

    std::vector<model> required_models({{.id = 0, .size = 100}});
    uint32_t output_size = 10;
    uint64_t exec_time = 10;

    task_profile_t task_profile;
    task_profile[sorted_tasks[0]] = {
            .dependencies = {},
            .models = required_models,
            .output_size = output_size,
            .exec_time = exec_time};

    if(is_horizontal) {
        for(int i = 1; i < num_tasks - 1; ++i) {
            task_profile[sorted_tasks[i]] = {
                    .dependencies = {sorted_tasks[0]},
                    .models = required_models,
                    .output_size = output_size,
                    .exec_time = exec_time};
        }

        task_profile[sorted_tasks.back()] = {
                .dependencies = {sorted_tasks.begin() + 1, sorted_tasks.end() - 1},
                .models = required_models,
                .output_size = output_size,
                .exec_time = exec_time};
    } else {
        for(int i = 1; i < num_tasks; ++i) {
            task_profile[sorted_tasks[i]] = {
                    .dependencies = {sorted_tasks[i - 1]},
                    .models = required_models,
                    .output_size = output_size,
                    .exec_time = exec_time};
        }
    }

    uint64_t sum_times = 0;
    for(int i = 0; i < 100; ++i) {
      uint64_t start_time = time_since_epoch();
        tide_scheduler("task0", task_profile, sorted_tasks, workers);
        uint64_t end_time = time_since_epoch();
        sum_times += (end_time - start_time);
    }

    std::cout << num_workers << "workers, "
              << "average runtime " << (sum_times / 100.0) << " us" << std::endl;

    return 0;
}

uint64_t time_since_epoch() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
                   std::chrono::high_resolution_clock::now().time_since_epoch())
            .count();
}

inline uint64_t round_double(double d) {
    if(d < UINT64_MAX)
    {
        return (uint64_t)(d);
    }
    return UINT64_MAX;
}

#define FIXED_COST_GROUP_FORMATION (2)
inline uint64_t host_to_GPU_delay(uint64_t object_size) {
    double throughput = 4.7 * (1 << 20) / 1000.0;
    double delay = (1000.0 * object_size / throughput) + FIXED_COST_GROUP_FORMATION;
    return round_double(delay);
}

inline uint64_t CPU_to_CPU_delay(uint64_t object_size) {
    double throughput;
    double delay;
    if(object_size < 1) {
        return 2;                        // 2 us
    } else if(object_size < (1 << 2)) {  // for size between 1 ~ 4 KB
        throughput = (3 + object_size * 2) * (1 << 20) / 1000.0;
        delay = (object_size / throughput) + FIXED_COST_GROUP_FORMATION;
    } else {
        throughput = 12 * (1 << 20) / 1000.0;
        delay = (object_size / throughput) + FIXED_COST_GROUP_FORMATION;
    }
    return round_double(delay);
}

inline uint64_t GPU_to_host_delay(uint64_t object_size) {
    double throughput = 4.7 * (1 << 20) / 1000.0;
    double delay = (object_size / throughput) + FIXED_COST_GROUP_FORMATION;
    return round_double(delay);
}

inline uint64_t GPU_to_GPU_delay(uint64_t object_size) {
    uint64_t localgpu_to_localcpu_delay = GPU_to_host_delay(object_size);
    uint64_t cpu_to_cpu_delay = CPU_to_CPU_delay(object_size);
    uint64_t remotecpu_to_remotegpu_delay = host_to_GPU_delay(object_size);
    return localgpu_to_localcpu_delay + cpu_to_cpu_delay + remotecpu_to_remotegpu_delay;
}

std::string tide_scheduler(std::string entry_prefix, task_profile_t task_profile, std::vector<std::string> sorted_tasks, const std::vector<worker>& workers) {
    // vertex pathname -> (node_id, finish_time(us))
    // in the algorithm denote task ~ vertex pathname
    std::unordered_map<std::string, std::pair<node_id_t, uint64_t>> allocated_tasks_info;
    uint64_t cur_us = time_since_epoch();
    std::unordered_map<node_id_t, uint64_t> earliest_available_time;
    for(const worker& worker : workers) {
      earliest_available_time[worker.id] = cur_us + worker.wait_time;
    }
    for(auto& task_name : sorted_tasks) {
        auto& task = task_profile.at(task_name);
        // 0. PRE-COMPUTE (used later by 2. case1) get the earliest start time, suppose all preq_tasks need to transfer data
        uint64_t earliest_start_time = cur_us;
        if(task_name == sorted_tasks[0]) {  // first task
            earliest_start_time += host_to_GPU_delay(task.output_size);
        }  
	node_id_t selected_worker_id = -1;
	uint64_t min_start_time = UINT64_MAX;
        for(const worker& worker : workers) {
            // should i cache this instead of recomputing it for every worker?
            for(auto& preq_task_name : task.dependencies) {
                auto& preq_task = task_profile.at(preq_task_name);
                auto& info = allocated_tasks_info.at(preq_task_name);
                uint64_t arrival_time = info.second;
                if(worker.id != info.first){ 
                    arrival_time += GPU_to_GPU_delay(preq_task.output_size);  
                }
                earliest_start_time = std::max(earliest_start_time, arrival_time);
            }
            uint64_t model_fetch_time = 0;
            auto& required_models = task.models;
            auto& model_ids = worker.cached_model_ids;
            for(model& required_model : required_models) {
                if(std::find(model_ids.begin(), model_ids.end(), required_model.id) == model_ids.end()) {
                    model_fetch_time += host_to_GPU_delay(required_model.size);
                }
            }
            uint64_t start_time = std::max(earliest_available_time[worker.id], earliest_start_time) + model_fetch_time;
	    if(start_time < min_start_time) {
	      min_start_time = start_time;
	      selected_worker_id = worker.id;
	    }
        }
        allocated_tasks_info[task_name] = {selected_worker_id, min_start_time + task.exec_time};
	earliest_available_time[selected_worker_id] = min_start_time + task.exec_time;
    }
    std::string allocated_machines;
    for(auto& task_name : sorted_tasks) {
        allocated_machines += std::to_string(allocated_tasks_info.at(task_name).first) + ",";
    }
    return allocated_machines;
}
