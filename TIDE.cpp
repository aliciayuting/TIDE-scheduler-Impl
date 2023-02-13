#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>


using node_id_t = uint32_t;

using pre_adfg_t = 
                std::unordered_map<
                    std::string, //pathname  ("/task6/cdacadc", [value, (0,2,4,5)])
                    std::tuple<
                        std::set<std::string>,                         // required_objects_pathnames
                        std::vector<uint32_t>,                            // required_models
                        std::vector<uint32_t>,                             // required_models_size
                        std::vector<std::string>,                       // sorted_pathnames
                        uint32_t,                                       // expected output size in KB
                        uint64_t                                        // estimated excution time in us
                    >
                >;

inline uint64_t round_double(double _d){
     if (_d >= -9223372036854775808.0   // -2^63
          && _d <   9223372036854775808.0)  // 2^63
     {
          return static_cast<uint64_t>(_d);
     }
     return UINT64_MAX;
}

#define FIXED_COST_GROUP_FORMATION  (2)
inline uint64_t host_to_GPU_delay(uint64_t object_size){
     double throughput = 4.7 * (1 << 20) / 1000.0;
     double delay = (1000.0 * object_size / throughput) + FIXED_COST_GROUP_FORMATION;
     return round_double(delay);
}

inline uint64_t CPU_to_CPU_delay(uint64_t object_size){
     double throughput;
     double delay;
     if(object_size < 1){
          return 2; // 2 us
     }else if(object_size < (1 << 2)){ // for size between 1 ~ 4 KB
          throughput = (3 + object_size * 2) * (1 << 20) / 1000.0;
          delay = (object_size / throughput) + FIXED_COST_GROUP_FORMATION;
     }else{
          throughput = 12 * (1 << 20) / 1000.0;
          delay = (object_size / throughput) + FIXED_COST_GROUP_FORMATION;
     }
     return round_double(delay);
     
}

inline uint64_t GPU_to_host_delay(uint64_t object_size){
     double throughput = 4.7 * (1 << 20) / 1000.0;
     double delay = (object_size / throughput) + FIXED_COST_GROUP_FORMATION;
     return round_double(delay);
}

inline uint64_t GPU_to_GPU_delay(uint64_t object_size){
     uint64_t localgpu_to_localcpu_delay = GPU_to_host_delay(object_size);
     uint64_t cpu_to_cpu_delay = CPU_to_CPU_delay(object_size);
     uint64_t remotecpu_to_remotegpu_delay = host_to_GPU_delay(object_size);
     return localgpu_to_localcpu_delay + cpu_to_cpu_delay + remotecpu_to_remotegpu_delay;
}


/** TODO: write a standalone scheduler for performance testing purposes. */
std::string tide_scheduler(std::string entry_prefix, pre_adfg_t pre_adfg, std::vector<node_id_t> workers_set,
                              std::unordered_map<node_id_t, uint64_t> worker_waittime, 
                              std::unordered_map<node_id_t, std::vector<uint32_t>> worker_cached_models){
     // vertex pathname -> (node_id, finish_time(us))
     // in the algorithm denote task ~ vertex pathname
     std::unordered_map<std::string, std::tuple<node_id_t,uint64_t>> allocated_tasks_info;
     uint64_t cur_us = std::chrono::duration_cast<std::chrono::microseconds>(
                              std::chrono::high_resolution_clock::now().time_since_epoch())
                              .count();
     /** TODO: optimize this get_sorted_pathnames step by better design of where to store sorted_pathnames in pre_adfg_t */
     std::vector<std::string>& sorted_pathnames = std::get<3>(pre_adfg.at(entry_prefix));
     for(auto& pathname: sorted_pathnames){
          auto& dependencies = pre_adfg.at(pathname);
          // 0. PRE-COMPUTE (used later by 2. case1) get the earliest start time, suppose all preq_tasks need to transfer data
          uint64_t prev_EST = cur_us;
          // the worker_ids where pre-requisit tasks are executed
          std::set<node_id_t> preq_workers;
          std::set<std::string>& required_tasks = std::get<0>(dependencies);
          for(auto& preq_task : required_tasks){
/** TODO: std::tuple<node_id_t,uint64_t>& of allocated_tasks_info.at(preq_task)  */
               preq_workers.emplace(std::get<0>(allocated_tasks_info.at(preq_task)));
               uint64_t preq_finish_time = std::get<1>(allocated_tasks_info.at(preq_task));
               uint32_t preq_result_size = std::get<4>(pre_adfg.at(pathname));
               uint64_t preq_arrive_time = preq_finish_time + GPU_to_GPU_delay(preq_result_size);
               prev_EST = std::max(prev_EST, preq_arrive_time);
          }
          if(pathname == sorted_pathnames[0]){ // first task
               /** TODO: assumming input_size=output_size, for finer-grained HEFT, use input size instead of output size*/
               prev_EST += host_to_GPU_delay(std::get<4>(pre_adfg.at(pathname))); 
          }
          std::map<node_id_t, uint64_t> workers_start_times;
          for(node_id_t cur_worker: workers_set){
               uint64_t cur_worker_waittime = 0;
               uint64_t model_fetch_time = 0;
               cur_worker_waittime = worker_waittime.at(cur_worker);
               bool models_in_cache;
               auto& required_models = std::get<1>(pre_adfg.at(pathname));
               auto& required_models_size = std::get<2>(pre_adfg.at(pathname));
               for(size_t idx = 0; idx < required_models.size(); idx ++){
                    auto& models = worker_cached_models.at(cur_worker);
                    models_in_cache = std::find(models.begin(), models.end(), required_models[idx]) != worker_cached_models.at(cur_worker).end();
                    if(!models_in_cache){
                         /** TODO: current design assume host loaded all models at the beginning.
                          *        later can extend to remote_host_to_GPU, with the udl&model centraliezd store
                         */
                         model_fetch_time = model_fetch_time + host_to_GPU_delay(required_models_size[idx]);
                    }
               } 
               /** case 2.1 cur_woker is not the same worker as any of the pre-req tasks'
                *  input fetching/sending is not blocked by waiting queue, whereas model fetching is
                */
               uint64_t start_time;
               if(preq_workers.find(cur_worker) == preq_workers.end()){
                    start_time = std::max(prev_EST, cur_us + cur_worker_waittime + model_fetch_time);
               }else{//case 2.2 cur_worker is on the same node of one of the pre-req tasks
                    uint64_t preq_arrival_time = 0;
                    for(auto& preq_task : required_tasks){
                         node_id_t& preq_worker = std::get<0>(allocated_tasks_info.at(preq_task));
                         uint64_t& preq_finish_time = std::get<1>(allocated_tasks_info.at(preq_task));
                         if(cur_worker == preq_worker){
                              preq_arrival_time = std::max(preq_arrival_time, preq_finish_time);
                         }else{
                              preq_arrival_time = std::max(preq_arrival_time, preq_finish_time + GPU_to_GPU_delay(std::get<4>(pre_adfg.at(pathname))));
                              start_time = std::max(preq_arrival_time, cur_us + cur_worker_waittime + model_fetch_time);
                         }
                    }
               }
               workers_start_times.emplace(cur_worker,start_time);
          }
          auto it = std::min_element(workers_start_times.begin(), workers_start_times.end(),
                                                       [](const auto& l, const auto& r) { return l.second < r.second; });
          /** TODO: TEST THIS!!! https://stackoverflow.com/questions/2659248/how-can-i-find-the-minimum-value-in-a-map */
          node_id_t selected_worker = it->second;
          uint64_t cur_task_finish_time = it->first + std::get<5>(pre_adfg.at(pathname));
          allocated_tasks_info.emplace(std::piecewise_construct, std::forward_as_tuple(pathname), std::forward_as_tuple(selected_worker, cur_task_finish_time));
     }    
     std::string allocated_machines;
     for(auto& pathname: sorted_pathnames){
          allocated_machines +=  std::to_string(std::get<1>(allocated_tasks_info.at(pathname))) + ",";
     }
     return allocated_machines;
}

int main(int argc, char** argv){
     int num_workers = 5;
     if(argc > 1){
          num_workers = std::atoi(argv[1]);
     }
     int num_tasks = 5;
     if(argc > 2){
          num_tasks = std::atoi(argv[2]);
     }
     int hov = 1;
     if(argc > 2){
          hov = std::atoi(argv[2]);
     }
     std::vector<node_id_t> workers_set;
     std::unordered_map<node_id_t, uint64_t> worker_waittime;
     std::unordered_map<node_id_t, std::vector<uint32_t>> worker_cached_models;
     uint64_t cur_us = std::chrono::duration_cast<std::chrono::microseconds>(
                              std::chrono::high_resolution_clock::now().time_since_epoch())
                              .count();
     for(uint32_t i = 0; i < num_workers ; ++i){
          workers_set.emplace_back(i);
          worker_waittime.emplace(std::make_pair(i, cur_us + 10));
          worker_cached_models.emplace(std::piecewise_construct, std::forward_as_tuple(i), std::forward_as_tuple(std::vector<uint32_t>({0, 1})));
     }
     
     pre_adfg_t pre_adfg;
     
     std::vector<std::string> sorted_tasks({"task0"});
     for(int i = 0 ; i < num_tasks; ++ i){
          sorted_tasks.emplace_back(std::string("task" + std::to_string(i)));
     }
     
     std::vector<uint32_t> required_models({0});
     std::vector<uint32_t> required_models_sizes({100});
     uint32_t output_size = 10;
     uint64_t exec_time = 10;

     std::set<std::string> task0_required_objects;
     pre_adfg.emplace(std::piecewise_construct, std::forward_as_tuple("task0"), std::forward_as_tuple(std::set<std::string>(), std::vector<uint32_t>({0}), 
                    std::vector<uint32_t>({0}), sorted_tasks, 10, 10));
     
     if(hov){
          std::set<std::string> mid_tasks_required_objects({"task0"});
          std::set<std::string> end_task_required_objects;
          for(int i = 1 ; i < num_tasks - 1; ++i){
               std::string task_name = "task" + std::to_string(i);
               end_task_required_objects.emplace(task_name);
               pre_adfg.emplace(std::piecewise_construct,std::forward_as_tuple(task_name), std::forward_as_tuple(mid_tasks_required_objects, required_models, required_models_sizes, sorted_tasks, 10, 10));
          }

          std::string end_task_name = "task" + std::to_string(num_tasks - 1);
          pre_adfg.emplace(std::piecewise_construct,std::forward_as_tuple(end_task_name), std::forward_as_tuple(end_task_required_objects, required_models, required_models_sizes, sorted_tasks, 10, 10));
     
     }else{
          std::set<std::string> mid_tasks_required_objects({"task0"});
          for(int i = 1 ; i < num_tasks; ++i){
               std::string task_name = "task" + std::to_string(i);
               pre_adfg.emplace(std::piecewise_construct,std::forward_as_tuple(task_name), std::forward_as_tuple(mid_tasks_required_objects, required_models, required_models_sizes, sorted_tasks, 10, 10));
               mid_tasks_required_objects.clear();
               mid_tasks_required_objects.emplace(task_name);
          }
     }
     
     

     uint64_t sum_times = 0;
     uint64_t before_scheduler_us;
     uint64_t after_scheduler_us;
     for(int i = 0; i < 100; ++i){
          before_scheduler_us = std::chrono::duration_cast<std::chrono::microseconds>(
                              std::chrono::high_resolution_clock::now().time_since_epoch())
                              .count();
          tide_scheduler("task0", pre_adfg, workers_set, worker_waittime, worker_cached_models);
          after_scheduler_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                   std::chrono::high_resolution_clock::now().time_since_epoch())
                                   .count();
          sum_times += (after_scheduler_us - before_scheduler_us);
     }
     
     std::cout << num_workers << "workers, " << "average runtime " << (sum_times / 100) << " us"<< std::endl;
     return 0;
}