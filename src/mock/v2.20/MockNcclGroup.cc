/* 
*Copyright (c) 2024, Alibaba Group;
*Licensed under the Apache License, Version 2.0 (the "License");
*you may not use this file except in compliance with the License.
*You may obtain a copy of the License at

*   http://www.apache.org/licenses/LICENSE-2.0

*Unless required by applicable law or agreed to in writing, software
*distributed under the License is distributed on an "AS IS" BASIS,
*WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*See the License for the specific language governing permissions and
*limitations under the License.
*/
#include "MockNcclGroup.h"
#include "MockNcclChannel.h"
#include<vector>
#include<map>
#include<set>
#include <queue>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <filesystem>
#include <tuple>
#include <string>
#include <cstring>
#include <functional>
#include <list>
#include <optional>
#include <climits>
#include "MockNcclLog.h"

using namespace std;
namespace MockNccl {

  MockNcclGroup::MockNcclGroup(int _ngpus,int _gpus_per_nodes,int _TP_size,int _DP_size,int _PP_size,int _EP_size,int _DP_EP_size,std::vector<int>_NVSwitch,GPUType _gpu_type, std::string _msccl_path)
  :msccl_path(std::move(_msccl_path)),g_flow_id(0),gpu_type(_gpu_type){
    /*init groups
    */
    MockNcclLog *NcclLog = MockNcclLog::getInstance();
    if (_ngpus % _gpus_per_nodes != 0 || _ngpus / _gpus_per_nodes <= 0){
      NcclLog->writeLog(NcclLogLevel::ERROR,"The number of GPUs used is not a multiple of the number of GPUs per node.");
      return;
    }
    int all_group_idx = 0;
    int nNodes = _ngpus/_gpus_per_nodes;
    // Ethernet-only topologies have no NVSwitch entries. If present, the
    // mapping must contain one entry per server before groups index it.
    if (!_NVSwitch.empty() && _NVSwitch.size() != static_cast<size_t>(nNodes)) {
      throw std::invalid_argument("NVSwitch mapping must be empty or contain one entry per server");
    }
    int nlocalranks = _gpus_per_nodes;
    int TP_nums = _ngpus/_TP_size;
    int DP_nums = _ngpus/_DP_size;
    int PP_nums = _ngpus/_PP_size;
    int EP_nums = _ngpus/_EP_size;
    int DP_EP_nums = _ngpus/_DP_EP_size;
    if (TP_nums <= 0 || DP_nums <= 0 || PP_nums <= 0 || EP_nums <= 0 || DP_EP_nums <= 0 || (_TP_size * _DP_size * _PP_size != _ngpus) || (_EP_size * _DP_EP_size != _DP_size)){
      NcclLog->writeLog(NcclLogLevel::ERROR,"The combination of parallelism groups is incorrect.");
      std::cerr << "The combination of parallelism groups is incorrect." << std::endl;
      exit(1);
    }
    int nNodesPerTPGroup = _TP_size / nlocalranks + (_TP_size % nlocalranks > 0 ? 1 : 0);
    std::vector<int>ranks;
    std::vector<int>NVSwitchs;
    // init TP group 
    if(_TP_size>1){
      std::set<int>TPnodes;
      for(int i =0;i<TP_nums;i++){
        ranks.clear();
        TPnodes.clear();
        for(int j =0;j<_TP_size;j++){
          int rank = i*_TP_size+j;
          ranks.push_back(rank);
          GroupIndex[std::make_pair(rank, TP)] = all_group_idx;
          int node_idx = rank / _gpus_per_nodes;
          TPnodes.insert(node_idx);
        }
        NVSwitchs.clear();
        for(int idx:TPnodes){
          if (idx < 0 || idx >= (int)_NVSwitch.size()) continue;
          NVSwitchs.push_back(_NVSwitch[idx]);
          GroupIndex[std::make_pair(_NVSwitch[idx],TP)] = all_group_idx;
        }
        AllGroups[all_group_idx]=GroupInfo(all_group_idx,TP,nNodesPerTPGroup,_TP_size,ranks,NVSwitchs);
        all_group_idx ++;
      }
    }
    // init DP group
    if(_DP_size>1){
      std::set<int>DPnodes;
      for(int i =0;i<DP_nums;i++){
        ranks.clear();
        DPnodes.clear();
        for(int j =0;j<_DP_size;j++){
          int rank = i+j*DP_nums;
          ranks.push_back(rank);
          GroupIndex[std::make_pair(rank, DP)] = all_group_idx;
          int node_idx = rank/_gpus_per_nodes;
          DPnodes.insert(node_idx);
        }
        NVSwitchs.clear();
        for(int idx:DPnodes){
          if (idx < 0 || idx >= (int)_NVSwitch.size()) continue;
          NVSwitchs.push_back(_NVSwitch[idx]);
          GroupIndex[std::make_pair(_NVSwitch[idx],DP)] = all_group_idx;
        }
        AllGroups[all_group_idx]=GroupInfo(all_group_idx,DP,DPnodes.size(),_DP_size,ranks,NVSwitchs);
        all_group_idx ++;
      }
    }
    // init PP group
    if(_PP_size > 1){

    }
    // init EP
    std::map<int,GroupInfo> AllTPGroups;
    if (_TP_size > 1) {
      for (auto it = AllGroups.begin(); it != AllGroups.end(); ++it) {
        if (it->second.type == TP) {
          AllTPGroups[it->second.group_index] = it->second;
        }
      }
    } else if (_TP_size == 1) {
      // init with single-node TP groups in case TP = 1
      NVSwitchs.clear();
      for (int i = 0; i < _ngpus; i++) {
        ranks.clear();
        ranks.push_back(i);
        AllTPGroups[i] = GroupInfo(i, TP, 1, 1, ranks, NVSwitchs);
      }
    }
    if(_EP_size>1){
      int TP_idx=0;
      std::set<int> EPnodes;
      for (int i = 0; i < TP_nums / _EP_size; i++){
        TP_idx = i*_EP_size;
        for(int k = 0;k<AllTPGroups[TP_idx].Ranks.size();k++){
          ranks.clear();
          EPnodes.clear();
          for(int l = TP_idx;l<TP_idx+_EP_size;l++){
            int tmp_rank = AllTPGroups[l].Ranks[k];
            int node_idx = tmp_rank/_gpus_per_nodes;
            ranks.push_back(tmp_rank);
            GroupIndex[std::make_pair(tmp_rank, EP)] = all_group_idx;
            EPnodes.insert(node_idx);
          }
          NVSwitchs.clear();
          for(int idx:EPnodes){
            if (idx < 0 || idx >= (int)_NVSwitch.size()) continue;
            NVSwitchs.push_back(_NVSwitch[idx]);
            GroupIndex[std::make_pair(_NVSwitch[idx],EP)] = all_group_idx;
          }
          AllGroups[all_group_idx] = GroupInfo(all_group_idx,EP,EPnodes.size(),_EP_size,ranks,NVSwitchs);
          all_group_idx++;
        }
      }
    }
    //init EP_DP
    if (_DP_EP_size > 1){
      int TP_idx = 0;
      std::set<int> DP_EP_nodes;
      for (int i = 0; i < TP_nums / _DP_EP_size; i++){
        TP_idx = i;
        for (int k = 0; k < AllTPGroups[TP_idx].Ranks.size(); k++){
          ranks.clear();
          DP_EP_nodes.clear();
          for (int l = TP_idx; l < TP_idx + _DP_EP_size * _EP_size; l += _EP_size){
            int tmp_rank = AllTPGroups[l].Ranks[k];
            int node_idx = tmp_rank / _gpus_per_nodes;
            ranks.push_back(tmp_rank);
            GroupIndex[std::make_pair(tmp_rank, DP_EP)] = all_group_idx;
            DP_EP_nodes.insert(node_idx);
          }
          NVSwitchs.clear();
          for (int idx : DP_EP_nodes){
            if (idx < 0 || idx >= (int)_NVSwitch.size()) continue;
            NVSwitchs.push_back(_NVSwitch[idx]);
            GroupIndex[std::make_pair(_NVSwitch[idx], DP_EP)] = all_group_idx;
          }
          AllGroups[all_group_idx] = GroupInfo(all_group_idx, DP_EP, DP_EP_nodes.size(), _DP_EP_size, ranks, NVSwitchs);
          all_group_idx++;
        }
      }
    }

    std::cout << "*********************    AllGroups:    *********************" << std::endl;
    for (const auto& [group_index, group_info] : AllGroups) {
      std::cout << "Group#" << group_info.group_index
                << " Type=" << to_cstr(group_info.type)
                << " Nodes=" << group_info.nNodes
                << " Ranks=" << group_info.nRanks
                << " " << AstraSim::asPrintable(group_info.Ranks)
                << " NVSwitches=" << AstraSim::asPrintable(group_info.NVSwitchs)
                << "\n";
    }
    std::cout << "************************************************************" << std::endl;

    return;
  }
  
  void MockNcclGroup::generateringchannels(std::map<int, std::vector<int>> localrings, MockNccl::GroupInfo* groupInfo, std::map<int, std::map<int, std::vector<int>>>& ringchannels) {
    std::map<int,std::vector<int>>::iterator ring_it;
    int current;
    int prev;
    int next;
    int end_rank;
    int nNodes = groupInfo->nNodes;
    int nlocalRanks = groupInfo->nRanks/nNodes;
    int delta = nNodes > 1 ? groupInfo->Ranks[nlocalRanks]-groupInfo->Ranks[0] : 0;
    for(ring_it = localrings.begin();ring_it != localrings.end();ring_it++) {
      prev = -1;
      next = -1;
      for(int i = 0; i < nNodes; i++) {
        int node_send;
        int node_recv;
        node_recv = ring_it->second[0] + i * delta;
        node_send = ring_it->second[nlocalRanks-1] + i * delta;
        for(int j = 0; j < nlocalRanks; j++) {
          current = ring_it->second[j] + i * delta;  
          if (j == nlocalRanks-1) {
            next = ring_it->second[0] + (i + 1) * delta;
          } else {
            next = ring_it->second[j+1] + i * delta;
          }
          ringchannels[ring_it->first][current] = {prev,next,node_recv,node_send};
          prev = current;
        }
      }
      end_rank = ring_it->second[nlocalRanks-1] + (nNodes - 1) * delta;
      ringchannels[ring_it->first][ring_it->second[0]][0] = end_rank;
      ringchannels[ring_it->first][end_rank][1] = ring_it->second[0];

    }
  }

  std::map<int, std::vector<int>> MockNcclGroup::gen_local_ring(int rank, GroupType type){
    GroupInfo gp_info;
    int gp_idx;
    std::vector<int>ranks;
    std::vector<int>localranks;
    std::map<int,std::vector<int>>localrings;
    int nNodes;
    int nlocalranks;
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    if(GroupIndex.count(std::make_pair(rank,type)) == 0){
      NcclLog->writeLog(NcclLogLevel::ERROR,"There is no relevant group info, resulting in an error in gen_local_ring");
      return {};
    } 
    gp_idx = GroupIndex[std::make_pair(rank,type)];
    gp_info = AllGroups[gp_idx];
    ranks = gp_info.Ranks;
    nNodes = gp_info.nNodes;
    nlocalranks = ranks.size()/nNodes;
    std::sort(ranks.begin(), ranks.end());
    for (int i = 0; i < nlocalranks; i++){
      localranks.push_back(ranks[i]);
    }
    for(int i =0;i<nlocalranks;i++){
      std::vector<int> vec;
      for (int j = 0; j < nlocalranks; ++j) {
        vec.push_back(localranks[(i + j) % nlocalranks]);
      }
      localrings[i] = vec;
    }
    return localrings;
  }

  RingChannels MockNcclGroup::genringchannels(int rank, MockNccl::GroupType type) {
    std::map<int,std::map<int,std::vector<int>>>ringchannels;
    std::map<int,std::vector<int>>localrings;
    std::map<int,std::vector<int>>::iterator ring_it;
    GroupInfo gp_info;
    int gp_idx;
    MockNcclLog* NcclLog = MockNcclLog::getInstance();

    int current;
    int prev;
    int next;
    int end_rank;
    int nNodes;
    int nlocalRanks;
    int delta;
    if(GroupIndex.count(std::make_pair(rank,type))==0){
      NcclLog->writeLog(NcclLogLevel::ERROR,"No corresponding group information is generated, and there is an error in creating the ring channel.");
    }
    gp_idx = GroupIndex[std::make_pair(rank,type)];
    gp_info = AllGroups[GroupIndex[std::make_pair(rank,type)]];
    nNodes = gp_info.nNodes;
    nlocalRanks = gp_info.nRanks/nNodes;
    localrings = gen_local_ring(rank,type);

    delta = nNodes > 1 ? gp_info.Ranks[nlocalRanks]-gp_info.Ranks[0] : 0;
    for(ring_it = localrings.begin();ring_it != localrings.end();ring_it++) {
      prev = -1;
      next = -1;
      for(int i = 0; i < nNodes; i++) {
        int node_send;
        int node_recv;
        node_recv = ring_it->second[0] + i * delta;
        node_send = ring_it->second[nlocalRanks-1] + i * delta;
        for(int j = 0; j < nlocalRanks; j++) {
          current = ring_it->second[j] + i * delta;  
          if (j == nlocalRanks-1) {
            next = ring_it->second[0] + (i + 1) * delta;
          } else {
            next = ring_it->second[j+1] + i * delta;
          }
          ringchannels[ring_it->first][current] = {prev,next,node_recv,node_send};
          prev = current;
        }
      }
      end_rank = ring_it->second[nlocalRanks-1] + (nNodes - 1) * delta;
      ringchannels[ring_it->first][ring_it->second[0]][0] = end_rank;
      ringchannels[ring_it->first][end_rank][1] = ring_it->second[0];
    }
    Allringchannels[gp_idx]=ringchannels;
    return ringchannels;
  }
  
  std::shared_ptr<void> MockNcclGroup::getFlowModels(GroupType type , int rank, AstraSim::ComType op,uint64_t data_size,int layer_num,State loopstate, bool& msccl){
    std::string flow_model_name;
    GroupInfo gp_info;
    std::string flow_model_name_msccl;
    int gp_idx;
    int end_rank;
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    if(GroupIndex.count(std::make_pair(rank,type))==0){
      NcclLog->writeLog(NcclLogLevel::ERROR,"There is no corresponding group info and group ring channel, resulting in an error in generating the flow model.");
      return nullptr;
    }
    gp_idx = GroupIndex[std::make_pair(rank,type)];
    gp_info = AllGroups[gp_idx];
    switch (type){
      case TP:
        flow_model_name = "TP";
        break;
      case DP:
        flow_model_name = "DP";
        break;
      case EP:
        flow_model_name = "EP";
        break;
      case DP_EP:
        flow_model_name = "DP_EP";
        break;
      default:
        break;
    }
    flow_model_name = flow_model_name + "_" + std::to_string(gp_idx) + "_" + std::to_string(static_cast<int>(op)) + "_" + std::to_string(data_size);
    flow_model_name_msccl= flow_model_name + "_1";
    flow_model_name += "_0";
    if(flow_models.count(flow_model_name)){
      NcclLog->writeLog(
        NcclLogLevel::INFO,
        "Found flow model without msccl at rank %d for group_idx %u (%s), op %s, data_size %lu, layer_num %d, loopstate %d.",
        rank, gp_idx, to_cstr(type), to_cstr(op), data_size, layer_num, static_cast<int>(loopstate));
      FlowName2nums[flow_model_name] ++;
      std::shared_ptr<void> presult;
      msccl = false;
      if(flow_models[flow_model_name].count(rank)!=0){
      // already found without msccl
        presult = flow_models[flow_model_name][rank];
      }else{
        presult = nullptr;
      }
      return presult;
    } else if(flow_models.count(flow_model_name_msccl)){
      auto src_it = msccl_flow_model_sources.find(flow_model_name_msccl);
      const std::string msccl_src_file =
          (src_it != msccl_flow_model_sources.end()) ? src_it->second : std::string("<unknown>");
      NcclLog->writeLog(
        NcclLogLevel::INFO,
        "Found flow model with msccl at rank %u for group_idx %u (%s), op %s, data_size %lu, layer_num %d, loopstate %d, schedule file %s.",
        rank, gp_idx, to_cstr(type), to_cstr(op), data_size, layer_num, static_cast<int>(loopstate), msccl_src_file.c_str());
      // Print once per unique (op, data_size, schedule file) instead of once per rank:
      // a single line if every group uses the same schedule, plus one line for each
      // different schedule. Every rank still gets the INFO log line above.
      if (printed_msccl_cache_hits
              .insert(std::make_tuple(static_cast<int>(op), data_size, msccl_src_file))
              .second) {
        std::cout << "ID: " << rank << " Reusing cached MSCCL flow model " << flow_model_name_msccl
              << " | Layer: " << layer_num
              << " | Op: " << to_cstr(op)
              << " | DataSize: " << data_size
              << " | Schedule file: " << msccl_src_file
              << std::endl;
      }
      FlowName2nums[flow_model_name_msccl] ++;
      std::shared_ptr<void> presult;
      msccl = true;  // tell the caller these are MSCCL flows, like the non-MSCCL branch does
      if(flow_models[flow_model_name_msccl].count(rank)!=0){
        // already found with msccl
        presult = flow_models[flow_model_name_msccl][rank];
      }else{
        presult = nullptr;
      }
      return presult;
    } else {
      // Only announce the XML search when it will actually happen: MSCCL requested and the
      // collective supports it (AllReduce never uses MSCCL, see genAllReduceFlowModels).
      if (msccl) {
        std::cout << "ID: " << rank << " Searching for MSCCL XML algorithms in folder: " << msccl_path
              << " | Layer: " << layer_num
              << " | Op: " << to_cstr(op)
              << " | DataSize: " << data_size
              << std::endl;
      }
      auto result = genFlowModels(type,rank,op,data_size,msccl);
      NcclLog->writeLog(
          NcclLogLevel::INFO,
          "Generated flow model %s at rank %u for group_idx %d (%s), op %s, data_size %lu, layer_num %d,"
          " loopstate %d, for a total of %u flows.",
          msccl ? "with msccl" : "without msccl", rank, gp_info.group_index, to_cstr(gp_info.type),
          to_cstr(op), data_size, layer_num, static_cast<int>(loopstate), countFlowsInFlowModels(result));
      // Switch: SIMAI_DUMP_DETAILED_FLOWS (default "1" = enabled, preserves existing behavior).
      // Set to "0" to skip generating ncclFlowModel_detailed_flows.csv. This dump only
      // reads flow_models and writes a file; it never mutates simulation state, so the
      // end-to-end result must be identical whether it is enabled or disabled.
      static int dump_detailed = [](){
          const char* e = std::getenv("SIMAI_DUMP_DETAILED_FLOWS");
          if (!e) e = std::getenv("AS_DUMP_DETAILED_FLOWS");  // fallback legacy name
          return (e==nullptr || strcmp(e,"0")!=0) ? 1 : 0;
      }();
      if(msccl){
        //delete the old flow model without msccl
        if(flow_models.count(flow_model_name)){
          flow_models.erase(flow_model_name);
          FlowName2nums.erase(flow_model_name);
        }
        //add the new flow model with msccl
        flow_models[flow_model_name_msccl] = result;
        FlowName2nums[flow_model_name_msccl]= 1;
        // remember which schedule file this cached flow model came from
        msccl_flow_model_sources[flow_model_name_msccl] = last_msccl_schedule_file;
        if (dump_detailed) {
          // Obtain algo/proto info for CSV provenance columns
          struct ncclInfo* dump_info = get_algo_proto_info(type, rank, op, data_size, msccl);
          dumpDetailedFlowModels(flow_model_name_msccl, op, data_size, flow_models[flow_model_name_msccl], dump_info);
        }
        return flow_models[flow_model_name_msccl][rank];
      }else{
        flow_models[flow_model_name] = result;
        FlowName2nums[flow_model_name]= 1;
        if (dump_detailed) {
          // Obtain algo/proto info for CSV provenance columns
          struct ncclInfo* dump_info = get_algo_proto_info(type, rank, op, data_size, msccl);
          dumpDetailedFlowModels(flow_model_name, op, data_size, flow_models[flow_model_name], dump_info);
        }
        return flow_models[flow_model_name][rank];
      }
    }
  }

  unsigned int MockNcclGroup::countFlowsInFlowModels(const map<int, shared_ptr<FlowModels>>& flowModelsMap) {
    unsigned int count = 0;
    for (auto& [_, flowModels] : flowModelsMap) {
      count += flowModels->size();
    }
    return count/2;
  }

  std::map<int,std::shared_ptr<FlowModels>> MockNcclGroup::genFlowModels(GroupType type, int rank, AstraSim::ComType op,uint64_t data_size,bool& msccl){
    // RING/MSCCL
    switch (op) {
      case AstraSim::ComType::All_Reduce:
        return genAllReduceFlowModels(type,rank,data_size,msccl);
      case AstraSim::ComType::All_Gather:
        return genAllGatherFlowModels(type,rank,data_size,msccl);
      case AstraSim::ComType::Reduce_Scatter:
        return genReduceScatterFlowModels(type,rank,data_size,msccl);
      case AstraSim::ComType::All_to_All:
        return genAlltoAllFlowModels(type,rank,data_size,msccl);
      default:
        break;
    }
    return {};
  }

  void logRankToFlowModels(
    const NcclLogLevel level,
    const std::string& algorithmName,
    const std::map<int, std::shared_ptr<FlowModels>>& rank2pflowmodels) {

    MockNcclLog* NcclLog = MockNcclLog::getInstance();

    // Header for the log output
    NcclLog->writeLog(level, "=======================================================================");
    NcclLog->writeLog(level, "--- Flow Model Details for: %s ---", algorithmName.c_str());
    NcclLog->writeLog(level, "=======================================================================");

    // Iterate over each rank that has associated flows.
    for (const auto& rank_pair : rank2pflowmodels) {
        int rank = rank_pair.first;
        const auto& models_ptr = rank_pair.second;

        if (!models_ptr) continue;

        std::stringstream header_ss;
        header_ss << "--- Flows visible to Rank " << rank << " ---";
        NcclLog->writeLog(level, "%s", header_ss.str().c_str());

        // Copy flows to a vector to sort them by flow_id for readability.
        std::vector<SingleFlow> sorted_flows;
        for (const auto& flow_pair : *models_ptr) {
            sorted_flows.push_back(flow_pair.second);
        }
        std::sort(sorted_flows.begin(), sorted_flows.end(),
                  [](const SingleFlow& a, const SingleFlow& b) {
                      return a.flow_id < b.flow_id;
                  });

        // Log the details of each sorted flow.
        for (const auto& flow : sorted_flows) {
            std::stringstream flow_ss;
            flow_ss << "  Flow ID: " << flow.flow_id
                    << "\t| Type: " << flow.conn_type
                    << "\t| Ch: " << flow.channel_id
                    << "\t| Chunk ID: " << flow.chunk_id
                    << "\t| " << flow.src << " -> " << flow.dest
                    << "\t| Size: " << flow.flow_size
                    << "\t| Parents: " << AstraSim::asPrintable(flow.parent_flow_id)
                    << "\t| Children: " << AstraSim::asPrintable(flow.child_flow_id)
                    << "\t| prev: " << AstraSim::asPrintable(flow.prev);
            NcclLog->writeLog(level, "%s", flow_ss.str().c_str());
        }
    }
    NcclLog->writeLog(level, "=======================================================================");
  }

  // --- Helper Struct for NCCL Constants and Dummy Values ---
  // Single place for every number the mock uses to pick algorithms, protocols and
  // pipeline sizes. Values marked "env" are defaults that NcclSelectionParams
  // below lets you override at runtime; everything else is fixed at compile time.
  struct NcclSimulationConstants {
      // ===================== Buffers / pipelining (MSCCL) ======================
      // values taken from init.cc and primitives.h of MSCCL
      // From init.cc: #define DEFAULT_BUFFSIZE (1 << 22) /* 4MiB */
      static constexpr uint64_t BUFFSIZE_SIMPLE = 4 * 1024 * 1024;

      // From init.cc: #define DEFAULT_LL_BUFFSIZE (NCCL_LL_LINES_PER_THREAD*NCCL_LL_MAX_NTHREADS*NCCL_STEPS*sizeof(union ncclLLFifoLine))
      // Dummy values for a typical configuration: 8 lines * 512 threads * 8 steps * 16 bytes/line
      static constexpr uint64_t BUFFSIZE_LL = 8 * 512 * 8 * 16; // Approx 512 KiB

      // From init.cc: #define DEFAULT_LL128_BUFFSIZE (NCCL_LL128_ELEMS_PER_THREAD*NCCL_LL128_MAX_NTHREADS*NCCL_STEPS*sizeof(uint64_t))
      // Dummy values: 28 elems * 256 threads * 8 steps * 8 bytes/elem
      static constexpr uint64_t BUFFSIZE_LL128 = 28 * 256 * 8 * 8; // Approx 448 KiB

      // From common.h (or assumed default)
      static constexpr int NCCL_STEPS = 8;

      static constexpr int MSCCL_CHUNKSTEPS = NCCL_STEPS / 2; // Value is 4

      // From common.h for LL128 protocol
      static constexpr int NCCL_LL128_DATAELEMS = 14;
      static constexpr int NCCL_LL128_LINEELEMS = 16;

      // ===================== Algorithm selection ===============================
      // NVLS AllReduce: TP groups only, on NVLS-capable GPUs, with AS_NVLS_ENABLE=1.
      // Real machine (H20x8, NCCL 2.26.5/2.30.7): RING for <=1MB, NVLS for >=4MB.
      static constexpr int      NVLS_MIN_RANKS = 8;
      static constexpr uint64_t NVLS_MIN_BYTES = 2ULL << 20;      // env AS_NVLS_MIN_BYTES (2MB)
      static constexpr bool     NVLS_ENABLE    = false;           // env AS_NVLS_ENABLE ("1" = on)
      static constexpr bool isNvlsCapable(GPUType t) {
          return t == GPUType::H100 || t == GPUType::H800 || t == GPUType::H20;
      }
      // AllGather / ReduceScatter / AllToAll: always RING in v2.20 (no PAT).

      // ===================== Protocol selection ================================
      // Shared protocol table. MSCCL always splits data into pipeline chunks with the
      // Simple sizing (BUFFSIZE_SIMPLE based; the smaller LL/LL128 chunks lose efficiency
      // to the TCP window), whatever protocol is selected here. For MSCCL the selected
      // protocol (this table, or SIMAI_FORCE_PROTO) only sets AS_SEND_LAT; UNDEF
      // (SIMAI_PROTO_AWARE=0) disables pipelining altogether.
      // NVLS / NVLS_TREE: LL up to this size, Simple above.
      static constexpr uint64_t NVLS_LL_MAX_BYTES    = 1ULL << 20;   // 1MB
      // Ring-class (RING/TREE): LL -> LL128 -> Simple. (H20x8 calib_260611)
      static constexpr uint64_t RING_LL_MAX_BYTES    = 4ULL << 20;   // 4MB
      static constexpr uint64_t RING_LL128_MAX_BYTES = 16ULL << 20;  // 16MB

      static constexpr bool PROTO_AWARE  = true;                  // env SIMAI_PROTO_AWARE / AS_PROTO_AWARE ("0" = UNDEF)
      static constexpr int  FORCED_PROTO = -1;                    // env SIMAI_FORCE_PROTO = LL | LL128 | Simple
  };

  // Runtime view of the constants above: compile-time default, overridden by the
  // environment. Resolved once, on first use.
  struct NcclSelectionParams {
      bool     nvls_enable;
      uint64_t nvls_min_bytes;
      bool     proto_aware;
      int      forced_proto;

      static const NcclSelectionParams& get() {
          static const NcclSelectionParams p = [] {
              using C = NcclSimulationConstants;
              auto env = [](const char* name) { return std::getenv(name); };
              auto envU64 = [&](const char* name, uint64_t def) {
                  const char* e = env(name);
                  return e ? strtoull(e, nullptr, 10) : def;
              };
              NcclSelectionParams r;
              const char* e = env("AS_NVLS_ENABLE");
              r.nvls_enable    = e ? (strcmp(e, "1") == 0) : C::NVLS_ENABLE;
              r.nvls_min_bytes = envU64("AS_NVLS_MIN_BYTES", C::NVLS_MIN_BYTES);
              e = env("SIMAI_PROTO_AWARE");
              if (!e) e = env("AS_PROTO_AWARE");  // legacy name
              r.proto_aware    = e ? (strcmp(e, "0") != 0) : C::PROTO_AWARE;
              r.forced_proto   = C::FORCED_PROTO;
              if ((e = env("SIMAI_FORCE_PROTO"))) {
                  if (strcmp(e, "LL") == 0)          r.forced_proto = NCCL_PROTO_LL;
                  else if (strcmp(e, "LL128") == 0)  r.forced_proto = NCCL_PROTO_LL128;
                  else if (strcmp(e, "Simple") == 0) r.forced_proto = NCCL_PROTO_SIMPLE;
              }
              return r;
          }();
          return p;
      }
  };

  // ---- MSCCL helpers (ported from GitLab) ----
  // Recursively follows one chunk unit through s/rcs steps across peers and
  // appends every (gpu, tb, step, offset) hop to path.
void trace_chunk_recursive(
    const MscclAlgorithm& algo,
    int gpu_id,
    int tb_id,
    int step_id,
    int step_offset_k,
    std::list<std::tuple<int, int, int, int>>& path) { // New: tuple includes offset

    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    const auto& current_step = algo.gpus.at(gpu_id).thread_blocks[tb_id].steps[step_id];

    NcclLog->writeLog(NcclLogLevel::DEBUG, "Tracing GPU %d, TB %d, Step %d, Offset %d, Type %s",
        gpu_id, tb_id, step_id, step_offset_k, current_step.type.c_str());

    if (current_step.type == "s" || current_step.type == "rcs") {
        // 1. Save the current combo (GPU, TB, Step, Offset) to the path
        path.push_back(std::make_tuple(gpu_id, tb_id, step_id, step_offset_k));

        int peer_gpu = algo.gpus.at(gpu_id).thread_blocks[tb_id].send_peer;
        if (peer_gpu == -1) return;

        // Calculate the specific global offset for this sub-chunk
        int global_dst_offset = current_step.dst_offset + step_offset_k;

        const auto& peer = algo.gpus.at(peer_gpu);
        for (const auto& peer_tb : peer.thread_blocks) {
            for (int p_step_idx = 0; p_step_idx < peer_tb.steps.size(); ++p_step_idx) {
                const auto& peer_step = peer_tb.steps[p_step_idx];

                // Match the consumer on the peer using range checks
                if (peer_step.src_buffer == current_step.dst_buffer &&
                    global_dst_offset >= peer_step.src_offset &&
                    global_dst_offset < peer_step.src_offset + peer_step.count) {

                    // Calculate relative offset in the peer step
                    int next_k = global_dst_offset - peer_step.src_offset;

                    trace_chunk_recursive(algo, peer_gpu, peer_tb.id, p_step_idx, next_k, path);
                }
            }
        }
    }
}


  // Checks the generated MSCCL flow DAG: no cycles, every flow reachable from a
  // root, and (for allgather) each GPU receives the right chunks and bytes.
bool MockNcclGroup::validate_flow_graph(const FlowModels& flows,
                                        const MscclAlgorithm& msccl_algorithm,
                                        uint64_t data_size,
                                        int total_unique_chunks,
                                        MockNcclLog* NcclLog,
                                        int protocol) {
    if (flows.empty()) {
        if (data_size == 0) {
            NcclLog->writeLog(NcclLogLevel::DEBUG, "Flow graph is empty and data_size is 0, no validation needed.");
            return true;
        } else {
            NcclLog->writeLog(NcclLogLevel::ERROR, "[VALIDATION] Flow graph is empty, but data_size > 0. This is an error.");
            return false;
        }
    }

    std::map<uint64_t, std::vector<uint64_t>> adj;
    std::map<uint64_t, const SingleFlow*> flow_map;
    std::set<uint64_t> all_flow_ids;

    for (const auto& pair : flows) {
        const auto& flow = pair.second;
        adj[flow.flow_id] = std::vector<uint64_t>(flow.child_flow_id.begin(), flow.child_flow_id.end());
        flow_map[flow.flow_id] = &flow;
        all_flow_ids.insert(flow.flow_id);
    }

    // --- Cycle Detection using DFS ---
    enum class NodeState { Unvisited, Visiting, Visited };
    std::map<uint64_t, NodeState> visited_state;
    for (const auto& id : all_flow_ids) {
        visited_state[id] = NodeState::Unvisited;
    }

    std::function<bool(uint64_t)> detect_cycle_util =
        [&](uint64_t u) -> bool {
        visited_state[u] = NodeState::Visiting;
        if (adj.count(u)) {
            for (uint64_t v : adj.at(u)) {
                if (visited_state.count(v) == 0) {
                    NcclLog->writeLog(NcclLogLevel::ERROR, "[VALIDATION] Schedule %s - Flow %lu has a child dependency on non-existent flow %lu.", msccl_algorithm.filename.c_str(), u, v);
                    return true;
                }
                if (visited_state.at(v) == NodeState::Visiting) {
                    NcclLog->writeLog(NcclLogLevel::ERROR, "[VALIDATION] Schedule %s - Flow %lu has a circular dependency with flow %lu.", msccl_algorithm.filename.c_str(), u, v);
                    return true;
                }
                if (visited_state.at(v) == NodeState::Unvisited) {
                    if (detect_cycle_util(v)) {
                      NcclLog->writeLog(NcclLogLevel::ERROR, "[VALIDATION] Schedule %s - Flow %lu has a circular dependency with flow %lu.", msccl_algorithm.filename.c_str(), u, v);
                      return true;
                    }
                }
            }
        }
        visited_state[u] = NodeState::Visited;
        return false;
    };

    for (const auto& id : all_flow_ids) {
        if (visited_state.at(id) == NodeState::Unvisited) {
            if (detect_cycle_util(id)) {
                NcclLog->writeLog(NcclLogLevel::ERROR, "[VALIDATION] Schedule %s - Flow graph contains circular dependencies.", msccl_algorithm.filename.c_str());
                return false;
            }
        }
    }
    NcclLog->writeLog(NcclLogLevel::INFO, "No circular dependencies found in the flow graph.");

    // --- Unreachable Node Detection ---
    std::set<uint64_t> root_nodes;
    for (const auto& id : all_flow_ids) {
        if (flow_map.at(id)->parent_flow_id.empty()) {
            root_nodes.insert(id);
        }
    }

    if (root_nodes.empty() && !all_flow_ids.empty()) {
        NcclLog->writeLog(NcclLogLevel::ERROR, "[VALIDATION] Schedule %s - No root nodes found, indicating a fully circular or disconnected graph.", msccl_algorithm.filename.c_str());
        return false;
    }

    std::set<uint64_t> reachable_nodes;
    std::function<void(uint64_t)> find_reachable_util =
        [&](uint64_t u) {
        reachable_nodes.insert(u);
        if (adj.count(u)) {
            for (uint64_t v : adj.at(u)) {
                if (reachable_nodes.find(v) == reachable_nodes.end()) {
                    find_reachable_util(v);
                }
            }
        }
    };

    for (const auto& root_id : root_nodes) {
        if (reachable_nodes.find(root_id) == reachable_nodes.end()) {
            find_reachable_util(root_id);
        }
    }

    if (reachable_nodes.size() < all_flow_ids.size()) {
        NcclLog->writeLog(NcclLogLevel::ERROR, "[VALIDATION] Schedule %s - Some nodes are unreachable from root nodes.", msccl_algorithm.filename.c_str());
        for (const auto& id : all_flow_ids) {
            if (reachable_nodes.find(id) == reachable_nodes.end()) {
                NcclLog->writeLog(NcclLogLevel::ERROR, " -> Unreachable Flow ID: %lu", id);
            }
        }
        return false;
    }
    NcclLog->writeLog(NcclLogLevel::INFO, "All nodes are reachable from root nodes.");

    // --- Collective-Specific Data Integrity Validation ---
    if (msccl_algorithm.coll_type == "allgather") {
        NcclLog->writeLog(NcclLogLevel::INFO, "--- Performing AllGather Data Integrity Validation ---");
        int n_gpus = msccl_algorithm.n_gpus;

        // 1. Correctly determine the data size being validated, mirroring the generator.
        // The generator simulates ONE loop. We must validate against the data size for that single loop.
        const uint64_t data_size_per_chunk  = data_size / msccl_algorithm.nchunksperloop;
        NcclLog->writeLog(NcclLogLevel::DEBUG, "Validating against data size for a single loop: %lu bytes", data_size_per_chunk );

        // 2. Setup validation checks using the CORRECT data size
        const uint64_t expected_total_received_size = ((n_gpus > 1) ? (uint64_t)(n_gpus - 1) * data_size_per_chunk  : 0) * (msccl_algorithm.nchunksperloop/msccl_algorithm.n_gpus);

        if (total_unique_chunks % n_gpus != 0) {
            NcclLog->writeLog(NcclLogLevel::ERROR, "[VALIDATION] Schedule %s - Total unique chunks %d is not divisible by number of GPUs %d.", msccl_algorithm.filename.c_str(), total_unique_chunks, n_gpus);
            return false;
        }

        size_t chunks_per_rank = total_unique_chunks / n_gpus;
        size_t data_size_per_rank = data_size_per_chunk  * chunks_per_rank;
        NcclLog->writeLog(NcclLogLevel::INFO, "Each GPU is expected to receive %zu unique chunks, totaling %lu bytes.", chunks_per_rank, data_size_per_rank);
        NcclLog->writeLog(NcclLogLevel::INFO, "Each Chunk is expected to be %lu bytes.", data_size_per_chunk);

        // Track what each GPU receives
        std::map<int, std::set<int>> gpu_received_chunks;   // Key: dest_gpu, Val: set of unique chunk IDs
        std::map<int, uint64_t> gpu_received_size;        // Key: dest_gpu, Val: total bytes received
        // find the data size per transfer based on the protocol
        size_t pipeline_chunk_size=-1;
        if (protocol == NCCL_PROTO_SIMPLE) {
            // Formula from primitives.h: ncclShmem.comm.buffSizes[NCCL_PROTO_SIMPLE] / NCCL_STEPS;
            const uint64_t byte_per_step = NcclSimulationConstants::BUFFSIZE_SIMPLE / NcclSimulationConstants::NCCL_STEPS;
            pipeline_chunk_size = byte_per_step * NcclSimulationConstants::MSCCL_CHUNKSTEPS;
        } else if (protocol == NCCL_PROTO_LL) {
            // Formula from primitives.h: ncclShmem.comm.buffSizes[NCCL_PROTO_LL] / NCCL_STEPS / 2;
            pipeline_chunk_size = (NcclSimulationConstants::BUFFSIZE_LL / NcclSimulationConstants::NCCL_STEPS) / 2;
        } else { // Assuming LL128 or other fine-grained protocol
            // Formula from primitives.h: (ncclShmem.comm.buffSizes[NCCL_PROTO_LL128]/NCCL_STEPS)*NCCL_LL128_DATAELEMS/NCCL_LL128_LINEELEMS;
            pipeline_chunk_size = (NcclSimulationConstants::BUFFSIZE_LL128 / NcclSimulationConstants::NCCL_STEPS)
                                  * NcclSimulationConstants::NCCL_LL128_DATAELEMS / NcclSimulationConstants::NCCL_LL128_LINEELEMS;
        }
        pipeline_chunk_size = std::min(pipeline_chunk_size, data_size_per_chunk);
        auto pipeline_leftover_size = data_size_per_chunk % pipeline_chunk_size;
        // Populate trackers by iterating through all generated flows
        NcclLog->writeLog(NcclLogLevel::INFO, "Each transfer will be split into chunks of size %lu bytes.", pipeline_chunk_size);
        for (const auto& [key, flow] : flows) {
            bool is_pxn_flow = (flow.conn_type == "PXN");
            if(is_pxn_flow){
                // PXN flows do not carry actual data for AllGather, skip them
                continue;
            }
            if (flow.src != flow.dest) { // In AllGather, we only care about inter-gpu traffic
                //for each chunk using cnt
                size_t chunk_count = -1;
                if (flow.flow_size % pipeline_chunk_size == 0) {
                    chunk_count = (flow.flow_size + pipeline_chunk_size - 1) / pipeline_chunk_size;
                } else {
                    // FIX: Check if pipeline_leftover_size is 0 before dividing
                    if (pipeline_leftover_size > 0) {
                        chunk_count = (flow.flow_size + pipeline_leftover_size - 1) / pipeline_leftover_size;
                    } else {
                        // If leftover is 0 but we are here, treat it as 1 chunk to avoid crash
                        chunk_count = 1;
                    }
                }
                NcclLog->writeLog(NcclLogLevel::DEBUG, "Flow ID: %lu, Src: %d, Dest: %d, Size: %lu, Chunk ID: %d, Chunk Count: %d", flow.flow_id, flow.src, flow.dest, flow.flow_size, flow.chunk_id, chunk_count);

                // Fixed: Do not loop over sub-chunks. The `chunk_id` is unique for the entire flow/block.
                if(gpu_received_chunks[flow.dest].count(flow.chunk_id) == 0) {
                    NcclLog->writeLog(NcclLogLevel::DEBUG, "[VALIDATION] Schedule %s - Flow %lu has received chunk ID %d for destination GPU %d.", msccl_algorithm.filename.c_str(), flow.flow_id, flow.chunk_id, flow.dest);
                    gpu_received_chunks[flow.dest].insert(flow.chunk_id);
                }else{
                    NcclLog->writeLog(NcclLogLevel::ERROR, "[VALIDATION] Schedule %s - Flow %lu has received chunk ID %d for destination GPU %d 2 times.", msccl_algorithm.filename.c_str(), flow.flow_id, flow.chunk_id, flow.dest);
                    return false;
                }

                gpu_received_size[flow.dest] += flow.flow_size;
            }
        }

        // Now, validate each GPU
        for (const auto& [i, _] : msccl_algorithm.gpus) {
            // Check 2: Is the total received data volume correct?
            if (gpu_received_size[i] != expected_total_received_size) {
                NcclLog->writeLog(NcclLogLevel::ERROR, "[VALIDATION] Schedule %s - GPU %d received %lu bytes, but expected %lu bytes.", msccl_algorithm.filename.c_str(), i, gpu_received_size[i], expected_total_received_size);
                return false;
            }

            // Check 3: Is the number of unique chunks received correct?
            // Each of the N-1 peers sends `chunks_per_rank` unique chunks.
            size_t expected_chunks_per_dest = (n_gpus - 1) * chunks_per_rank;
            if (gpu_received_chunks[i].size() != expected_chunks_per_dest) {
                 NcclLog->writeLog(NcclLogLevel::ERROR, "[VALIDATION] Schedule %s - GPU %d received %zu unique chunks, but expected %zu unique chunks.", msccl_algorithm.filename.c_str(), i, gpu_received_chunks[i].size(), expected_chunks_per_dest);
                 return false;
            }
        }
        NcclLog->writeLog(NcclLogLevel::INFO, "AllGather data integrity validation successful: all GPUs receive the correct data from all peers.");
    }

    NcclLog->writeLog(NcclLogLevel::INFO, "Flow graph validation successful.");
    return true;
  }

  // common generation algorithms
  std::optional<std::map<int, std::shared_ptr<FlowModels>>>
  MockNcclGroup::genCustomFlowModels(GroupType type, int rank, uint64_t data_size, ncclInfo* info) {
    FlowModels result = {};
    std::map<int,FlowModels>rank2flowmodels;
    std::map<int,std::shared_ptr<FlowModels>>rank2pflowmodels;
    GroupInfo gp_info;
    int gp_idx;
    MscclAlgorithm msccl_algorithm;
    MockNcclLog* NcclLog = MockNcclLog::getInstance();

    NcclLog->writeLog(NcclLogLevel::DEBUG,"Generating flow model for Custom Collective with type: %d, rank: %d, data_size: %lu, protocol: %d", type, rank, data_size, info->protocol);

    if(GroupIndex.count(std::make_pair(rank,type))==0){
      NcclLog->writeLog(NcclLogLevel::ERROR,"There is no corresponding group info, resulting in an error in generating the flow model.");
      return {};
    } else {
      gp_idx = GroupIndex[std::make_pair(rank,type)];
      gp_info = AllGroups[gp_idx];
    }

    // --- PXN Configuration ---
    bool PXN_ENABLE = false;
    const char* PXN_ENV = std::getenv("AS_PXN_ENABLE");
    if (PXN_ENV && strcmp(PXN_ENV, "1") == 0) {
      PXN_ENABLE = true;
    }
    int nlocalRanks = (gp_info.nNodes > 0) ? (gp_info.nRanks / gp_info.nNodes) : 0;

    // Collect schedule files
    std::vector<std::filesystem::path> schedule_files;
    std::filesystem::directory_options options =
        std::filesystem::directory_options::follow_directory_symlink |
        std::filesystem::directory_options::skip_permission_denied;
    try {
      for (auto it = std::filesystem::recursive_directory_iterator(msccl_path, options);
           it != std::filesystem::recursive_directory_iterator(); ++it) {

        if (it->is_directory() && it->path().filename() == "disabled") {
          it.disable_recursion_pending();
          continue;
        }

        if (it->is_regular_file() && it->path().extension() == ".xml") {
          schedule_files.push_back(it->path());
        }
           }
    } catch (const std::filesystem::filesystem_error& e) {
      NcclLog->writeLog(NcclLogLevel::ERROR, "Filesystem error: %s. Current path: %s", e.what(), std::filesystem::current_path().string().c_str());
      return std::nullopt;
    }

    // Sort by path (alphabetical order is used to enforce schedule priority)
    std::sort(schedule_files.begin(), schedule_files.end(), [this](const auto& a, const auto& b) {
        auto rel_a = a.lexically_relative(msccl_path);
        auto rel_b = b.lexically_relative(msccl_path);
        const size_t depth_a = std::distance(rel_a.begin(), rel_a.end());
        const size_t depth_b = std::distance(rel_b.begin(), rel_b.end());
        // Process deeper paths first, then sort lexicographically
        if (depth_a != depth_b) return depth_a > depth_b;
        return a < b;
    });

    // Find and load the correct MSCCL algorithm
    bool found_algo = false;
    last_msccl_schedule_file.clear();
    NcclLog->writeLog(NcclLogLevel::DEBUG, "Searching for MSCCL XML algorithms in folder: %s", msccl_path.c_str());
    NcclLog->writeLog(
        NcclLogLevel::DEBUG,
        "Requirements are coll_type: %s, data_size: %lu, n_gpus: %d, group_type: %s, group_idx: %d",
        to_cstr(info->coll), data_size, gp_info.nRanks, to_cstr(gp_info.type),
        gp_info.group_index);

    // Group that owns a schedule's first_rank_id for this group type, or -1. Uses find()
    // so probing a rank that is not in this group type does not insert a bogus entry.
    auto group_of_rank = [this, &gp_info](int r) -> int {
        auto it = GroupIndex.find(std::make_pair(r, gp_info.type));
        return it == GroupIndex.end() ? -1 : it->second;
    };
    for (const auto& path : schedule_files) {
        MscclAlgorithm tmp = parseMscclXmlHeader(path.string());
        NcclLog->writeLog(NcclLogLevel::DEBUG, " - Parsed MSCCL XML algorithm root at %s", path.c_str());
        NcclLog->writeLog(NcclLogLevel::DEBUG, " - Algorithm attributes: n_gpus: %d, coll_type: %s, minBytes: %lu, maxBytes: %lu, group_type: %s, group_idx: %d",
                          tmp.n_gpus, tmp.coll_type.c_str(), tmp.minBytes, tmp.maxBytes, tmp.group_type.c_str(), tmp.first_rank_id >= 0 ? group_of_rank(tmp.first_rank_id) : -1);
        if (tmp.coll_type == to_cstr(info->coll)
                && data_size >= tmp.minBytes
                && (data_size <= tmp.maxBytes || tmp.maxBytes==0)
                && tmp.nchunksperloop > 0
                && tmp.n_gpus == gp_info.nRanks
                && (tmp.group_type == to_cstr(NONE) || tmp.group_type == to_cstr(gp_info.type))
                && (tmp.first_rank_id < 0 || group_of_rank(tmp.first_rank_id) == gp_info.group_index)) {
            NcclLog->writeLog(NcclLogLevel::DEBUG, " - Attributes match, parsing whole algorithm...", path.c_str());
            msccl_algorithm = parseMscclXml(path.string());
            found_algo = true;
            last_msccl_schedule_file = path.string();
            NcclLog->writeLog(NcclLogLevel::INFO, "Found matching MSCCL XML algorithm: %s", path.string().c_str());
            std::cout << "Found matching MSCCL XML algorithm for Group#" << gp_info.group_index
                      << " (" << gp_info.type << "): "
                      << path.string() << std::endl;
            break;
        }
    }

    if (!found_algo) {
        NcclLog->writeLog(NcclLogLevel::INFO, "No matching MSCCL XML algorithm found for the given data_size. Fallback required.");
        std::cout << "No matching MSCCL XML algorithm found for Group#" << gp_info.group_index
                  << " (" << gp_info.type << "). Falling back to NCCL algorithm selection (NcclFlowModel)."
                  << std::endl;
        return std::nullopt;
    }
    NcclLog->writeLog(NcclLogLevel::DEBUG, "Using MSCCL algorithm: %s for %s with data_size: %lu", msccl_algorithm.filename.c_str(), to_cstr(info->coll), data_size);

    // Map algorithm to ranks of the current group
    if (msccl_algorithm.first_rank_id < 0) {
        NcclLog->writeLog(NcclLogLevel::DEBUG, "Mapping schedule to group rank ids...");
        // TODO ensure gpu ids in the schedule are always in a [0, N-1] range when first_rank_id is not set
        // Copy every header field (n_gpus included: validate_flow_graph divides by it),
        // then rebuild only the per-GPU part with group rank ids.
        MscclAlgorithm mapped_algorithm = msccl_algorithm;
        mapped_algorithm.gpus.clear();
        for (auto const& [gpu_id, gpu] : msccl_algorithm.gpus) {
            int mapped_gpu_id = gp_info.Ranks[gpu_id];
            MscclGpu mapped_gpu;
            mapped_gpu.id = mapped_gpu_id;
            for (auto const& tb : gpu.thread_blocks) {
                MscclThreadBlock mapped_tb;
                mapped_tb.id = tb.id;
                mapped_tb.send_peer = tb.send_peer >= 0 ? gp_info.Ranks[tb.send_peer] : tb.send_peer;
                mapped_tb.recv_peer = tb.recv_peer >= 0 ? gp_info.Ranks[tb.recv_peer] : tb.recv_peer;
                mapped_tb.channel = tb.channel;
                mapped_tb.steps = tb.steps;
                mapped_gpu.thread_blocks.push_back(mapped_tb);
            }
            mapped_algorithm.gpus[mapped_gpu_id] = mapped_gpu;
        }
        msccl_algorithm = mapped_algorithm;
    }

    // =================================================================================
    // 1. Pipeline Sizing Logic
    // =================================================================================
    uint64_t pipeline_chunk_size;
    NcclLog->writeLog(NcclLogLevel::INFO, "MSCCL Algorithm specifies %d chunks per loop for data size %lu",
                      msccl_algorithm.nchunksperloop, data_size);
    const uint64_t total_data_per_algo_loop = data_size / msccl_algorithm.nchunksperloop;

    if (info->protocol != NCCL_PROTO_UNDEF) {
        // the traffic is always shaped with SIMPLE currently, as the other protocols are not emulated. The only reason for having them is setting the as_send_lat.
        const uint64_t byte_per_step = NcclSimulationConstants::BUFFSIZE_SIMPLE / NcclSimulationConstants::NCCL_STEPS;
        pipeline_chunk_size = byte_per_step * NcclSimulationConstants::MSCCL_CHUNKSTEPS / 4; // 512KB
    } else {
        pipeline_chunk_size = UINT64_MAX;
    }

    if (pipeline_chunk_size > total_data_per_algo_loop) {
        pipeline_chunk_size = total_data_per_algo_loop;
    }

    // reverted to nccl_proto to work the same for everyone whatever is the chunk size
    // pipeline_chunk_size = total_data_per_algo_loop;

    int num_pipeline_chunks = (total_data_per_algo_loop + pipeline_chunk_size - 1) / pipeline_chunk_size;
    NcclLog->writeLog(NcclLogLevel::INFO, "Generating %d Pipeline Stages (Chunk Size: %lu, from Total Data: %lu)",
                      num_pipeline_chunks, pipeline_chunk_size, total_data_per_algo_loop);


    // =================================================================================
    // 2. Identification of Logical Flows (Spatial Chunks)
    // =================================================================================
    // Trace the logical dependencies (DAG) of the algorithm once.
    std::map<int, std::list<std::tuple<int, int, int, int>>> logical_chunk_paths;
    std::map<std::tuple<int, std::string, int>, int> source_data_to_chunk_id;

    size_t unique_logical_chunk_count = 0;

    for (auto const& [gpu_id, gpu] : msccl_algorithm.gpus) {
        size_t max_steps = 0;
        for (const auto& tb : gpu.thread_blocks) {
            max_steps = std::max(max_steps, tb.steps.size());
        }
        for (size_t step_idx = 0; step_idx < max_steps; ++step_idx) {
            for (auto const& tb : gpu.thread_blocks) {
                // Guard: Not all TBs have the same number of steps
                if (step_idx >= tb.steps.size()) continue;
                const auto& step = tb.steps[step_idx];

                // Check 1: Explicit dependency defined in XML
                bool has_no_explicit_dep = (step.dep_id == -1);

                // Check 2: Implicit sequential dependency within the same TB.
                // If a previous step (like 'r', 'rcs', 'cpy') wrote to the same buffer area
                // that this 's' step is reading from, it is NOT a source, but a forwarder.
                bool has_implicit_dep = false;
                if (has_no_explicit_dep && step.type == "s") {
                    for (int prev_idx = step_idx - 1; prev_idx >= 0; --prev_idx) {
                        const auto& prev_step = tb.steps[prev_idx];
                        
                        // Check if previous step writes to a destination buffer
                        // Types that write data: 'r' (recv), 'rcs' (recv-copy-send), 'cpy' (copy), 'rre' (reduce-recv)
                        bool is_producer = (prev_step.type == "r" || prev_step.type == "rcs" || 
                                            prev_step.type == "cpy" || prev_step.type == "rre");

                        if (is_producer && prev_step.dst_buffer == step.src_buffer) {
                             // Check for overlapping offset ranges
                             int s_start = step.src_offset;
                             int s_end = step.src_offset + step.count;
                             int p_start = prev_step.dst_offset;
                             int p_end = prev_step.dst_offset + prev_step.count;

                             // Simple interval overlap check: max(start1, start2) < min(end1, end2)
                             if (std::max(s_start, p_start) < std::min(s_end, p_end)) {
                                 has_implicit_dep = true;
                                 NcclLog->writeLog(NcclLogLevel::DEBUG, 
                                     "Detected implicit intra-TB dependency: GPU %d TB %d Step %d (src %s off %d) depends on Step %d (dst %s off %d)",
                                     gpu_id, tb.id, step_idx, step.src_buffer.c_str(), s_start, prev_idx, prev_step.dst_buffer.c_str(), p_start);
                                 break; 
                             }
                        }
                    }
                }

                // A step is a root source ONLY if it has neither explicit nor implicit dependencies
                bool is_true_source = has_no_explicit_dep && !has_implicit_dep;

                bool depends_on_copy = false;
                if (!has_no_explicit_dep && step.dep_id < gpu.thread_blocks.size()) {
                    const auto& dep_tb = gpu.thread_blocks[step.dep_id];
                    if (step.dep_step < dep_tb.steps.size()) {
                        depends_on_copy = (dep_tb.steps[step.dep_step].type == "cpy");
                    }
                }

                // If it's a SEND step acting as a source (either fresh data or after a local copy from fresh data)
                if ((step.type == "s") && (is_true_source || depends_on_copy)) {
                    for(int k = 0; k < step.count; ++k) {

                        // Calculate the absolute global offset for this specific unit of data
                        int global_src_offset = step.src_offset + k;

                        // Create a unique key for the data source
                        auto source_key = std::make_tuple(gpu_id, step.src_buffer, global_src_offset);

                        int current_chunk_id;

                        // Check if we have already assigned an ID to this specific source data
                        if (source_data_to_chunk_id.find(source_key) != source_data_to_chunk_id.end()) {
                            // Reuse existing ID (e.g., multicast scenario: 0->1 and 0->8)
                            current_chunk_id = source_data_to_chunk_id[source_key];
                        } else {
                            // New data source: Assign new ID
                            current_chunk_id = unique_logical_chunk_count++;
                            source_data_to_chunk_id[source_key] = current_chunk_id;
                        }

                        // Trace the path for this specific transmission branch
                        std::list<std::tuple<int, int, int, int>> path;
                        NcclLog->writeLog(NcclLogLevel::DEBUG,
                            "Tracing Logical Chunk ID %d (Source: GPU %d, Buf %s, Off %d) via TB %d Step %d",
                            current_chunk_id, gpu_id, step.src_buffer.c_str(), global_src_offset, tb.id, step.id);

                        trace_chunk_recursive(msccl_algorithm, gpu_id, tb.id, step.id, k, path);

                        // Append this path to the collective paths for this logical chunk.
                        // If this ID was reused, we are adding a new branch to the existing chunk's history.
                        logical_chunk_paths[current_chunk_id].splice(logical_chunk_paths[current_chunk_id].end(), path);
                    }
                }
            }
        }
    }

    NcclLog->writeLog(NcclLogLevel::INFO, "Identified %lu unique logical chunks in the algorithm",
                      unique_logical_chunk_count);

    // =================================================================================
    // 3. PASS 1: CREATE FLOWS (Inline PXN Generation)
    // =================================================================================

    // Global lookup: <PipelineStage, LogicalChunkID, GPU, TB, Step> -> FinalFlowID
    // Stores the "Output" Flow ID (Remote if PXN, Standard if not) for downstream consumption.
    std::map<std::tuple<int, int, int, int, int>, uint64_t> pipeline_flow_map;
    
    // Map: RemoteFlowID -> LocalFlowID (Used to resolve entry point for PXN children)
    std::map<uint64_t, uint64_t> pxn_entry_map;

    
    // Sequential execution inside a thread block (MSCCL interpreter order): in every pipeline
    // iteration p a TB runs its steps one after the other, and iteration p+1 starts after the
    // last step of iteration p. So each send waits for the previous step of its TB: the TB's
    // previous send has completed and every receive issued since then has arrived. For a TB
    // with a single send this is exactly "same step, previous iteration".
    // Conservative: on hardware the GPU can move on once the data is in the FIFO buffer.
    // (gpu, tb) -> events the TB's next send waits for, as (flow id, channel)
    std::map<std::pair<int, int>, std::vector<std::pair<uint64_t, int>>> tb_frontier;
    // PXN relays (proxy -> dest) of a TB's sends: the proxy forwards the steps of one connection
    // in order (FIFO), so each relay waits for the previous relay of the same TB to be posted.
    // (gpu, tb) -> (last relay flow id, channel)
    std::map<std::pair<int, int>, std::pair<uint64_t, int>> pxn_relay_frontier;
    size_t tb_order_links = 0;
    auto link_flows = [&](uint64_t parent_fid, int parent_chan, uint64_t child_fid, int child_chan) {
        if (parent_fid == child_fid) return;
        auto child_it = result.find(std::make_pair(child_chan, child_fid));
        auto parent_it = result.find(std::make_pair(parent_chan, parent_fid));
        if (child_it == result.end() || parent_it == result.end()) return;
        auto& parents = child_it->second.parent_flow_id;
        if (std::find(parents.begin(), parents.end(), static_cast<int>(parent_fid)) != parents.end()) return;
        parents.push_back(static_cast<int>(parent_fid));
        child_it->second.prev.push_back(parent_it->second.src);
        parent_it->second.child_flow_id.push_back(static_cast<int>(child_fid));
        ++tb_order_links;
    };

    for (int p = 0; p < num_pipeline_chunks; ++p) {
        uint64_t current_stage_bytes = pipeline_chunk_size;
        if (p == num_pipeline_chunks - 1) {
            current_stage_bytes = total_data_per_algo_loop - (p * pipeline_chunk_size);
        }
        uint64_t flow_size = current_stage_bytes;

        for (auto const& [chunk_id, path] : logical_chunk_paths) {
            for (auto const& [curr_gpu, curr_tb, curr_step_idx, curr_k] : path) {
                const auto& msccl_tb = msccl_algorithm.gpus.at(curr_gpu).thread_blocks[curr_tb];
                int dest_gpu = msccl_tb.send_peer;
                int effective_chunk_id = chunk_id + (p * unique_logical_chunk_count);
                if(dest_gpu == -1) continue;

                int node_src = curr_gpu / nlocalRanks;
                int node_dest = dest_gpu / nlocalRanks;
                int proxy_rank = -1;

                // Determine if PXN transformation is needed
                bool apply_pxn = false;
                if (PXN_ENABLE && gp_info.nNodes > 1 && nlocalRanks > 0 && node_src != node_dest) {
                    proxy_rank = (node_src * nlocalRanks) + (dest_gpu % nlocalRanks);
                    if (proxy_rank != curr_gpu) {
                        apply_pxn = true;
                    }
                }

                if (apply_pxn) {
                    // --- CASE A: PXN Split (Source -> Proxy -> Dest) ---
                    
                    // 1. Local Flow (Source -> Proxy) - The "Entry" Flow
                    uint64_t local_fid = g_flow_id++;
                    SingleFlow local_flow(local_fid, curr_gpu, proxy_rank, flow_size,
                                          {}, {}, {}, msccl_tb.channel, effective_chunk_id,
                                          unique_logical_chunk_count * num_pipeline_chunks, "PXN");
                    
                    // 2. Remote Flow (Proxy -> Dest) - The "Exit" Flow
                    uint64_t remote_fid = g_flow_id++;
                    SingleFlow remote_flow(remote_fid, proxy_rank, dest_gpu, flow_size,
                                           {curr_gpu}, { (int)local_fid }, {}, msccl_tb.channel, effective_chunk_id,
                                           unique_logical_chunk_count * num_pipeline_chunks, "RING");

                    // Internal Link: Local -> Remote
                    local_flow.child_flow_id.push_back(remote_fid);

                    // Store both
                    result[std::make_pair(msccl_tb.channel, local_fid)] = local_flow;
                    result[std::make_pair(msccl_tb.channel, remote_fid)] = remote_flow;

                    // Map Logical Step to Remote ID (because that's the output)
                    pipeline_flow_map[std::make_tuple(p, chunk_id, curr_gpu, curr_tb, curr_step_idx)] = remote_fid;

                    // Record entry mapping for parents to link to
                    pxn_entry_map[remote_fid] = local_fid;

                } else {
                    // --- CASE B: Standard Flow (Source -> Dest) ---
                    uint64_t f_id = g_flow_id++;
                    SingleFlow flow(f_id, curr_gpu, dest_gpu, flow_size,
                                    {}, {}, {}, msccl_tb.channel, effective_chunk_id,
                                    unique_logical_chunk_count * num_pipeline_chunks, "RING");

                    result[std::make_pair(msccl_tb.channel, f_id)] = flow;
                    pipeline_flow_map[std::make_tuple(p, chunk_id, curr_gpu, curr_tb, curr_step_idx)] = f_id;
                }
            }
        }

        // Each send of iteration p depends on the previous step of its thread block.
        // Linked once every flow of the iteration exists: flows are created chunk by chunk,
        // not in TB step order, so the previous step may not have existed yet above.
        std::map<std::tuple<int, int, int>, uint64_t> stage_send;                      // (gpu, tb, step) -> output flow
        std::map<std::tuple<int, int, int>, std::pair<uint64_t, int>> stage_arrival;   // (dest, src, offset) -> (output flow, channel)
        for (auto it = pipeline_flow_map.lower_bound(std::make_tuple(p, INT_MIN, INT_MIN, INT_MIN, INT_MIN));
             it != pipeline_flow_map.end() && std::get<0>(it->first) == p; ++it) {
            int g = std::get<2>(it->first), t = std::get<3>(it->first), st = std::get<4>(it->first);
            const auto& tb_obj = msccl_algorithm.gpus.at(g).thread_blocks[t];
            stage_send[std::make_tuple(g, t, st)] = it->second;
            stage_arrival[std::make_tuple(tb_obj.send_peer, g, tb_obj.steps[st].dst_offset)] =
                std::make_pair(it->second, tb_obj.channel);
        }
        for (auto const& [g, gpu] : msccl_algorithm.gpus) {
            for (auto const& tb : gpu.thread_blocks) {
                auto& frontier = tb_frontier[std::make_pair(g, tb.id)];
                for (int st = 0; st < static_cast<int>(tb.steps.size()); ++st) {
                    const auto& step = tb.steps[st];
                    const bool recvs = (step.type == "r" || step.type == "rcs" || step.type == "rrc" ||
                                        step.type == "rrcs" || step.type == "rrs");
                    const bool sends = (step.type == "s" || step.type == "rcs" ||
                                        step.type == "rrcs" || step.type == "rrs");
                    if (recvs && tb.recv_peer >= 0) {
                        auto a = stage_arrival.find(std::make_tuple(g, tb.recv_peer, step.src_offset));
                        if (a != stage_arrival.end()) frontier.push_back(a->second);
                    }
                    if (sends) {
                        auto f = stage_send.find(std::make_tuple(g, tb.id, st));
                        if (f == stage_send.end()) continue;
                        // With PXN the TB's own send is observed through its first hop (GPU -> proxy).
                        uint64_t entry_fid = pxn_entry_map.count(f->second) ? pxn_entry_map.at(f->second) : f->second;
                        for (auto const& [parent_fid, parent_chan] : frontier) {
                            link_flows(parent_fid, parent_chan, entry_fid, tb.channel);
                        }
                        frontier.assign(1, std::make_pair(entry_fid, tb.channel));
                        if (pxn_entry_map.count(f->second)) {
                            // f->second is the relay (proxy -> dest) of this PXN send
                            auto relay = pxn_relay_frontier.find(std::make_pair(g, tb.id));
                            if (relay != pxn_relay_frontier.end()) {
                                link_flows(relay->second.first, relay->second.second, f->second, tb.channel);
                            }
                            pxn_relay_frontier[std::make_pair(g, tb.id)] = std::make_pair(f->second, tb.channel);
                        }
                    }
                }
            }
        }
    }
    NcclLog->writeLog(NcclLogLevel::INFO, "Added %zu intra-TB ordering dependencies.", tb_order_links);

    NcclLog->writeLog(NcclLogLevel::INFO, "Created %lu instantiated flows (including PXN splits)", result.size());

    // =================================================================================
    // 4. PASS 2: LINK DATA DEPENDENCIES
    // =================================================================================
    
    // We iterate over the map to find the "Child" step.
    // child_output_fid is the flow that completes this step (Remote if PXN, Standard otherwise).
    for(auto const& [key, child_output_fid] : pipeline_flow_map) {
        int p = std::get<0>(key);
        int chunk_id = std::get<1>(key);
        int curr_gpu = std::get<2>(key);
        int curr_tb = std::get<3>(key);
        int curr_step_idx = std::get<4>(key);

        // Determine the "Entry" flow for this child step.
        // If PXN, parents must link to the Local flow. If Standard, they link to the flow itself.
        uint64_t child_input_fid = child_output_fid;
        if(pxn_entry_map.count(child_output_fid)) {
            child_input_fid = pxn_entry_map[child_output_fid];
        }

        const auto& path = logical_chunk_paths[chunk_id];
        auto it = path.begin();
        while(it != path.end()) {
            if (std::get<0>(*it) == curr_gpu && std::get<2>(*it) == curr_step_idx && std::get<1>(*it) == curr_tb) break;
            it++;
        }

        if (it != path.end() && it != path.begin()) {
             auto current_node = *it;
             int c_tb_id = std::get<1>(current_node);
             int c_k_offset = std::get<3>(current_node);
             const auto& c_step_obj = msccl_algorithm.gpus.at(curr_gpu).thread_blocks[c_tb_id].steps[curr_step_idx];

             auto prev_it = it;
             while (prev_it != path.begin()) {
                 prev_it--;
                 auto parent_node = *prev_it;
                 int p_gpu = std::get<0>(parent_node);
                 int p_tb = std::get<1>(parent_node);
                 int p_step = std::get<2>(parent_node);
                 int p_k = std::get<3>(parent_node);

                 const auto& p_tb_obj = msccl_algorithm.gpus.at(p_gpu).thread_blocks[p_tb];
                 if (p_tb_obj.send_peer == curr_gpu) {
                      const auto& p_step_obj = p_tb_obj.steps[p_step];
                      if (p_step_obj.dst_offset + p_k == c_step_obj.src_offset + c_k_offset) {
                          
                          auto parent_lookup_key = std::make_tuple(p, chunk_id, p_gpu, p_tb, p_step);
                          
                          if(pipeline_flow_map.count(parent_lookup_key)) {
                              // Parent Output ID: If PXN, this is the Remote Flow. If Standard, it's the Flow itself.
                              // This is exactly what we want: dependencies come FROM the end of the previous step.
                              uint64_t parent_output_fid = pipeline_flow_map[parent_lookup_key];

                              int c_chan = msccl_algorithm.gpus.at(curr_gpu).thread_blocks[c_tb_id].channel;
                              int p_chan = p_tb_obj.channel;

                              // Safely link Parent(Output) -> Child(Input)
                              // (skipped if the intra-TB ordering already added this parent)
                              if (result.count({c_chan, child_input_fid}) && result.count({p_chan, parent_output_fid})) {
                                  auto& c_parents = result.at({c_chan, child_input_fid}).parent_flow_id;
                                  if (std::find(c_parents.begin(), c_parents.end(),
                                                static_cast<int>(parent_output_fid)) == c_parents.end()) {
                                      result.at({c_chan, child_input_fid}).parent_flow_id.push_back(parent_output_fid);
                                      result.at({c_chan, child_input_fid}).prev.push_back(result.at({p_chan, parent_output_fid}).src);
                                      result.at({p_chan, parent_output_fid}).child_flow_id.push_back(child_input_fid);
                                  }
                              }
                          }
                          break; 
                      }
                 }
             }
        }
    }

    NcclLog->writeLog(NcclLogLevel::INFO, "Linked data dependencies using correct PXN Input/Output logic.");

    // =================================================================================
    // 5. Cleanup & Return
    // =================================================================================

    for(const auto& [key, flow] : result) {
        rank2flowmodels[flow.src][key] = flow;
        if (flow.src != flow.dest) {
            rank2flowmodels[flow.dest][key] = flow;
        }
    }

    if (!validate_flow_graph(result, msccl_algorithm, data_size, unique_logical_chunk_count * num_pipeline_chunks, NcclLog, info->protocol)) {
        NcclLog->writeLog(NcclLogLevel::ERROR, "Flow graph validation failed. Aborting flow model generation.");
        return std::nullopt;
    }

    for(const auto& [curr_rank, models] : rank2flowmodels) {
        rank2pflowmodels[curr_rank] = std::make_shared<FlowModels>(models);
    }

    NcclLog->writeLog(NcclLogLevel::INFO, "--- Flow generation complete for rank %d ---", rank);
    logRankToFlowModels(NcclLogLevel::INFO, "Custom MSCCL Pipelined (Inline PXN)", rank2pflowmodels);
    return rank2pflowmodels;
  }

  // collectives + each single specialized algorithm generator

  std::map<int,std::shared_ptr<FlowModels>> MockNcclGroup::genAlltoAllFlowModels(GroupType type, int rank, uint64_t data_size,bool& msccl){
      ncclInfo* ncc_info = get_algo_proto_info(type, rank, AstraSim::ComType::All_to_All, data_size,msccl);
      MockNcclLog* NcclLog = MockNcclLog::getInstance();

      switch (ncc_info->algorithm) {
          case NCCL_ALGO_TREE:
          case NCCL_ALGO_RING:
              {
                NcclLog->writeLog(NcclLogLevel::DEBUG, "Using genAllToAllRingFlowModels for type %d, rank %d, data_size %lu", type, rank, data_size);
                return genAllToAllRingFlowModels(type, rank, data_size);
              }

          case NCCL_ALGO_NVLS:
              // It's generally better to throw an exception or handle the error gracefully than calling exit(1)
              // For example: throw std::runtime_error("NCCL_ALGO_NVLS is not supported");
              exit(1);

          case NCCL_ALGO_NVLS_TREE:
              return {};

          case NCCL_ALGO_MSCCL:
          {
              NcclLog->writeLog(NcclLogLevel::INFO, "Using genCustomFlowModels for operator %s type %s, rank %d, data_size %lu, protocol %d", to_cstr(ncc_info->coll), to_cstr(type), rank, data_size, ncc_info->protocol);
              auto retval = genCustomFlowModels(type, rank, data_size, ncc_info);
              if (!retval) { // A more idiomatic check for std::optional
                  NcclLog->writeLog(NcclLogLevel::INFO, "genCustomFlowModels failed, falling back to NCCL algorithm selection for operator %s type %s, rank %d, data_size %lu, protocol %d", to_cstr(ncc_info->coll), to_cstr(type), rank, data_size, ncc_info->protocol);
                  // Re-run the full NCCL selection without MSCCL (cannot recurse again:
                  // with msccl == false the MSCCL case is never selected).
                  msccl = false;
                  return genAlltoAllFlowModels(type, rank, data_size, msccl);
              } else {
                  return retval.value();
              }
          }

          default:
              NcclLog->writeLog(NcclLogLevel::ERROR, "Unsupported algorithm %d for AllToAll in type %d, rank %d, data_size %lu", ncc_info->algorithm, type, rank, data_size);
              return {};
      }
      return {};
  }

  std::map<int,std::shared_ptr<FlowModels>> MockNcclGroup::genAllToAllRingFlowModels(GroupType type , int rank,uint64_t data_size) {
    FlowModels result = {};
    std::map<int,FlowModels>rank2flowmodels;
    std::map<int,std::shared_ptr<FlowModels>>rank2pflowmodels;
    SingleFlow tmp_result;
    uint64_t chunksize;
    uint64_t send_size;
    int nranks;
    int chunkcount;
    int chunkid;
    GroupInfo gp_info;
    int gp_idx;
    RingChannels ringchannels;
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    if(GroupIndex.count(std::make_pair(rank,type))==0){
      NcclLog->writeLog(NcclLogLevel::ERROR,"There is no corresponding group info and group ring channel, resulting in an error in generating the flow model.");
      return {};
    } else {
      gp_idx = GroupIndex[std::make_pair(rank,type)];
      ringchannels = Allringchannels[gp_idx];
      gp_info = AllGroups[gp_idx];
    }
    nranks = gp_info.nRanks;
    size_t ranks_per_node = gp_info.nRanks / gp_info.nNodes;
    chunkcount = nranks - 1;
    chunksize = data_size / nranks;
    data_size = data_size / nranks;
    bool PXN_ENABLE = false;
    const char* PXN_ENV = std::getenv("AS_PXN_ENABLE");
    if (PXN_ENV && strcmp(PXN_ENV, "1") == 0) {
      PXN_ENABLE = true;
    } else {
      PXN_ENABLE = false;
    }
    for (int i = 0; i < gp_info.Ranks.size(); i++) {
      for (int j = 0; j < gp_info.Ranks.size(); j++) {
        if (i == j) {
          continue;
        }
        std::vector<int> prev;
        int src = gp_info.Ranks[i];
        vector<int> parent_flows = {};
        string connection_type = "PTP";
        uint32_t channel_id = 0;
        if (PXN_ENABLE
            && gp_info.Ranks[i] / ranks_per_node != gp_info.Ranks[j] / ranks_per_node
            && gp_info.Ranks[i] % ranks_per_node != gp_info.Ranks[j] % ranks_per_node) {
          int pxn_rank = gp_info.Ranks[j] % ranks_per_node + ranks_per_node * (gp_info.Ranks[i] / ranks_per_node);
          channel_id = (AstraSim::pos_mod(
              gp_info.Ranks[j] / ranks_per_node - gp_info.Ranks[i] / ranks_per_node,
              gp_info.nNodes) - 1) * ranks_per_node;
          channel_id += AstraSim::pos_mod(
              gp_info.Ranks[j] % ranks_per_node - gp_info.Ranks[i] % ranks_per_node,
              ranks_per_node);
          tmp_result = SingleFlow(
              g_flow_id,
              gp_info.Ranks[i],
              pxn_rank,
              chunksize,
              {},
              {},
              {g_flow_id + 1},
              channel_id,
              0,
              1,
              "PTP_PXN_START");
          result[std::make_pair(channel_id, g_flow_id)] = tmp_result;
          src = pxn_rank;
          parent_flows.push_back(g_flow_id);
          connection_type = "PTP_PXN_END";
          prev = {gp_info.Ranks[i]};
          g_flow_id++;
        }
        tmp_result = SingleFlow(
            g_flow_id,
            src,
            gp_info.Ranks[j],
            chunksize,
            prev,
            parent_flows,
            {},
            channel_id,
            0,
            1,
            connection_type);
        result[std::make_pair(channel_id, g_flow_id)] = tmp_result;
        g_flow_id++;
      }
    }
    for(auto flow_models_it = result.begin();flow_models_it!=result.end();flow_models_it++){
      int src = flow_models_it->second.src;
      int dst = flow_models_it->second.dest;
      rank2flowmodels[src][std::make_pair(flow_models_it->first.first,flow_models_it->first.second)]=flow_models_it->second;
      rank2flowmodels[dst][std::make_pair(flow_models_it->first.first,flow_models_it->first.second)]=flow_models_it->second;
    }
    for(auto it = rank2flowmodels.begin();it!=rank2flowmodels.end();it++){
      rank2pflowmodels[it->first] = std::make_shared<FlowModels>(it->second);
    }
    logRankToFlowModels(NcclLogLevel::INFO, "RING AllToAll", rank2pflowmodels);
    return rank2pflowmodels;
  }

  std::map<int, std::shared_ptr<FlowModels>> MockNcclGroup::genReduceScatterFlowModels(
      GroupType type,
      int rank,
      uint64_t data_size,
      bool& msccl) {
    ncclInfo* ncc_info =
        get_algo_proto_info(type, rank, AstraSim::ComType::Reduce_Scatter, data_size, msccl);
    MockNcclLog* NcclLog = MockNcclLog::getInstance();

    switch (ncc_info->algorithm) {
    case NCCL_ALGO_TREE:
    case NCCL_ALGO_RING:
      {
        NcclLog->writeLog(
            NcclLogLevel::DEBUG,
            "Using genReduceScatterRingFlowModels for type %d, rank %d, data_size %lu",
            type,
            rank,
            data_size);
        return genReduceScatterRingFlowModels(type, rank, data_size);
      }

    case NCCL_ALGO_NVLS:
      // It's generally better to throw an exception or handle the error gracefully than calling exit(1)
      // For example: throw std::runtime_error("NCCL_ALGO_NVLS is not supported");
      exit(1);

    case NCCL_ALGO_NVLS_TREE:
      return {};

    case NCCL_ALGO_MSCCL:
      {
        NcclLog->writeLog(
            NcclLogLevel::INFO,
            "Using genCustomFlowModels for operator %s type %s, rank %d, data_size %lu, protocol %d",
            to_cstr(ncc_info->coll),
            to_cstr(type),
            rank,
            data_size,
            ncc_info->protocol);
        auto retval = genCustomFlowModels(type, rank, data_size, ncc_info);
        if (!retval) { // A more idiomatic check for std::optional
          NcclLog->writeLog(
              NcclLogLevel::INFO,
              "genCustomFlowModels failed, falling back to NCCL algorithm selection for operator %s type %s, rank "
              "%d, data_size %lu, protocol %d",
              to_cstr(ncc_info->coll),
              to_cstr(type),
              rank,
              data_size,
              ncc_info->protocol);
          // Re-run the full NCCL selection without MSCCL (cannot recurse again:
          // with msccl == false the MSCCL case is never selected).
          msccl = false;
          return genReduceScatterFlowModels(type, rank, data_size, msccl);
        } else {
          return retval.value();
        }
      }

    default:
      NcclLog->writeLog(
          NcclLogLevel::ERROR,
          "Unsupported algorithm %d for ReduceScatter in type %d, rank %d, data_size %lu",
          ncc_info->algorithm,
          type,
          rank,
          data_size);
      return {};
    }
    return {};
  }

  std::map<int,std::shared_ptr<FlowModels>> MockNcclGroup::genReduceScatterRingFlowModels(
      GroupType type,
      int rank,
      uint64_t data_size) {
    FlowModels result = {};
    std::map<int,FlowModels>rank2flowmodels;
    std::map<int,std::shared_ptr<FlowModels>>rank2pflowmodels;
    std::map<int, SingleFlow> task_list = {}; 
    std::map<int, SingleFlow> task_list2 = {};
    SingleFlow tmp_result;
    uint64_t chunksize;
    uint64_t send_size;
    int nranks;
    int chunkcount;
    int chunkid;
    GroupInfo gp_info;
    int gp_idx;
    RingChannels ringchannels;
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    if(GroupIndex.count(std::make_pair(rank,type))==0){
      NcclLog->writeLog(NcclLogLevel::ERROR,"There is no corresponding group info and group ring channel, resulting in an error in generating the flow model.");
      return {};
    } else {
      gp_idx = GroupIndex[std::make_pair(rank,type)];
      ringchannels = Allringchannels[gp_idx];
      gp_info = AllGroups[gp_idx];
    }
    bool PXN_ENABLE = false;
    const char* PXN_ENV = std::getenv("AS_PXN_ENABLE");
    if (PXN_ENV && strcmp(PXN_ENV, "1") == 0) {
      PXN_ENABLE = true;
    } else {
      PXN_ENABLE = false;
    }
    nranks = gp_info.nRanks;
    chunkcount = nranks - 1;
    chunksize = data_size / nranks / ringchannels.size();
    data_size = data_size / nranks / ringchannels.size();
    for (auto it = ringchannels.begin(); it != ringchannels.end(); it++) {
      auto ring = it->second;
      auto ring_id = it->first;
      task_list = {};
      send_size = 0;
      chunkid = 0;
      while (send_size < data_size) {
        uint64_t real_chunksize = std::min(chunksize, data_size - send_size);
        int prenoderecvrank = ring.rbegin()->second[2];
        int prenodesendrank = ring.rbegin()->second[3];
        int curnoderecvrank = ring.begin()->second[2];
        int curnodesendrank = ring.begin()->second[3];
        std::vector<int> prevranks = {};
        for (auto rank_it = ring.begin(); rank_it != ring.end(); rank_it++) {
          int cur_rank = rank_it->first;
          if (curnoderecvrank != rank_it->second[2] &&
              curnodesendrank != rank_it->second[3]) {
            prenoderecvrank = curnoderecvrank;
            prenodesendrank = curnodesendrank;
            curnoderecvrank = rank_it->second[2];
            curnodesendrank = rank_it->second[3];
          }
          if (rank_it->second[3] == cur_rank &&
              rank_it->second[2] != cur_rank && gp_info.nNodes > 1 &&
              PXN_ENABLE) { 
            prevranks.clear();
            if (rank_it->second[0] != -1)
              prevranks = {rank_it->second[0]};
            tmp_result = SingleFlow(
                g_flow_id,
                rank_it->first,
                rank_it->second[2],
                data_size,
                prevranks,
                {},
                {g_flow_id + 1},
                ring_id,
                chunkid,
                chunkcount,
                "RING");
            result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
            g_flow_id++;
            if (rank_it->first != -1) {
              prevranks = {rank_it->first};
            } else {
              prevranks = {};
            }
            tmp_result = SingleFlow(
                g_flow_id,
                rank_it->second[2],
                rank_it->second[1],
                data_size,
                prevranks,
                {g_flow_id - 1},
                {},
                ring_id,
                chunkid,
                chunkcount,
                "PXN_INIT");
            result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
            task_list[rank_it->first] = tmp_result;
            g_flow_id++;
          } else if (
              rank_it->second[2] == cur_rank &&
              rank_it->second[3] != cur_rank && gp_info.nNodes > 1 &&
              PXN_ENABLE) {
            prevranks.clear();
            if(prenoderecvrank!=-1){
              prevranks = {prenoderecvrank};
            }
            tmp_result = SingleFlow(
                g_flow_id,
                rank_it->first,
                rank_it->second[1],
                data_size,
                prevranks,
                {},
                {},
                ring_id,
                chunkid,
                chunkcount,
                "RING");
            result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
            task_list[rank_it->first] = tmp_result;
            g_flow_id++;
          } else { 
            prevranks.clear();
            if(rank_it->second[0]!=-1){
              prevranks = {rank_it->second[0]};
            }
            tmp_result = SingleFlow(
                g_flow_id,
                rank_it->first,
                rank_it->second[1],
                data_size,
                prevranks,
                {},
                {},
                ring_id,
                chunkid,
                chunkcount,
                "RING");
            result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
            task_list[rank_it->first] = tmp_result;
            g_flow_id++;
          }
        }
        chunkid++;
        for (int i = 0; i < nranks - 2; i++) {
          task_list2 = {};
          prenoderecvrank = ring.rbegin()->second[2];
          prenodesendrank = ring.rbegin()->second[3];
          curnoderecvrank = ring.begin()->second[2];
          curnodesendrank = ring.begin()->second[3];
          for (auto rank_it = ring.begin(); rank_it != ring.end(); rank_it++) {
            if (curnoderecvrank != rank_it->second[2] &&
              curnodesendrank != rank_it->second[3]) {
            prenoderecvrank = curnoderecvrank;
            prenodesendrank = curnodesendrank;
            curnoderecvrank = rank_it->second[2];
            curnodesendrank = rank_it->second[3];
          }
            int cur_rank = rank_it->first;
            int partner_flow_id = task_list[rank_it->second[0]].flow_id;
            if (rank_it->second[3] == cur_rank &&
                rank_it->second[2] != cur_rank && gp_info.nNodes > 1 &&
                PXN_ENABLE) { 
              prevranks.clear();
              if (rank_it->second[0] != -1) {
                prevranks = {rank_it->second[0]};
              }
              tmp_result = SingleFlow(
                  g_flow_id,
                  rank_it->first,
                  rank_it->second[2],
                  data_size,
                  prevranks,
                  {partner_flow_id},
                  {g_flow_id + 1},
                  ring_id,
                  chunkid,
                  chunkcount,
                  "RING");
              result[std::make_pair(ring_id, partner_flow_id)].child_flow_id.push_back(g_flow_id);
              result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
              g_flow_id++;
              if(rank_it->first!=-1){
                prevranks={rank_it->first};
              }else{
                prevranks ={};
              }
              tmp_result = SingleFlow(
                  g_flow_id,
                  rank_it->second[2],
                  rank_it->second[1],
                  data_size,
                  prevranks,
                  {g_flow_id - 1},
                  {},
                  ring_id,
                  chunkid,
                  chunkcount,
                  "PXN");
              task_list2[rank_it->first] = tmp_result;
              result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
              g_flow_id++;
            } else if (
                rank_it->second[2] == cur_rank &&
                rank_it->second[3] != cur_rank && gp_info.nNodes > 1 &&
                PXN_ENABLE) {
              prevranks.clear();
              if (prenoderecvrank != -1) {
                prevranks = {prenoderecvrank};
              }
              tmp_result = SingleFlow(
                  g_flow_id,
                  rank_it->first,
                  rank_it->second[1],
                  data_size,
                  prevranks,
                  {partner_flow_id},
                  {},
                  ring_id,
                  chunkid,
                  chunkcount,
                  "RING");
              result[std::make_pair(ring_id, partner_flow_id)].child_flow_id .push_back(g_flow_id);
              task_list2[rank_it->first] = tmp_result;
              result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
              g_flow_id++;
            } else { 
              prevranks.clear();
              if(rank_it->second[0]!=-1){
                prevranks= {rank_it->second[0]};
              }
              tmp_result = SingleFlow(
                  g_flow_id,
                  rank_it->first,
                  rank_it->second[1],
                  data_size,
                  prevranks,
                  {partner_flow_id},
                  {},
                  ring_id,
                  chunkid,
                  chunkcount,
                  "RING");
              result[std::make_pair(ring_id, partner_flow_id)].child_flow_id.push_back(g_flow_id);
              task_list2[rank_it->first] = tmp_result;
              result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
              g_flow_id++;
            }
          }
          task_list = task_list2;
          chunkid++;
        }
        send_size += real_chunksize;
      }
    }
    for(auto flow_models_it = result.begin();flow_models_it!=result.end();flow_models_it++){
      int src = flow_models_it->second.src;
      int dst = flow_models_it->second.dest;
      rank2flowmodels[src][std::make_pair(flow_models_it->first.first,flow_models_it->first.second)]=flow_models_it->second;
      rank2flowmodels[dst][std::make_pair(flow_models_it->first.first,flow_models_it->first.second)]=flow_models_it->second;
    }
    for(auto it = rank2flowmodels.begin();it!=rank2flowmodels.end();it++){
      rank2pflowmodels[it->first] = std::make_shared<FlowModels>(it->second);
    }
    logRankToFlowModels(NcclLogLevel::INFO, "RING ReduceScatter", rank2pflowmodels);
    return rank2pflowmodels;
  }

  std::map<int,std::shared_ptr<FlowModels>> MockNcclGroup::genAllReduceFlowModels(GroupType type , int rank,uint64_t data_size,bool& msccl){
    // AllReduce never uses MSCCL: clear the flag so the result is cached under the
    // non-MSCCL name (otherwise the re-use print would report a stale schedule file).
    msccl = false;
    ncclInfo* ncc_info = get_algo_proto_info(type,rank,AstraSim::ComType::All_Reduce,data_size, msccl);
    switch (ncc_info->algorithm) {
      case NCCL_ALGO_TREE:
      case NCCL_ALGO_RING:
        return genAllReduceRingFlowModels(type, rank, data_size);
      case NCCL_ALGO_NVLS:
        return genAllreduceNVLSFlowModels(type,rank,data_size);
      case NCCL_ALGO_NVLS_TREE:
        return {};
      default:
        return {};
    }
  }

  std::map<int,std::shared_ptr<FlowModels>> MockNcclGroup::genAllReduceRingFlowModels(GroupType type , int rank,uint64_t data_size){
    FlowModels result = {};
    std::map<int,FlowModels>rank2flowmodels;
    std::map<int,std::shared_ptr<FlowModels>>rank2pflowmodels;
    std::map<int,SingleFlow> task_list = {}; 
    std::map<int,SingleFlow> task_list2 = {};
    SingleFlow tmp_result;
    uint64_t chunksize;
    uint64_t send_size;
    int nranks;
    GroupInfo gp_info;
    int gp_idx;
    RingChannels ringchannels;
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    if(GroupIndex.count(std::make_pair(rank,type))==0){
      NcclLog->writeLog(NcclLogLevel::ERROR,"There is no corresponding group info and group ring channel, resulting in an error in generating the flow model.");
      return {};
    } else {
      gp_idx = GroupIndex[std::make_pair(rank,type)];
      ringchannels = Allringchannels[gp_idx];
      gp_info = AllGroups[gp_idx];
    }
    nranks = gp_info.nRanks;
    bool PXN_ENABLE = false;
    const char* PXN_ENV = std::getenv("AS_PXN_ENABLE");
    if (PXN_ENV && strcmp(PXN_ENV, "1") == 0) {
      PXN_ENABLE = true;
    } else {
      PXN_ENABLE = false;
    }
    chunksize = data_size / nranks / ringchannels.size();
    data_size = data_size / nranks / ringchannels.size();
    int chunkcout = 2*(gp_info.nRanks-1);

    for(auto it = ringchannels.begin(); it !=ringchannels.end(); it++) {
      auto ring = it->second;
      auto ring_id = it->first;
      task_list = {};
      send_size = 0;
      int chunk_id = 0;
      while (send_size < data_size)
      {
        uint64_t real_chunksize = std::min(chunksize, data_size - send_size);
        int prenoderecvrank = ring.rbegin()->second[2];
        int prenodesendrank = ring.rbegin()->second[3];
        int curnoderecvrank = ring.begin()->second[2];
        int curnodesendrank = ring.begin()->second[3];
        std::vector<int> prevranks = {};
        for (auto rank_it = ring.begin(); rank_it != ring.end(); rank_it++) {
          int cur_rank = rank_it->first;
          if (curnoderecvrank != rank_it->second[2] &&
              curnodesendrank != rank_it->second[3]) {
            prenoderecvrank = curnoderecvrank;
            prenodesendrank = curnodesendrank;
            curnoderecvrank = rank_it->second[2];
            curnodesendrank = rank_it->second[3];
          }
          if (rank_it->second[3] == cur_rank &&
              rank_it->second[2] != cur_rank && gp_info.nNodes > 1 &&
              PXN_ENABLE) { 
            prevranks.clear();
            if(rank_it->second[0]!=-1){
              prevranks={rank_it->second[0]};
            }
            tmp_result = SingleFlow(
                g_flow_id,
                rank_it->first,
                rank_it->second[2],
                data_size,
                prevranks,
                {},
                {g_flow_id + 1},
                ring_id,
                chunk_id,
                chunkcout,
                "RING");
            result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
            g_flow_id++;
            prevranks.clear();
            prevranks = {rank_it->first};          
            tmp_result = SingleFlow(
                g_flow_id,
                rank_it->second[2],
                rank_it->second[1],
                data_size,
                prevranks,
                {g_flow_id - 1},
                {},
                ring_id,
                chunk_id,
                chunkcout,
                "PXN_INIT");
            result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
            task_list[rank_it->first] = tmp_result;
            g_flow_id++;
          } else if (
              rank_it->second[2] == cur_rank &&
              rank_it->second[3] != cur_rank && gp_info.nNodes > 1 &&
              PXN_ENABLE) {
            prevranks.clear();
            if (prenoderecvrank != -1) {
              prevranks = {prenoderecvrank};
            }
            tmp_result = SingleFlow(
                g_flow_id,
                rank_it->first,
                rank_it->second[1],
                data_size,
                prevranks,
                {},
                {},
                ring_id,
                chunk_id,
                chunkcout,
                "RING");
            result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
            task_list[rank_it->first] = tmp_result;
            g_flow_id++;
          } else { 
            prevranks.clear();
            if(rank_it->second[0]!=-1){
              prevranks={rank_it->second[0]};
            }
            tmp_result = SingleFlow(
                g_flow_id,
                rank_it->first,
                rank_it->second[1],
                data_size,
                prevranks,
                {},
                {},
                ring_id,
                chunk_id,
                chunkcout,
                "RING");
            result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
            task_list[rank_it->first] = tmp_result;
            g_flow_id++;
          }
        }
        chunk_id++;
        for(int i =0; i < nranks -1; i++) {
          task_list2 = {};
          prenoderecvrank = ring.rbegin()->second[2];
          prenodesendrank = ring.rbegin()->second[3];
          curnoderecvrank = ring.begin()->second[2];
          curnodesendrank = ring.begin()->second[3];
          for (auto rank_it = ring.begin(); rank_it != ring.end(); rank_it++) {
            if (curnoderecvrank != rank_it->second[2] &&
              curnodesendrank != rank_it->second[3]) {
            prenoderecvrank = curnoderecvrank;
            prenodesendrank = curnodesendrank;
            curnoderecvrank = rank_it->second[2];
            curnodesendrank = rank_it->second[3];
          }
            int cur_rank = rank_it->first;
            int partner_flow_id = task_list[rank_it->second[0]].flow_id;
            if (rank_it->second[3] == cur_rank &&
                rank_it->second[2] != cur_rank && gp_info.nNodes > 1 &&
                PXN_ENABLE) { 
              prevranks.clear();
              if (rank_it->second[0] != -1) {
                prevranks = {rank_it->second[0]};
              }
              tmp_result = SingleFlow(
                  g_flow_id,
                  rank_it->first,
                  rank_it->second[2],
                  data_size,
                  prevranks,
                  {partner_flow_id},
                  {g_flow_id + 1},
                  ring_id,
                  chunk_id,
                  chunkcout,
                  "RING");
              result[std::make_pair(ring_id, partner_flow_id)].child_flow_id.push_back(g_flow_id);
              result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
              g_flow_id++;
              prevranks.clear();
              prevranks={rank_it->first};
              tmp_result = SingleFlow(
                  g_flow_id,
                  rank_it->second[2],
                  rank_it->second[1],
                  data_size,
                  prevranks,
                  {g_flow_id - 1},
                  {},
                  ring_id,
                  chunk_id,
                  chunkcout,
                  "PXN_INIT");
              task_list2[rank_it->first] = tmp_result;
              result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
              g_flow_id++;
            } else if (
                rank_it->second[2] == cur_rank &&
                rank_it->second[3] != cur_rank && gp_info.nNodes > 1 &&
                PXN_ENABLE) {
              prevranks.clear();
              if(prenoderecvrank!=-1){
                prevranks = {prenoderecvrank};
              }
              tmp_result = SingleFlow(
                  g_flow_id,
                  rank_it->first,
                  rank_it->second[1],
                  data_size,
                  prevranks,
                  {partner_flow_id},
                  {},
                  ring_id,
                  chunk_id,
                  chunkcout,
                  "RING");
              result[std::make_pair(ring_id, partner_flow_id)].child_flow_id.push_back(g_flow_id);
              task_list2[rank_it->first] = tmp_result;
              result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
              g_flow_id++;
            } else { 
            prevranks.clear();
            if(rank_it->second[0]!=-1)
            {
              prevranks ={rank_it->second[0]};
            }
              tmp_result = SingleFlow(
                  g_flow_id,
                  rank_it->first,
                  rank_it->second[1],
                  data_size,
                  prevranks,
                  {partner_flow_id},
                  {},
                  ring_id,
                  chunk_id,
                  chunkcout,
                  "RING");
              result[std::make_pair(ring_id, partner_flow_id)].child_flow_id.push_back(g_flow_id);
              task_list2[rank_it->first] = tmp_result;
              result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
              g_flow_id++;
            }
          }
          task_list = task_list2;
          chunk_id++;
        }
        for (int i = 0; i < nranks - 2; i++) {
          task_list2 = {};
          prenoderecvrank = ring.rbegin()->second[2];
          prenodesendrank = ring.rbegin()->second[3];
          curnoderecvrank = ring.begin()->second[2];
          curnodesendrank = ring.begin()->second[3];
          for (auto rank_it = ring.begin(); rank_it != ring.end(); rank_it++) {
            if (curnoderecvrank != rank_it->second[2] &&
              curnodesendrank != rank_it->second[3]) {
            prenoderecvrank = curnoderecvrank;
            prenodesendrank = curnodesendrank;
            curnoderecvrank = rank_it->second[2];
            curnodesendrank = rank_it->second[3];
          }
            int cur_rank = rank_it->first;
            int partner_flow_id = task_list[rank_it->second[0]].flow_id;
            if (rank_it->second[3] == cur_rank &&
                rank_it->second[2] != cur_rank && gp_info.nNodes > 1 &&
                PXN_ENABLE) { 
              prevranks.clear();
              if(rank_it->second[0]!=-1){
                prevranks = {rank_it->second[0]};
              }
              tmp_result = SingleFlow(
                  g_flow_id,
                  rank_it->first,
                  rank_it->second[2],
                  data_size,
                  prevranks,
                  {partner_flow_id},
                  {g_flow_id + 1},
                  ring_id,
                  chunk_id,
                  chunkcout,
                  "RING");
              result[std::make_pair(ring_id, partner_flow_id)].child_flow_id.push_back(g_flow_id);
              result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
              g_flow_id++;
              prevranks.clear();
              if (rank_it->first != -1) {
                prevranks = {rank_it->first};
              }
              tmp_result = SingleFlow(
                  g_flow_id,
                  rank_it->second[2],
                  rank_it->second[1],
                  data_size,
                  prevranks,
                  {g_flow_id - 1},
                  {},
                  ring_id,
                  chunk_id,
                  chunkcout,
                  "PXN_INIT");
              task_list2[rank_it->first] = tmp_result;
              result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
              g_flow_id++;
            } else if (
                rank_it->second[2] == cur_rank &&
                rank_it->second[3] != cur_rank && gp_info.nNodes > 1 &&
                PXN_ENABLE) {
              prevranks.clear();
              if(prenoderecvrank!=-1){
                prevranks = {prenoderecvrank};
              }
              tmp_result = SingleFlow(
                  g_flow_id,
                  rank_it->first,
                  rank_it->second[1],
                  data_size,
                  prevranks,
                  {partner_flow_id},
                  {},
                  ring_id,
                  chunk_id,
                  chunkcout,
                  "RING");
              result[std::make_pair(ring_id, partner_flow_id)].child_flow_id.push_back(g_flow_id);
              task_list2[rank_it->first] = tmp_result;
              result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
              g_flow_id++;
            } else { 
              prevranks.clear();
              if(rank_it->second[0]!=-1){
                prevranks = {rank_it->second[0]};
              }
              tmp_result = SingleFlow(
                  g_flow_id,
                  rank_it->first,
                  rank_it->second[1],
                  data_size,
                  prevranks,
                  {partner_flow_id},
                  {},
                  ring_id,
                  chunk_id,
                  chunkcout,
                  "RING");
              result[std::make_pair(ring_id, partner_flow_id)].child_flow_id.push_back(g_flow_id);
              task_list2[rank_it->first] = tmp_result;
              result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
              g_flow_id++;
            }
          }
          task_list = task_list2;
          chunk_id++;
        }
        send_size += real_chunksize;
      }
    }
    rank2flowmodels.clear();
    for(auto flow_models_it = result.begin();flow_models_it!=result.end();flow_models_it++){
      int src = flow_models_it->second.src;
      int dst = flow_models_it->second.dest;
      rank2flowmodels[src][std::make_pair(flow_models_it->first.first,flow_models_it->first.second)]=flow_models_it->second;
      rank2flowmodels[dst][std::make_pair(flow_models_it->first.first,flow_models_it->first.second)]=flow_models_it->second;
    }
    for(auto it = rank2flowmodels.begin();it!=rank2flowmodels.end();it++){
      rank2pflowmodels[it->first] = std::make_shared<FlowModels>(it->second);
    }
    logRankToFlowModels(NcclLogLevel::INFO, "RING AllReduce", rank2pflowmodels);
    return rank2pflowmodels;
  }

  std::shared_ptr<FlowModels> MockNcclGroup::genallReduceNVLSTreeFlowModels(
      GroupType type,
      int rank,
      uint64_t data_size) {
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    GroupInfo gp_info;
    int gp_idx;
    int chunk_count = 1;
    int chunk_size;
    NVLStreechannels nvlstreechannels;
    NVLStreechannels::iterator nvlstree;
    FlowModels result = {};
    if(GroupIndex.count(std::make_pair(rank,type)) == 0){
      NcclLog->writeLog(NcclLogLevel::ERROR,"There is no relevant group info, resulting in an error in generating genallReduceNVLSTreeFlowModels.");
      return nullptr;
    } 
    gp_idx = GroupIndex[std::make_pair(rank,type)];
    gp_info = AllGroups[gp_idx];
    nvlstreechannels = AllNVLStreechannels[gp_idx];
    NcclLog->writeLog(NcclLogLevel::DEBUG," nvlstreechannels.size()  %d",nvlstreechannels.size());
    chunk_size = data_size / nvlstreechannels.size() / chunk_count;
    for (nvlstree = nvlstreechannels.begin();
         nvlstree != nvlstreechannels.end();
         nvlstree++) {
      std::map<int, std::vector<ncclChannelNode*>>::iterator nvlstreenodes_it;
      if (rank == 0) {
        for (nvlstreenodes_it = nvlstree->second.begin();
             nvlstreenodes_it != nvlstree->second.end();
             nvlstreenodes_it++) {
          NcclLog->writeLog(NcclLogLevel::DEBUG," rank  %d nvls tree nodes ",nvlstreenodes_it->first);
          int i = 0;
          for (auto nvlstreenode : nvlstreenodes_it->second) {
            NcclLog->writeLog(NcclLogLevel::DEBUG," node  %d rank  %d",i,nvlstreenode->rank);
            if(nvlstreenode->up!=nullptr){
              NcclLog->writeLog(NcclLogLevel::DEBUG," up  %d",nvlstreenode->up->rank);
            }
            NcclLog->writeLog(NcclLogLevel::DEBUG," down ");
            for (auto down : nvlstreenode->down) {
              NcclLog->writeLog(NcclLogLevel::DEBUG,"%d ",down->rank);
            }
          }
        }
      }
      std::unordered_map<ncclChannelNode*, int> upinDegree;
      std::unordered_map<ncclChannelNode*, int> downinDegree;
      std::unordered_map<ncclChannelNode*, std::vector<int>> nodeprevs;
      for (int ck = 0; ck < chunk_count; ck++) {
        nodeprevs = {};
        std::vector<ncclChannelNode*> ncclchannelnodes;
        for (auto nvlstreenodes : nvlstree->second) {
          for (auto nvlstreenode : nvlstreenodes.second) {
            ncclchannelnodes.push_back(nvlstreenode);
            upinDegree[nvlstreenode] = nvlstreenode->down.size();
            if (nvlstreenode->up == nullptr)
              downinDegree[nvlstreenode] = 0;
            else
              downinDegree[nvlstreenode] = 1;
          }
        }
        generate_flow_model_nvls_tree_allreduce_up(
            ncclchannelnodes,
            upinDegree,
            nodeprevs,
            chunk_size,
            ck,
            chunk_count,
            nvlstree->first,
            result);
        generate_flow_model_nvls_tree_allreduce_down(
            ncclchannelnodes,
            downinDegree,
            nodeprevs,
            chunk_size,
            ck,
            chunk_count,
            nvlstree->first,
            result);
      }
    }
    std::shared_ptr<FlowModels> ptr_result =
        std::make_shared<FlowModels>(result);
    return ptr_result;
  }

  FlowModels MockNcclGroup::generate_flow_model_nvls_tree_allreduce_up(
      std::vector<ncclChannelNode*> nvlstreenodes,
      std::unordered_map<ncclChannelNode*, int> upinDegree,
      std::unordered_map<ncclChannelNode*, std::vector<int>>& nodeprevs,
      int chunk_size,
      int chunk_id,
      int chunk_count,
      int channle_id,
      FlowModels& result) {
    std::queue<ncclChannelNode*> q;
    SingleFlow tmp_result;
    for (auto entry : upinDegree) {
      if (entry.second == 0) {
        q.push(entry.first);
        nodeprevs[entry.first] = {};
      }
    }
    std::string conn_tag = "NVLS_TREE";
    while (!q.empty()) {
      ncclChannelNode* current = q.front();
      q.pop();
      if (current->up != nullptr) {
        upinDegree[current->up]--;
        std::vector<int> _prev;
        if (current->down.size() == 0)
          _prev = {current->up->rank};
        else {
          for (auto down : current->down) {
            _prev.push_back(down->rank);
          }
        }
        tmp_result = SingleFlow(
            g_flow_id,
            current->rank,
            current->up->rank,
            chunk_size,
            _prev,
            nodeprevs[current],
            {},
            channle_id,
            chunk_id,
            chunk_count,
            conn_tag);
        for (int parent_flow_id : nodeprevs[current]) {
          result[std::make_pair(channle_id, parent_flow_id)]
              .child_flow_id.push_back(g_flow_id);
        }
        result[std::make_pair(channle_id, g_flow_id)] = tmp_result;
        g_flow_id++;
        nodeprevs[current->up].push_back(tmp_result.flow_id);
        nodeprevs.erase(current);
        if (upinDegree[current->up] == 0)
          q.push(current->up);
      }
    }
    return result;
  }

  FlowModels MockNcclGroup::generate_flow_model_nvls_tree_allreduce_down(
      std::vector<ncclChannelNode*> nvlstreenodes,
      std::unordered_map<ncclChannelNode*, int> downinDegree,
      std::unordered_map<ncclChannelNode*, std::vector<int>>& nodeprevs,
      int chunk_size,
      int chunk_id,
      int chunk_count,
      int channle_id,
      FlowModels& result) {
    std::queue<ncclChannelNode*> q;
    SingleFlow tmp_result;
    for (auto entry : downinDegree) {
      if (entry.second == 0) {
        q.push(entry.first);
      }
    }
    std::string conn_tag = "NVLS_TREE";
    while (!q.empty()) {
      ncclChannelNode* current = q.front();
      q.pop();

      if (current->down.size() > 0) {
        for (ncclChannelNode* down : current->down) {
          downinDegree[down]--;
          std::vector<int> _prev;
          if (current->up == nullptr) {
            for (ncclChannelNode* down1 : current->down) {
              _prev.push_back(down1->rank);
            }
          } else {
            _prev = {current->up->rank};
          }
          tmp_result = SingleFlow(
              g_flow_id,
              current->rank,
              down->rank,
              chunk_size,
              _prev,
              nodeprevs[current],
              {},
              channle_id,
              chunk_id,
              chunk_count,
              conn_tag);
          for (int parent_flow_id : nodeprevs[current]) {
            result[std::make_pair(channle_id, parent_flow_id)]
                .child_flow_id.push_back(g_flow_id);
          }
          result[std::make_pair(channle_id, g_flow_id)] = tmp_result;
          g_flow_id++;
          nodeprevs[down].push_back(tmp_result.flow_id);
          if (downinDegree[down] == 0)
            q.push(down);
        }
      }
    }
    return result;
  }

  std::map<int,std::shared_ptr<FlowModels>> MockNcclGroup::genAllreduceNVLSFlowModels(GroupType type,int rank,uint64_t data_size){
    GroupInfo gp_info;
    int gp_idx;
    // A-Task1 (calib round2, 260611): NVLS chunk_count tunable via AS_NVLS_CHUNK_COUNT.
    // Larger chunk_count -> smaller per-flow size -> better pipeline overlap -> higher
    // effective busbw (closer to real machine, which the default 4 underestimates).
    // Default stays 4 so existing behavior is unchanged unless explicitly overridden.
    static const int chunk_count = [](){
      const char* e = std::getenv("AS_NVLS_CHUNK_COUNT");
      int v = e ? atoi(e) : 4;
      return (v >= 1) ? v : 4;
    }();
    std::map<int,FlowModels>rank2flowmodels;
    std::map<int,std::shared_ptr<FlowModels>>rank2pflowmodels;
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    if(GroupIndex.count(std::make_pair(rank,type))==0){
      NcclLog->writeLog(NcclLogLevel::ERROR,"There is no corresponding group info , resulting in an error in genAllreduceNVLSFlowModels.");
      return {};
    }
    gp_idx = GroupIndex[std::make_pair(rank,type)];
    gp_info = AllGroups[gp_idx];
    FlowModels result={};
    SingleFlow treeflow;
    if(gp_info.nNodes == 1){  
      std::vector<int>NVswitchs = gp_info.NVSwitchs;
      std::vector<int>ranks = gp_info.Ranks;
      int chunk_size = data_size / chunk_count;
      for(int ck =0;ck<chunk_count;ck++){
        for(int j = 0;j<NVswitchs.size();j++){
          std::vector<int>prevs;
          std::vector<int>parents;
          for(int k = 0;k<ranks.size();k++){
            treeflow = SingleFlow(g_flow_id,ranks[k],NVswitchs[j],chunk_size,{NVswitchs[j]},{},{},0,ck,chunk_count,"NVLS");
            result[std::make_pair(0,g_flow_id)]=treeflow;
            prevs.push_back(ranks[k]);
            parents.push_back(g_flow_id);
            g_flow_id++;
          }
          for(int k =0;k<ranks.size();k++){
            treeflow = SingleFlow(g_flow_id,NVswitchs[j],ranks[k],chunk_size,prevs,parents,{},0,ck,chunk_count,"NVLS");
            result[std::make_pair(0,g_flow_id)]=treeflow;
            for(auto parent:parents){
              result[std::make_pair(0,parent)].child_flow_id.push_back(g_flow_id);
            }
            g_flow_id++;
          }
        }

      }

    }
    rank2flowmodels.clear();
    for(auto flow_models_it = result.begin();flow_models_it!=result.end();flow_models_it++){
      int src = flow_models_it->second.src;
      int dst = flow_models_it->second.dest;
      rank2flowmodels[src][std::make_pair(flow_models_it->first.first,flow_models_it->first.second)]=flow_models_it->second;
      rank2flowmodels[dst][std::make_pair(flow_models_it->first.first,flow_models_it->first.second)]=flow_models_it->second;
    }
    for(auto it = rank2flowmodels.begin();it!=rank2flowmodels.end();it++){
      rank2pflowmodels[it->first] = std::make_shared<FlowModels>(it->second);
    }
    return rank2pflowmodels;
  }

  std::shared_ptr<FlowModels> MockNcclGroup::genAllReduceTreeFlowModels(GroupType type , int rank,uint64_t data_size){
    int chunk_count = 64;
    int chunk_size;
    SingleFlow tmp_result;
    FlowModels result1 = {};
    FlowModels result = {};
    std::map<int,int> task_list = {}; 
    std::map<int,std::map<int,ncclTree>>::iterator tree;
    GroupInfo gp_info;
    int gp_idx;
    TreeChannels treechannels;
    MockNcclLog* NcclLog = MockNcclLog::getInstance();

    if(GroupIndex.count(std::make_pair(rank,type))==0){
      NcclLog->writeLog(NcclLogLevel::ERROR,"There is no corresponding group info, resulting in an error in genAllReduceTreeFlowModels.");
      return {};
    }
    gp_idx = GroupIndex[std::make_pair(rank,type)];
    if(Alltreechannels.count(gp_idx)==0){
      NcclLog->writeLog(NcclLogLevel::ERROR,"No tree channels available for gp_idx=%d, falling back.", gp_idx);
      return {};
    }
    gp_info = AllGroups[gp_idx];
    treechannels = Alltreechannels[gp_idx];
    chunk_size = data_size / treechannels.size() / chunk_count;
    for(tree = treechannels.begin(); tree !=treechannels.end(); tree++) {
      std::unordered_map<int, int> upinDegree;
      std::unordered_map<int, int> downinDegree;
      std::unordered_map<int,std::vector<int>> nodeprevs;
      for(int ck = 0; ck < chunk_count; ck++){
        nodeprevs = {};
        for(auto treenode:tree->second){
          upinDegree[treenode.first] = treenode.second.down.size();
          if(treenode.second.up == -1)
            downinDegree[treenode.first] = 0;
          else
            downinDegree[treenode.first] = 1;
        }
        generate_flow_model_tree_allreduce_up(tree->second,upinDegree,nodeprevs,chunk_size,ck,chunk_count,tree->first,result);
        generate_flow_model_tree_allreduce_down(tree->second,downinDegree,nodeprevs,chunk_size,ck,chunk_count,tree->first,result);
      }
    }
    std::shared_ptr<FlowModels> ptr_result = std::make_shared<FlowModels>(result);
    return  ptr_result;
  }

  FlowModels MockNcclGroup::generate_flow_model_tree_allreduce_up(std::map<int,ncclTree> &nodes,std::unordered_map<int, int> upinDegree,std::unordered_map<int,std::vector<int>>& nodeprevs,int chunk_size,int chunk_id,int chunk_count,int channle_id,FlowModels& result){
    std::queue<ncclTree> q;
    std::map<int,int> task_list2={};
    SingleFlow tmp_result;
    for (auto entry : upinDegree) {
      if (entry.second == 0) {
        q.push(nodes[entry.first]);
        nodeprevs[entry.first]={};
      }
    }
    std::string conn_tag = "TREE_INIT";
    while (!q.empty()) {
      ncclTree current = q.front();
      q.pop();
      if(current.up != -1) {
        upinDegree[current.up]--; 
        std::vector<int> _prev; 
        if (current.down.size() == 0)
          _prev = {current.up};
        else
          _prev = current.down;
        tmp_result = SingleFlow(g_flow_id,current.rank,current.up,chunk_size,_prev,nodeprevs[current.rank],{},channle_id,chunk_id,chunk_count,conn_tag);
        for(int parent_flow_id:nodeprevs[current.rank])
          result[std::make_pair(channle_id,parent_flow_id)].child_flow_id.push_back(g_flow_id);  
        result[std::make_pair(channle_id,g_flow_id)] = tmp_result;
        g_flow_id++;
        nodeprevs[current.up].push_back(tmp_result.flow_id); 
        nodeprevs.erase(current.rank); 
        if(upinDegree[current.up] == 0)
          q.push(nodes[current.up]);
      }
    }
    return result;
  }

  FlowModels MockNcclGroup::generate_flow_model_tree_allreduce_down(std::map<int,ncclTree> &nodes,std::unordered_map<int, int> downinDegree,std::unordered_map<int,std::vector<int>>& nodeprevs,int chunk_size,int chunk_id,int chunk_count,int channle_id,FlowModels& result){
    std::queue<ncclTree> q;
    std::map<int,int> task_list2={};
    SingleFlow tmp_result;
    for (auto entry : downinDegree) {
      if (entry.second == 0) {
        q.push(nodes[entry.first]);
      }
    }
    std::string conn_tag =  "TREE_INIT";
    while (!q.empty()) {
      ncclTree current = q.front();
      q.pop();
      if(current.down.size() >0 ) {
        for(int down:current.down) {
          downinDegree[down] --;
          std::vector<int> _prev;
          if (current.up == -1) {
            _prev = current.down;
          } else {
            _prev = {current.up};
          }
          tmp_result = SingleFlow(g_flow_id,current.rank,down,chunk_size,_prev,nodeprevs[current.rank],{},channle_id,chunk_id,chunk_count,conn_tag);
          for(int parent_flow_id:nodeprevs[current.rank])
            result[std::make_pair(channle_id,parent_flow_id)].child_flow_id.push_back(g_flow_id);
          result[std::make_pair(channle_id,g_flow_id)] = tmp_result;
          g_flow_id++;
          nodeprevs[down].push_back(tmp_result.flow_id); 
          if(downinDegree[down] == 0)
            q.push(nodes[down]);
        }
      }
    }
    return result;
  }

  std::map<int, std::shared_ptr<FlowModels>> MockNcclGroup::genAllGatherFlowModels(GroupType type, int rank, uint64_t data_size,bool& msccl) {
      // Assuming AstraSim::ComType::All_Gather is an enum or constant
      ncclInfo* ncc_info = get_algo_proto_info(type, rank, AstraSim::ComType::All_Gather, data_size,msccl);
      MockNcclLog* NcclLog = MockNcclLog::getInstance();

      switch (ncc_info->algorithm) {
          case NCCL_ALGO_TREE:
          case NCCL_ALGO_RING:
              {
                NcclLog->writeLog(NcclLogLevel::DEBUG, "Using genAllGatherRingFlowModels for type %d, rank %d, data_size %lu", type, rank, data_size);
                return genAllGatherRingFlowModels(type, rank, data_size);
              }

          case NCCL_ALGO_NVLS:
              // It's generally better to throw an exception or handle the error gracefully than calling exit(1)
              // For example: throw std::runtime_error("NCCL_ALGO_NVLS is not supported");
              exit(1);

          case NCCL_ALGO_NVLS_TREE:
              return {};

          case NCCL_ALGO_MSCCL:
          {
              NcclLog->writeLog(NcclLogLevel::INFO, "Using genCustomFlowModels for operator %s type %s, rank %d, data_size %lu, protocol %d", to_cstr(ncc_info->coll), to_cstr(type), rank, data_size, ncc_info->protocol);
              auto retval = genCustomFlowModels(type, rank, data_size, ncc_info);
              if (!retval) { // A more idiomatic check for std::optional
                  NcclLog->writeLog(NcclLogLevel::INFO, "genCustomFlowModels failed, falling back to NCCL algorithm selection for operator %s type %s, rank %d, data_size %lu, protocol %d", to_cstr(ncc_info->coll), to_cstr(type), rank, data_size, ncc_info->protocol);
                  // Re-run the full NCCL selection without MSCCL (cannot recurse again:
                  // with msccl == false the MSCCL case is never selected).
                  msccl = false;
                  return genAllGatherFlowModels(type, rank, data_size, msccl);
              } else {
                  return retval.value();
              }
          }

          default:
              NcclLog->writeLog(NcclLogLevel::ERROR, "Unsupported algorithm %d for AllGather in type %d, rank %d, data_size %lu", ncc_info->algorithm, type, rank, data_size);
              return {};
      }
      return {};
  }

  std::map<int, std::shared_ptr<FlowModels>> MockNcclGroup::genAllGatherRingFlowModels(GroupType type, int rank, uint64_t data_size) {
    FlowModels result = {};
    std::map<int,FlowModels> rank2flowmodels;
    std::map<int,std::shared_ptr<FlowModels>> rank2pflowmodels;
    std::map<int,SingleFlow> task_list = {};
    std::map<int,SingleFlow> task_list2 = {};
    SingleFlow tmp_result;
    uint64_t chunksize;
    uint64_t send_size;
    int nranks;
    int chunkcount;
    int chunkid;
    GroupInfo gp_info;
    int gp_idx;
    RingChannels ringchannels;
    MockNcclLog* NcclLog = MockNcclLog::getInstance();

    if(GroupIndex.count(std::make_pair(rank,type))==0){
      NcclLog->writeLog(NcclLogLevel::ERROR,"There is no corresponding group info and group ring channel, resulting in an error in generating the flow model.");
      return {};
    } else {
      gp_idx = GroupIndex[std::make_pair(rank,type)];
      ringchannels = Allringchannels[gp_idx];
      gp_info = AllGroups[gp_idx];
    }

    nranks = gp_info.nRanks;
    chunkcount = gp_info.nRanks-1;
    chunksize = data_size / nranks / ringchannels.size();
    data_size = data_size / nranks / ringchannels.size();
    bool PXN_ENABLE = false;
    const char* PXN_ENV = std::getenv("AS_PXN_ENABLE");
    if(PXN_ENV && strcmp(PXN_ENV,"1")==0){
      PXN_ENABLE = true;
    } else {
      PXN_ENABLE = false;
    }
    for(auto it = ringchannels.begin(); it !=ringchannels.end(); it++) {
      auto& ring = it->second;
      auto ring_id = it->first;
      task_list = {};
      send_size = 0;
      chunkid = 0;
      while (send_size < data_size) {
        uint64_t real_chunksize = std::min(chunksize, data_size - send_size);
        int prenoderecvrank = ring.rbegin()->second[2];
        int prenodesendrank = ring.rbegin()->second[3];
        int curnoderecvrank = ring.begin()->second[2];
        int curnodesendrank = ring.begin()->second[3];
        std::vector<int> prevranks = {};
        for (auto rank_it = ring.begin(); rank_it != ring.end(); rank_it++) {
          int cur_rank = rank_it->first;
          if (curnoderecvrank != rank_it->second[2] &&
              curnodesendrank != rank_it->second[3]) {
            prenoderecvrank = curnoderecvrank;
            prenodesendrank = curnodesendrank;
            curnoderecvrank = rank_it->second[2];
            curnodesendrank = rank_it->second[3];
          }
          if (rank_it->second[3] == cur_rank &&
              rank_it->second[2] != cur_rank && gp_info.nNodes > 1 &&
              PXN_ENABLE) { 
            prevranks.clear();
            if(rank_it->second[0]!=-1){
              prevranks = {rank_it->second[0]};
            }
            tmp_result = SingleFlow(
                g_flow_id,
                rank_it->first,
                rank_it->second[2],
                data_size,
                prevranks,
                {},
                {g_flow_id + 1},
                ring_id,
                chunkid,
                chunkcount,
                "RING");
            result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
            g_flow_id++;
            prevranks.clear();
            if(rank_it->first!=-1){
              prevranks = {rank_it->first};
            }
            tmp_result = SingleFlow(
                g_flow_id,
                rank_it->second[2],
                rank_it->second[1],
                data_size,
                prevranks,
                {g_flow_id - 1},
                {},
                ring_id,
                chunkid,
                chunkcount,
                "PXN_INIT");
            result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
            task_list[rank_it->first] = tmp_result;
            g_flow_id++;
          } else if (
              rank_it->second[2] == cur_rank &&
              rank_it->second[3] != cur_rank && gp_info.nNodes > 1 &&
              PXN_ENABLE) {
            prevranks.clear();
            if(prenoderecvrank!=-1){
              prevranks = {prenoderecvrank};
            }
            tmp_result = SingleFlow(
                g_flow_id,
                rank_it->first,
                rank_it->second[1],
                data_size,
                prevranks,
                {},
                {},
                ring_id,
                chunkid,
                chunkcount,
                "RING");
            result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
            task_list[rank_it->first] = tmp_result;
            g_flow_id++;
          } else { 
            prevranks.clear();
            if (rank_it->second[0] != -1) {
              prevranks = {rank_it->second[0]};
            }
            tmp_result = SingleFlow(
                g_flow_id,
                rank_it->first,
                rank_it->second[1],
                data_size,
                prevranks,
                {},
                {},
                ring_id,
                chunkid,
                chunkcount,
                "RING");
            result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
            task_list[rank_it->first] = tmp_result;
            g_flow_id++;
          }
        }
        chunkid++;
        for (int i = 0; i < nranks - 2; i++) {
          task_list2 = {};
          prenoderecvrank = ring.rbegin()->second[2];
          prenodesendrank = ring.rbegin()->second[3];
          curnoderecvrank = ring.begin()->second[2];
          curnodesendrank = ring.begin()->second[3];
          for (auto rank_it = ring.begin(); rank_it != ring.end(); rank_it++) {
            if (curnoderecvrank != rank_it->second[2] &&
                curnodesendrank != rank_it->second[3]) {
              prenoderecvrank = curnoderecvrank;
              prenodesendrank = curnodesendrank;
              curnoderecvrank = rank_it->second[2];
              curnodesendrank = rank_it->second[3];
            }
            int cur_rank = rank_it->first;
            // for chunks > 0, set previous chunk as parent (e.g., 0 -> 1 chunk 0 is parent for 1 -> 2 chunk 1)
            int partner_flow_id = task_list[rank_it->second[0]].flow_id;
            if (rank_it->second[3] == cur_rank &&
                rank_it->second[2] != cur_rank && gp_info.nNodes > 1 &&
                PXN_ENABLE) { 
              prevranks.clear();
              if(rank_it->second[0]!=-1){
                prevranks = {rank_it->second[0]};
              }
              tmp_result = SingleFlow(
                  g_flow_id,
                  rank_it->first,
                  rank_it->second[2],
                  data_size,
                  prevranks,
                  {partner_flow_id},
                  {g_flow_id + 1},
                  ring_id,
                  chunkid,
                  chunkcount,
                  "RING");
              result[std::make_pair(ring_id, partner_flow_id)].child_flow_id.push_back(g_flow_id);
              result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
              g_flow_id++;
              prevranks.clear();
              if (rank_it->first != -1) {
                prevranks = {rank_it->first};
              }
              tmp_result = SingleFlow(
                  g_flow_id,
                  rank_it->second[2],
                  rank_it->second[1],
                  data_size,
                  prevranks,
                  {g_flow_id - 1},
                  {},
                  ring_id,
                  chunkid,
                  chunkcount,
                  "PXN");
              task_list2[rank_it->first] = tmp_result;
              result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
              g_flow_id++;
            } else if (
                rank_it->second[2] == cur_rank &&
                rank_it->second[3] != cur_rank && gp_info.nNodes > 1 &&
                PXN_ENABLE) {
              prevranks.clear();
              if (prenoderecvrank != -1) {
                prevranks = {prenoderecvrank};
              }
              tmp_result = SingleFlow(
                  g_flow_id,
                  rank_it->first,
                  rank_it->second[1],
                  data_size,
                  prevranks,
                  {partner_flow_id},
                  {},
                  ring_id,
                  chunkid,
                  chunkcount,
                  "RING");
              result[std::make_pair(ring_id, partner_flow_id)].child_flow_id.push_back(g_flow_id);
              task_list2[rank_it->first] = tmp_result;
              result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
              g_flow_id++;
            } else { 
              prevranks.clear();
              if (rank_it->second[0] != -1) {
                prevranks = {rank_it->second[0]};
              }
              tmp_result = SingleFlow(
                  g_flow_id,
                  rank_it->first,
                  rank_it->second[1],
                  data_size,
                  prevranks,
                  {partner_flow_id},
                  {},
                  ring_id,
                  chunkid,
                  chunkcount,
                  "RING");
              result[std::make_pair(ring_id, partner_flow_id)].child_flow_id.push_back(g_flow_id);
              task_list2[rank_it->first] = tmp_result;
              result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
              g_flow_id++;
            }
          }
          task_list = task_list2;
          chunkid++;
        }
        send_size += real_chunksize;
      }
    }
    for(auto flow_models_it = result.begin();flow_models_it!=result.end();flow_models_it++){
      int src = flow_models_it->second.src;
      int dst = flow_models_it->second.dest;
      rank2flowmodels[src][std::make_pair(flow_models_it->first.first,flow_models_it->first.second)]=flow_models_it->second;
      rank2flowmodels[dst][std::make_pair(flow_models_it->first.first,flow_models_it->first.second)]=flow_models_it->second;
    }
    for(auto it = rank2flowmodels.begin();it!=rank2flowmodels.end();it++){
      rank2pflowmodels[it->first] = std::make_shared<FlowModels>(it->second);
    }
    logRankToFlowModels(NcclLogLevel::INFO, "RING AllGather", rank2pflowmodels);
    return rank2pflowmodels;
  }
  
  ncclChannelNode* MockNcclGroup::gen_nvls_tree_intra_channels(std::vector<int> intra_topo,std::map<int, vector<ncclChannelNode*>> &nvlstreechannel){
    ncclChannelNode* root = new ncclChannelNode(-1,intra_topo[0],nullptr,{});
    nvlstreechannel[root->rank].push_back(root);
    ncclChannelNode* nvswitch = new ncclChannelNode(-1,intra_topo[1],root,{});
    nvlstreechannel[nvswitch->rank].push_back(nvswitch);
    root->down.push_back(nvswitch);
    for(int i =2;i<intra_topo.size();i++){
      ncclChannelNode*leaf = new ncclChannelNode(-1,intra_topo[i],nvswitch,{});
      nvswitch->down.push_back(leaf);
      nvlstreechannel[leaf->rank].push_back(leaf);
    }
    return root;
  }

  TreeChannels MockNcclGroup::get_nvls_channels(int rank,GroupType type){
    GroupInfo gp_info;
    int gp_idx;
    TreeChannels nvlschannel;
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    if(GroupIndex.count(std::make_pair(rank,type))==0){
      NcclLog->writeLog(NcclLogLevel::ERROR,"There is no corresponding group info and group ring channel, resulting in an error in get_nvls_channels.");
      return {};
    }
    gp_idx = GroupIndex[std::make_pair(rank,type)];
    gp_info = AllGroups[gp_idx];
    // Communicator initialization also requests NVLS channels for
    // Ethernet-only groups, which have no NVSwitch.
    if (gp_info.NVSwitchs.empty()) {
      return {};
    }
    if (gp_info.nNodes > 1) {
      NcclLog->writeLog(NcclLogLevel::DEBUG, "get_nvls_channels: NVLS not supported for multi-node group (rank %d, nNodes %d)", rank, gp_info.nNodes);
      return {};
    } else {
      std::vector<int> ranks = gp_info.Ranks;
      int NVswitch = gp_info.NVSwitchs[0];
      for (int i = 0; i < ranks.size(); i++) {
        nvlschannel[0][ranks[i]] = ncclTree(-1, ranks[i], NVswitch, {});
      }
      nvlschannel[0][ranks.size()] = ncclTree(-1, NVswitch, -1, ranks);
    }
    AllNVLSchannels[gp_idx] = nvlschannel;
    return nvlschannel;
  }

  NVLStreechannels MockNcclGroup::get_nvls_tree_channels(int rank,GroupType type){
    std::map<int,std::map<int,std::vector<ncclChannelNode*>>> nvlstreechannels;
    std::map<int,std::vector<int>>localrings;
    std::map<int,std::vector<int>>::iterator ring_it;
    GroupInfo gp_info;
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    int current;
    int nNodes;
    int nlocalRanks;
    int delta;
    int gp_idx;
    if(GroupIndex.count(std::make_pair(rank,type))==0){
      NcclLog->writeLog(NcclLogLevel::ERROR,"There is no corresponding group info , resulting in an error in get_nvls_tree_channels.");
      return {};
    }
    gp_idx = GroupIndex[std::make_pair(rank,type)];
    gp_info = AllGroups[gp_idx];
    if(AllNVLStreechannels.count(gp_idx)){
      return AllNVLStreechannels[gp_idx];
    }
    std::vector<DoubleBinaryTreeNode*>roots;
    roots = genInterDouBinTree(gp_info);

    nNodes = gp_info.nNodes;
    nlocalRanks = gp_info.nRanks/nNodes;
    localrings = gen_local_ring(rank,type);
    delta = nNodes > 1 ? gp_info.Ranks[nlocalRanks]-gp_info.Ranks[0] : 0;
    std::map<int,std::vector<int>>rings;
    for(ring_it = localrings.begin();ring_it != localrings.end();ring_it++) {
      for(int i = 0; i < nNodes; i++) {
        for(int j = 0; j < nlocalRanks; j++) {
          current = ring_it->second[j] + i * delta;
          rings[ring_it->first].push_back(current);
        }
      }
    }
    std::map<int, std::map<int, std::vector<int>>>
        allnode2ranks; 
    for (ring_it = rings.begin(); ring_it != rings.end(); ring_it++) {
      int nrankspernode = gp_info.nRanks / nNodes;
      for (int i = 0; i < gp_info.nNodes; i++) {
        for (int j = 0; j < nrankspernode; j++) {
          allnode2ranks[ring_it->first][i].push_back(
              ring_it->second[i * nrankspernode + j]);
        }
      }
    }

    std::map<int, std::map<int, std::vector<int>>>::iterator allnode2ranks_it;
    int channel_id = 0;
    std::map<int, std::vector<int>> node2ranks = allnode2ranks[0];
    for (DoubleBinaryTreeNode* root : roots) {
      for (int index = 0; index < nlocalRanks; index++) {
        std::map<int, vector<ncclChannelNode*>> nvlstreechannel;
        std::map<int,ncclChannelNode*> nodencclchannlenodes;
        for (int i = 0; i < nNodes; i++) {
          std::vector<int> noderanks = node2ranks[i];
          std::vector<int> intra_topo;
          intra_topo.push_back(noderanks[index]);
          intra_topo.push_back(gp_info.NVSwitchs[i]);
          intra_topo.insert(
              intra_topo.end(), noderanks.begin(), noderanks.end());
          NcclLog->writeLog(NcclLogLevel::DEBUG," node  %d intra_topo",i);
          for(auto num:intra_topo){
            NcclLog->writeLog(NcclLogLevel::DEBUG," %d",num);
          }
          ncclChannelNode* root =
              gen_nvls_tree_intra_channels(intra_topo, nvlstreechannel);
          nodencclchannlenodes[i] = root;
        }

        std::map<int, std::vector<ncclChannelNode*>>::iterator nvlstreenodes_it;
        if (rank == 0) {
          for (nvlstreenodes_it = nvlstreechannel.begin();
               nvlstreenodes_it != nvlstreechannel.end();
               nvlstreenodes_it++) {
              NcclLog->writeLog(NcclLogLevel::DEBUG," rank  %d nvls tree nodes ",nvlstreenodes_it->first);
            int i = 0;
            for (auto nvlstreenode : nvlstreenodes_it->second) {
              NcclLog->writeLog(NcclLogLevel::DEBUG," node  %d rank  %d",i,nvlstreenode->rank);
              if(nvlstreenode->up!=nullptr)
                NcclLog->writeLog(NcclLogLevel::DEBUG," up  %d",nvlstreenode->up->rank);
              NcclLog->writeLog(NcclLogLevel::DEBUG," down ");
              for (auto down : nvlstreenode->down) {
                NcclLog->writeLog(NcclLogLevel::DEBUG," %d ",down->rank);
              }
            }
          }
        }

        gen_nvls_tree_inter_channels(
            root, nodencclchannlenodes, nvlstreechannel);

        nvlstreechannels[channel_id] = nvlstreechannel;
        channel_id++;
      }
    }
    AllNVLStreechannels[gp_idx] = nvlstreechannels;
    return nvlstreechannels;
  }

  ncclChannelNode* MockNcclGroup::gen_nvls_tree_inter_channels(
      DoubleBinaryTreeNode* root,
      std::map<int, ncclChannelNode*> nodencclchannlenodes,
      std::map<int, vector<ncclChannelNode*>>& nvlstreechannel) {
      MockNcclLog* NcclLog = MockNcclLog::getInstance();
    if (root == nullptr)
      return nullptr;
    else {
      NcclLog->writeLog(NcclLogLevel::DEBUG,"before root->right:  %d",root->right);
      NcclLog->writeLog(NcclLogLevel::DEBUG,"before root->left:  %d",root->left);
      if (root->left != nullptr) {
        NcclLog->writeLog(NcclLogLevel::DEBUG,"after root->left:  %d",root->left);
        ncclChannelNode* cur = nodencclchannlenodes[root->node];
        ncclChannelNode* left = nodencclchannlenodes[root->left->node];
        cur->down.push_back(left);
        left->up = cur;
        gen_nvls_tree_inter_channels(root->left,nodencclchannlenodes,nvlstreechannel);
      }
      if (root->right != nullptr) {
        NcclLog->writeLog(NcclLogLevel::DEBUG,"after root->right:  %d",root->right);
        ncclChannelNode* cur = nodencclchannlenodes[root->node];
        ncclChannelNode* right = nodencclchannlenodes[root->right->node];
        cur->down.push_back(right);
        right->up = cur;
        gen_nvls_tree_inter_channels(root->right,nodencclchannlenodes,nvlstreechannel);
      }
    }
    return nodencclchannlenodes[root->node];
  }

  TreeChannels MockNcclGroup::gettreechannels(int rank, GroupType type){
    TreeChannels treechannels;
    std::map<int,std::vector<int>>localrings;
    std::map<int,std::vector<int>>::iterator ring_it;
    GroupInfo gp_info;
    int gp_idx;
    int current;
    int nNodes;
    int nlocalRanks;
    int delta;
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    if(GroupIndex.count(std::make_pair(rank,type))==0){
      NcclLog->writeLog(NcclLogLevel::ERROR,"There is no corresponding group info and group ring channel, resulting in an error in gettreechannels.");
      return {};
    }
    gp_idx = GroupIndex[std::make_pair(rank,type)];
    gp_info = AllGroups[gp_idx];
    if(Alltreechannels.count(gp_idx)){
      return Alltreechannels[gp_idx];
    }
  
    nNodes = gp_info.nNodes;
    nlocalRanks = gp_info.nRanks/nNodes;
    localrings = gen_local_ring(rank,type);
    delta = nNodes > 1 ? gp_info.Ranks[nlocalRanks]-gp_info.Ranks[0] : 0;
    std::map<int,std::vector<int>>rings;
    for(ring_it = localrings.begin();ring_it != localrings.end();ring_it++) {
      for(int i = 0; i < nNodes; i++) {
        for(int j = 0; j < nlocalRanks; j++) {
          current = ring_it->second[j] + i * delta; 
          rings[ring_it->first].push_back(current);
        }
      }
    }
    std::vector<DoubleBinaryTreeNode*> roots;
    roots = genInterDouBinTree(gp_info);
    std::map<int, std::map<int, std::vector<int>>>
        allnode2ranks; 
    for (ring_it = rings.begin(); ring_it != rings.end(); ring_it++) {
      int nrankspernode = gp_info.nRanks / nNodes;
      for (int i = 0; i < gp_info.nNodes; i++) {
        for (int j = 0; j < nrankspernode; j++) {
          allnode2ranks[ring_it->first][i].push_back(
              ring_it->second[i * nrankspernode + j]);
        }
      }
    }
    std::map<int, std::map<int, std::vector<int>>>::iterator allnode2ranks_it;
    int channel_id = 0;
    for (allnode2ranks_it = allnode2ranks.begin();
         allnode2ranks_it != allnode2ranks.end();
         allnode2ranks_it++) {
      std::map<int, std::vector<int>> node2ranks = allnode2ranks_it->second;
      for (DoubleBinaryTreeNode* root : roots) {
        std::map<int, ncclTree> treechannel;
        for (int rank : gp_info.Ranks) {
          ncclTree cur =  ncclTree(-1, rank, -1, {});
          treechannel[rank] = cur;
        }
        ConnInterIntraTree(root, node2ranks, treechannel);
        treechannels[channel_id] = treechannel;
        channel_id++;
      }
      Alltreechannels[gp_idx] = treechannels;
    }
    return treechannels;
  }

  void MockNcclGroup::ConnInterIntraTree(DoubleBinaryTreeNode*root,std::map<int,std::vector<int>>node2ranks,std::map<int,ncclTree>&treechannel) {
    if(root == nullptr) return;
    std::vector<int>ranks = node2ranks[root->node];
    for(int i=0;i<ranks.size()-1;i++) {
      ncclTree *current = &treechannel[ranks[i]];
      ncclTree *down = &treechannel[ranks[i+1]];
      current->down.push_back(ranks[i+1]);
      down->up=ranks[i];
    }

    if(root->left!=nullptr){
      ncclTree *current = &treechannel[ranks[0]];
      int downrank = node2ranks[root->left->node][0];
      ncclTree *down = &treechannel[downrank];
      current->down.push_back(downrank);
      down->up = ranks[0];
      ConnInterIntraTree(root->left,node2ranks,treechannel);
    }
    if(root->right!=nullptr){
      ncclTree *current = &treechannel[ranks[0]];
      int downrank = node2ranks[root->right->node][0];
      ncclTree *down = &treechannel[downrank];
      current->down.push_back(downrank);
      down->up = ranks[0];
      ConnInterIntraTree(root->right,node2ranks,treechannel);
    }
  }

  std::vector<MockNcclGroup::DoubleBinaryTreeNode*> MockNcclGroup::genInterDouBinTree(GroupInfo gp_info){
    vector<DoubleBinaryTreeNode*> q;
    vector<DoubleBinaryTreeNode*> tmp_q;
    vector<DoubleBinaryTreeNode*> result;
    int nNodes = gp_info.nNodes; 
    std::vector<int> nodes;
    for(int i = 0;i < nNodes; i++)
      nodes.push_back(i);
    for(int i = 0;i < nodes.size();i++){
      q.push_back(new DoubleBinaryTreeNode(nodes[i]));
    }
    while (q.size() > 1){
      tmp_q = {};
      int i = 0;
      for(i = 0;(i + 2) < q.size();i +=4){
        DoubleBinaryTreeNode* node0 = q[i];
        DoubleBinaryTreeNode* node1 = q[i+1];
        DoubleBinaryTreeNode* node2 = q[i+2];
        node1->left = node0;
        node1->right = node2;
        tmp_q.push_back(node1);
        if(i+3 < q.size()) {
          DoubleBinaryTreeNode* node3 = q[i+3];
          tmp_q.push_back((node3));
        }
      }
      if(q.size() - i == 1) {
        DoubleBinaryTreeNode* node0 = q[i];
        tmp_q.push_back(node0);
      } else if(q.size() - i == 2){
        DoubleBinaryTreeNode* node0 = q[i];
        DoubleBinaryTreeNode* node1 = q[i+1];
        node1->left = node0;
        tmp_q.push_back(node1);
      }
      q = tmp_q;
    }
    DoubleBinaryTreeNode* root1 = InterDouBinTreeShift(q[0],nodes);
    int chunk_count = 1;
    for(int i =0;i<chunk_count;i++){
      result.push_back(q[0]);
      result.push_back(root1);
    }
    return result;
  }

  MockNcclGroup::DoubleBinaryTreeNode* MockNcclGroup::InterDouBinTreeShift(DoubleBinaryTreeNode* root,std::vector<int>nodes){
    std::map<int,DoubleBinaryTreeNode*>node2treenode;
    std::map<int,int>rank2index;
    std::queue<DoubleBinaryTreeNode*>q;
    for(int i =0 ;i<nodes.size();i++) {
      node2treenode[nodes[i]] = new DoubleBinaryTreeNode(nodes[i]);
      rank2index[nodes[i]] = i;
    }
    q.push(root);
    while (!q.empty())
    {
      DoubleBinaryTreeNode* current = q.front();
      q.pop();
      int node = current->node;
      int nodeshift = nodes[(rank2index[node] + 1) % nodes.size()];
      DoubleBinaryTreeNode* currentshift = node2treenode[nodeshift];
      if(current->left != nullptr) {
        int leftnode = current->left->node;
        int leftnodeshift = nodes[(rank2index[leftnode] + 1) % nodes.size()];
        currentshift->left = node2treenode[leftnodeshift];
        q.push(current->left);
      }
      if(current->right != nullptr) {
        int rightnode = current->right->node;
        int rightnodeshift = nodes[(rank2index[rightnode] + 1) % nodes.size()];
        currentshift->right = node2treenode[rightnodeshift];
        q.push(current->right);
      }
    }
    return node2treenode[nodes[(rank2index[root->node] + 1) % nodes.size()]];
  }

  // Merged: internal MSCCL + v2.20 algorithm selection + shared protocol selection.
  // Every threshold comes from NcclSimulationConstants (via NcclSelectionParams).
  ncclInfo* MockNcclGroup::get_algo_proto_info(
      GroupType type,
      int rank,
      AstraSim::ComType op,
      uint64_t data_size,
      bool msccl) {
    using C = NcclSimulationConstants;
    const NcclSelectionParams& P = NcclSelectionParams::get();
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    NcclLog->writeLog(NcclLogLevel::DEBUG, "get_algo_proto_info type %d rank %d op %d data_size %lu msccl %d",
                      type, rank, static_cast<int>(op), data_size, msccl);

    if (GroupIndex.count(std::make_pair(rank, type)) == 0) {
      NcclLog->writeLog(NcclLogLevel::ERROR, "There is no corresponding group info, resulting in an error with get_algo_proto_info.");
      return nullptr;
    }
    const GroupInfo& gp_info = AllGroups[GroupIndex[std::make_pair(rank, type)]];

    // ---- cache ----------------------------------------------------------------
    std::string ncclInfoName = std::string(to_cstr(type)) + "_" + std::to_string(static_cast<int>(op)) + "_"
                             + std::to_string(data_size) + "_" + std::to_string(msccl);
    if (auto it = nccl_infos.find(ncclInfoName); it != nccl_infos.end()) {
      return it->second;
    }

    struct ncclInfo* info = new ncclInfo();
    info->nBytes = data_size;
    info->nChannels = 0;
    // Requires ncclInfo::coll to be AstraSim::ComType: genCustomFlowModels matches
    // to_cstr(info->coll) against the XML coll_type.
    info->coll = op;

    // ---- algorithm --------------------------------------------------------------
    if (msccl) {
      info->algorithm = NCCL_ALGO_MSCCL;   // custom XML schedule
    } else switch (op) {
      case AstraSim::ComType::All_Reduce: {
        const bool nvls = type == TP
                       && C::isNvlsCapable(gpu_type)
                       && P.nvls_enable
                       && gp_info.nRanks >= C::NVLS_MIN_RANKS
                       && data_size >= P.nvls_min_bytes;
        info->algorithm = nvls ? NCCL_ALGO_NVLS : NCCL_ALGO_RING;
        break;
      }
      case AstraSim::ComType::All_Gather:
      case AstraSim::ComType::Reduce_Scatter:
      case AstraSim::ComType::All_to_All:
      default:
        info->algorithm = NCCL_ALGO_RING;
        break;
    }

    // ---- protocol -----------------------------------------------------------------
    if (P.forced_proto >= 0) {
      info->protocol = P.forced_proto;      // applies to MSCCL too
    } else if (!P.proto_aware) {
      info->protocol = NCCL_PROTO_UNDEF;
    } else if (info->algorithm == NCCL_ALGO_NVLS || info->algorithm == NCCL_ALGO_NVLS_TREE) {
      info->protocol = (data_size <= C::NVLS_LL_MAX_BYTES) ? NCCL_PROTO_LL : NCCL_PROTO_SIMPLE;
    } else if (data_size <= C::RING_LL_MAX_BYTES) {
      info->protocol = NCCL_PROTO_LL;
    } else if (data_size <= C::RING_LL128_MAX_BYTES) {
      info->protocol = NCCL_PROTO_LL128;
    } else {
      info->protocol = NCCL_PROTO_SIMPLE;
    }

    // in MSCCL Simple chunk splitting is implemented whatever protocol is selected (to disable, use NCCL_PROTO_UNDEF).
    // LL and LL128 are not simulated and are only used for definying the AS_SEND_LAT.

    NcclLog->writeLog(NcclLogLevel::INFO, "%s: algorithm %d, protocol %d",
                      ncclInfoName.c_str(), info->algorithm, info->protocol);
    nccl_infos[ncclInfoName] = info;
    return info;
  }

  // Dump the per-collective point-to-point flow list (the "detailed" flow model)
  // to ncclFlowModel_detailed_flows.csv in the current working directory.
  // This was previously a dead output in Workload.cc (created but never written).
  // The full flow set is available here when a flow_model_name is first generated.
  void MockNcclGroup::dumpDetailedFlowModels(
      const std::string& coll_name,
      AstraSim::ComType op,
      uint64_t data_size,
      const std::map<int,std::shared_ptr<FlowModels>>& rank2flows,
      const struct ncclInfo* info) {
    const std::string fname = "ncclFlowModel_detailed_flows.csv";
    std::ofstream ofs;
    if (!detailed_dump_header_written) {
      ofs.open(fname, std::ios::out | std::ios::trunc);
      if (ofs.is_open()) {
        ofs << "collective,op,data_size,algorithm,protocol,channel_id,flow_id,src,dest,flow_size,"
               "chunk_id,chunk_count,conn_type,parent_flow_ids,prev_flow_ids\n";
        detailed_dump_header_written = true;
      }
    } else {
      ofs.open(fname, std::ios::out | std::ios::app);
    }
    if (!ofs.is_open()) return;
    int algo = info ? info->algorithm : -1;
    int proto = info ? info->protocol : -1;
    for (const auto& rk : rank2flows) {
      int rank = rk.first;
      if (!rk.second) continue;
      for (const auto& kv : *rk.second) {
        const SingleFlow& f = kv.second;
        // Each flow is stored under both its src and dest rank; emit once (src view).
        if (f.src != rank) continue;
        std::string parents;
        for (size_t i = 0; i < f.parent_flow_id.size(); ++i) {
          parents += (i ? ";" : "") + std::to_string(f.parent_flow_id[i]);
        }
        std::string prevs;
        for (size_t i = 0; i < f.prev.size(); ++i) {
          prevs += (i ? ";" : "") + std::to_string(f.prev[i]);
        }
        ofs << coll_name << ',' << static_cast<int>(op) << ',' << data_size << ','
            << algo << ',' << proto << ','
            << f.channel_id << ',' << f.flow_id << ',' << f.src << ',' << f.dest << ','
            << f.flow_size << ',' << f.chunk_id << ',' << f.chunk_count << ','
            << f.conn_type << ',' << parents << ',' << prevs << '\n';
      }
    }
    ofs.close();
  }
}