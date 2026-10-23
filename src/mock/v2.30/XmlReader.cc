#include "XmlReader.hh"
#include <libxml/parser.h>
#include <libxml/tree.h>
#include <iostream>
#include <stdexcept>
#include <cstring> // for strcmp
#include <algorithm> // for std::sort
#include "MockNcclLog.h"

// --- Helper Class ---
// Manages the xmlDoc lifecycle and provides safe attribute retrieval methods
class XmlParserHelper {
private:
    xmlDocPtr doc = nullptr;
    bool ownsDoc = true;

public:
    // Constructor attempts to load the file
    explicit XmlParserHelper(const std::string& filepath) {
        doc = xmlReadFile(filepath.c_str(), NULL, 0);
    }

    explicit XmlParserHelper(xmlDocPtr existingDoc, bool takeOwnership = true)
        : doc(existingDoc), ownsDoc(takeOwnership) {}

    // Modify destructor to check ownership
    ~XmlParserHelper() {
        if (doc && ownsDoc) {
            xmlFreeDoc(doc);
        }
    }

    // Disable copying to prevent double-free issues
    XmlParserHelper(const XmlParserHelper&) = delete;
    XmlParserHelper& operator=(const XmlParserHelper&) = delete;

    // Check if file was loaded successfully
    bool isValid() const {
        return doc != nullptr;
    }

    // Get the root element
    xmlNodePtr getRoot() const {
        return doc ? xmlDocGetRootElement(doc) : nullptr;
    }

    // Get string attribute, converting xmlChar* and freeing it
    std::string getAttr(xmlNodePtr node, const char* name, const std::string& def = "") const {
        xmlChar* prop = xmlGetProp(node, reinterpret_cast<const xmlChar*>(name));
        if (!prop) return def;
        
        std::string res(reinterpret_cast<char*>(prop));
        xmlFree(prop); // libxml2 requires freeing the string returned by xmlGetProp
        return res;
    }

    // Get integer attribute
    int getInt(xmlNodePtr node, const char* name, int def = 0) const {
        xmlChar* prop = xmlGetProp(node, reinterpret_cast<const xmlChar*>(name));
        if (!prop) return def;
        
        int res = std::atoi(reinterpret_cast<char*>(prop));
        xmlFree(prop);
        return res;
    }

    // Get boolean attribute
    bool getBool(xmlNodePtr node, const char* name, bool def = false) const {
        xmlChar* prop = xmlGetProp(node, reinterpret_cast<const xmlChar*>(name));
        if (!prop) return def;

        bool res = (xmlStrcmp(prop, reinterpret_cast<const xmlChar*>("true")) == 0 || 
                    xmlStrcmp(prop, reinterpret_cast<const xmlChar*>("1")) == 0);
        xmlFree(prop);
        return res;
    }
};

MscclAlgorithm parseMscclXmlHeader(const std::string& filepath) {
  MockNcclLog* NcclLog = MockNcclLog::getInstance();

  std::ifstream file(filepath);
  std::string header_string;
  char ch;

  // Read character by character until we find the closing '>' of <algo>
  bool foundAlgo = false;
  while (file.get(ch)) {
    header_string += ch;
    if (!foundAlgo && header_string.find("<algo") != std::string::npos) {
      foundAlgo = true;
    }
    if (foundAlgo && ch == '>') {
      header_string.pop_back();
      header_string += "/>";
      break;
    }
  }

  // Parse the header string with libxml2
  xmlDocPtr doc = xmlReadMemory(header_string.c_str(), header_string.size(), NULL, NULL, 0);
  if (!doc) {
    NcclLog->writeLog(NcclLogLevel::ERROR, "Failed to parse XML header from '%s'.", filepath.c_str());
    return {};
  }
  XmlParserHelper xml(doc);
  xmlNodePtr root = xml.getRoot();

  if (!root || xmlStrcmp(root->name, reinterpret_cast<const xmlChar*>("algo")) != 0) {
    NcclLog->writeLog(NcclLogLevel::ERROR, "XML missing root <algo> node in file: %s", filepath.c_str());
    return {};
  }

  MscclAlgorithm header;

  // Now you can use xml.getAttr() like before!
  header.name = xml.getAttr(root, "name");
  header.filename = filepath;
  header.coll_type = xml.getAttr(root, "coll");
  header.n_gpus = xml.getInt(root, "ngpus");
  header.n_channels = xml.getInt(root, "nchannels");
  header.nchunksperloop = xml.getInt(root, "nchunksperloop");
  header.maxBytes = xml.getInt(root, "maxBytes");
  header.minBytes = xml.getInt(root, "minBytes");
  header.group_type = xml.getAttr(root, "groupType", "NONE");
  header.first_rank_id = xml.getInt(root, "firstRankId", -1);


  NcclLog->writeLog(NcclLogLevel::DEBUG, "Parsed header for algorithm '%s': %d GPUs, %d channels, bytes [%d, %d]",
                    header.name.c_str(), header.n_gpus, header.n_channels, header.minBytes, header.maxBytes);

  return header;
}

MscclAlgorithm parseMscclXml(const std::string& filepath) {
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    NcclLog->writeLog(NcclLogLevel::INFO, "Parsing MSCCL XML file: %s", filepath.c_str());

    // 1. Initialize Helper (Loads file)
    XmlParserHelper xml(filepath);

    if (!xml.isValid()) {
        NcclLog->writeLog(NcclLogLevel::ERROR, "Failed to parse XML file '%s'.", filepath.c_str());
        return {};
    }

    xmlNodePtr algo_node = xml.getRoot();

    // Check if root exists and is named "algo"
    if (!algo_node || xmlStrcmp(algo_node->name, (const xmlChar*)"algo") != 0) {
        NcclLog->writeLog(NcclLogLevel::ERROR, "XML missing root <algo> node in file: %s", filepath.c_str());
        throw std::runtime_error("XML missing <algo> node.");
    }

    MscclAlgorithm algo;

    // 2. Parse <algo> attributes
    algo.name           = xml.getAttr(algo_node, "name");
    algo.filename       = filepath;
    algo.coll_type      = xml.getAttr(algo_node, "coll");
    algo.n_gpus         = xml.getInt(algo_node, "ngpus");
    algo.n_channels     = xml.getInt(algo_node, "nchannels");
    algo.nchunksperloop = xml.getInt(algo_node, "nchunksperloop");
    algo.maxBytes       = xml.getInt(algo_node, "maxBytes");
    algo.minBytes       = xml.getInt(algo_node, "minBytes");
    algo.group_type     = xml.getAttr(algo_node, "groupType", "NONE");
    algo.first_rank_id  = xml.getInt(algo_node, "firstRankId", -1);
    
    NcclLog->writeLog(NcclLogLevel::DEBUG, "Parsed algorithm '%s' for %d GPUs and %d channels.", algo.name.c_str(), algo.n_gpus, algo.n_channels);

    // 3. Parse children nodes (GPUs)
    for (xmlNodePtr gpu_node = algo_node->children; gpu_node; gpu_node = gpu_node->next) {
        if (gpu_node->type == XML_ELEMENT_NODE && xmlStrcmp(gpu_node->name, (const xmlChar*)"gpu") == 0) {
            
            MscclGpu gpu;
            gpu.id = xml.getInt(gpu_node, "id");

            // Parse each <tb> (thread block) within the GPU
            for (xmlNodePtr tb_node = gpu_node->children; tb_node; tb_node = tb_node->next) {
                if (tb_node->type == XML_ELEMENT_NODE && xmlStrcmp(tb_node->name, (const xmlChar*)"tb") == 0) {
                    
                    MscclThreadBlock tb;
                    tb.id = xml.getInt(tb_node, "id");
                    tb.send_peer = xml.getInt(tb_node, "send", -1);
                    tb.recv_peer = xml.getInt(tb_node, "recv", -1);
                    tb.channel = xml.getInt(tb_node, "chan");

                    // Parse each <step> within the thread block
                    for (xmlNodePtr step_node = tb_node->children; step_node; step_node = step_node->next) {
                        // Filter to ensure the parser only processes valid <step> elements and ignores non-xml elements e.g., comments or spaces
                        if (step_node->type == XML_ELEMENT_NODE && xmlStrcmp(step_node->name, (const xmlChar*)"step") == 0) {                            MscclStep step;
                            step.id             = xml.getInt(step_node, "s");
                            step.type           = xml.getAttr(step_node, "type");
                            step.src_buffer     = xml.getAttr(step_node, "srcbuf");
                            step.src_offset     = xml.getInt(step_node, "srcoff");
                            step.dst_buffer     = xml.getAttr(step_node, "dstbuf");
                            step.dst_offset     = xml.getInt(step_node, "dstoff");
                            step.count          = xml.getInt(step_node, "cnt");
                            step.dep_id         = xml.getInt(step_node, "depid", -1);
                            step.dep_step       = xml.getInt(step_node, "deps", -1);
                            step.has_dependence = xml.getBool(step_node, "hasdep", false);
                            
                            tb.steps.push_back(step);
                        }
                    }
                    gpu.thread_blocks.push_back(tb);
                }
            }
            algo.gpus[gpu.id] = gpu;
        }
    }

    // 4. Sort (Deterministic ordering)
    for (auto& gpu_pair : algo.gpus) {
        MscclGpu& gpu = gpu_pair.second;
        std::sort(gpu.thread_blocks.begin(), gpu.thread_blocks.end(), 
            [](const MscclThreadBlock& a, const MscclThreadBlock& b) {
                return a.id < b.id;
        });
        for (auto& tb : gpu.thread_blocks) {
            std::sort(tb.steps.begin(), tb.steps.end(), 
                [](const MscclStep& a, const MscclStep& b) {
                    return a.id < b.id;
            });
        }
    }

    NcclLog->writeLog(NcclLogLevel::INFO, "Successfully parsed MSCCL XML for algorithm '%s'.", algo.name.c_str());
    return algo;
}