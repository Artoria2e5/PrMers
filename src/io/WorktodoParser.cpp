/*
 * Mersenne OpenCL Primality Test Host Code
 *
 * This code is inspired by:
 *   - "mersenne.cpp" by Yves Gallot (Copyright 2020, Yves Gallot) based on
 *     Nick Craig-Wood's IOCCC 2012 entry (https://github.com/ncw/ioccc2012).
 *   - The Armprime project, explained at:
 *         https://www.craig-wood.com/nick/armprime/
 *     and available on GitHub at:
 *         https://github.com/ncw/
 *   - Yves Gallot (https://github.com/galloty), author of Genefer 
 *     (https://github.com/galloty/genefer22), who helped clarify the NTT and IDBWT concepts.
 *   - The GPUOwl project (https://github.com/preda/gpuowl), which performs Mersenne
 *     searches using FFT and double-precision arithmetic.
 * This code performs a Mersenne prime search using integer arithmetic and an IDBWT via an NTT,
 * executed on the GPU through OpenCL.
 *
 * Author: Cherubrock
 *
 * This code is released as free software. 
 */
// io/WorktodoParser.cpp
#include "io/WorktodoParser.hpp"
#include "math/Cofactor.hpp"
#include "util/StringUtils.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstdio>
#include <vector>

#define CONTINUE_WITH_REASON(reason) { std::cerr << reason << std::endl; continue; }
#define FALSE_WITH_REASON(reason) { std::cerr << reason << std::endl; return false; }

namespace io {

std::string WorktodoEntry::toString() const {
    std::ostringstream oss;
    oss << "entry: ";
    if (testType == TESTTYPE_PRP) oss << "PRP ";
    else if (testType == TESTTYPE_LL) oss << "LL ";
    else if (testType == TESTTYPE_PM1) oss << "P-1 ";
    else oss << "Unsupported op ";
    oss << "on " << k << "*" << b << "^" << exponent << (c >= 0 ? "+" : "") << c;
    if (isMersenne()) oss << " (Mersenne)";
    if (isWagstaff()) oss << " (Wagstaff)";
    oss << (knownFactors.empty() ? "" : " with " << knownFactors.size() << " known factors.");
    if (testType == TESTTYPE_PM1) {
        oss << " B1=" << options.factoring.B1 << ", B2=" << options.factoring.B2;
    } else if (testType == TESTTYPE_PRP || testType == TESTTYPE_LL) {
        oss << " residueType=" << options.primality.residueType;
    }
    if (!aid.empty()) oss << ", AID=" << aid;
}

WorktodoParser::WorktodoParser(const std::string& filename)
  : filename_(filename)
{}

static bool isHex(const std::string& s) {
    if (s.size() != 32) return false;
    for (char c : s)
        if (!std::isxdigit(static_cast<unsigned char>(c))) return false;
    return true;
}

// Split string respecting quoted sections (for PRP-CF assignment parsing)
std::vector<std::string> splitRespectingQuotes(const std::string& s, char delim) {
  std::vector<std::string> result;
  std::string current;
  bool inQuotes = false;
  
  for (char c : s) {
    if (c == '"') {
      inQuotes = !inQuotes;
      current += c;
    } else if (c == delim && !inQuotes) {
      result.push_back(current);
      current.clear();
    } else {
      current += c;
    }
  }
  if (!current.empty()) {
    result.push_back(current);
  }
  return result;
}

// Parse comma-separated factors from quoted string like "36357263,145429049,8411216206439"
static std::vector<std::string> parseFactors(const std::string& factorStr) {
    std::vector<std::string> factors;
    
    // Remove leading/trailing whitespace and check for quotes
    std::string trimmed = factorStr;
    while (!trimmed.empty() && std::isspace(trimmed.back())) {
        trimmed.pop_back();
    }
    
    if (trimmed.size() >= 2 && trimmed.front() == '"' && trimmed.back() == '"') {
        // Remove quotes and split by comma
        std::string content = trimmed.substr(1, trimmed.size() - 2);
        factors = util::split(content, ',');
    }
    
    return factors;
}

static bool parsePartsFactors(WorktodoEntry& entry, std::vector<std::string>& parts) {
    if (parts.empty()) return false;
    std::vector<std::string> kf = parseFactors(parts.back());
    if (!kf.empty()) {
        entry.knownFactors = std::move(kf);
        parts.pop_back();
    }
    return true;
}

static bool parseKbnc(WorktodoEntry& entry, std::vector<std::string>& kbnc) {
    if (kbnc.size() < 4) FALSE_WITH_REASON("Skip: not enough parts for k,b,n,c");
    try {
        entry.k = static_cast<uint32_t>(std::stoul(kbnc[0]));
        entry.b = static_cast<uint32_t>(std::stoul(kbnc[1]));
        entry.exponent = static_cast<uint32_t>(std::stoul(kbnc[2]));
        entry.c = static_cast<int32_t>(std::stoi(kbnc[3]));
    } catch (...) {
        return false;
    }
    if (entry.k == 0 || entry.b < 2 || entry.exponent == 0) FALSE_WITH_REASON("Skip: invalid k,b,n,c values");
    kbnc.erase(kbnc.begin(), kbnc.begin() + 4);
    return true;
}

static bool parseExponent(WorktodoEntry& entry, std::vector<std::string>& parts) {
    if (parts.empty()) return false;
    try {
        entry.exponent = static_cast<uint32_t>(std::stoul(parts[0]));
    } catch (...) {
        return false;
    }
    if (entry.exponent == 0) FALSE_WITH_REASON("Skip: invalid exponent 0");
    parts.erase(parts.begin());
    return true;
}

// assume parts[0] is either exponent or "1" for k,b,n,c
static bool parseExponentOrKbnc(WorktodoEntry& entry, std::vector<std::string>& parts) {
    if (parts[0] == "1" && parseKbnc(entry, parts)) return true;
    return parseExponent(entry, parts);
}

static bool parseMandatorySomething(std::vector<std::string>& parts) {
    if (parts.empty()) return false;
    parts.erase(parts.begin());
    return true;
}

static bool parseFactoringOpts(WorktodoEntry& entry, std::vector<std::string>& parts) {
    if (parts.size() < 2) FALSE_WITH_REASON("Skip: not enough parts for B1,B2");
    try {
        entry.options.factoring.B1 = static_cast<uint64_t>(std::stoull(parts[0]));
        entry.options.factoring.B2 = static_cast<uint64_t>(std::stod(parts[1])); // "1.3" accepté
    } catch (...) {
        return false;
    }
    if (entry.options.factoring.B1 == 0 || entry.options.factoring.B2 < entry.options.factoring.B1)
        FALSE_WITH_REASON("Skip: invalid B1,B2 values");
    parts.erase(parts.begin(), parts.begin() + 2);
    return true;
}

std::optional<WorktodoEntry> WorktodoParser::parse() {
    std::ifstream file(filename_);
    if (!file.is_open()) {
        std::cerr << "Cannot open " << filename_ << "\n";
        return std::nullopt;
    }
    auto trim_inplace = [](std::string& s){
        size_t a = s.find_first_not_of(" \t\r\n");
        size_t b = s.find_last_not_of(" \t\r\n");
        if (a == std::string::npos) { s.clear(); return; }
        s = s.substr(a, b - a + 1);
    };
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        auto top = util::split(line, '=');
        if (top.size() < 2) continue;

        WorktodoEntry entry;
        bool isPRP  = (top[0] == "PRP" || top[0] == "PRPDC");
        bool isLL   = (top[0] == "Test" || top[0] == "DoubleCheck");
        bool isPF   = (top[0] == "PFactor");
        bool isPM1  = (top[0] == "Pminus1");

        if (isLL) entry.testType = TESTTYPE_LL;
        else if (isPRP) entry.testType = TESTTYPE_PRP;
        else if (isPF || isPM1) entry.testType = TESTTYPE_PM1;
        else CONTINUE_WITH_REASON("Skip unsupported test type: " + top[0]);

        auto parts = splitRespectingQuotes(top[1], ',');
        if (!parts.empty() && (parts[0].empty() || parts[0] == "N/A"))
            parts.erase(parts.begin());

        std::string aid;
        if (!parts.empty() && (isHex(parts[0]) || ((isPF || isPM1) && (parts[0]) == "AID"))) {
            aid = parts[0];
            parts.erase(parts.begin());
        }

        entry.rawLine   = line;
        entry.aid       = aid;

        if (isPF) {
            // Pfactor=exponent,how_far_factored,ll_tests_saved_if_factor_found[,known_factors]	
            // Pfactor=k,b,n,c,how_far_factored,ll_tests_saved_if_factor_found[,known_factors]	
            if (!parseExponentOrKbnc(entry, parts)) continue;
            if (!entry.isMersenne()) CONTINUE_WITH_REASON("Skip unsupported PFactor line (only Mersenne supported)");

            if (!parseMandatorySomething(parts)) CONTINUE_WITH_REASON("Bad PFactor line (missing how_far_factored)");
            if (!parseMandatorySomething(parts)) CONTINUE_WITH_REASON("Bad PFactor line (missing ll_tests_saved_if_factor_found)");

            if (!parseFactoringOpts(entry, parts)) continue;
            parseMaybeFactors(entry, parts);
        } else if (isPM1) {
            // Pminus1=k,b,n,c,B1,B2[,how_far_factored][,B2_start][,"factors"]
            if (!parseKbnc(entry, parts)) continue;
            if (!entry.isMersenne()) CONTINUE_WITH_REASON("Skip unsupported Pminus1 line (only Mersenne supported)");

            if (!parseFactoringOpts(entry, parts)) continue;
            if (!parseMandatorySomething(parts)) CONTINUE_WITH_REASON("Bad Pminus1 line (missing how_far_factored)");
            if (!parseMaybeFactors(entry, parts)) continue;
        } else if (isLL) {
            // {Test|DoubleCheck}=exponent,how_far_factored,has_been_pminus1ed
            if (!parseExponent(entry, parts)) continue;
            if (!parseMandatorySomething(parts)) CONTINUE_WITH_REASON("Bad LL line (missing how_far_factored)");
            if (!parseMandatorySomething(parts)) CONTINUE_WITH_REASON("Bad LL line (missing has_been_pminus1ed)");
        } else if (isPRP) {
            // PRP{|DC}=k,b,n,c[,how_far_factored,tests_saved[,base,residue_type]][,known_factors]
            if (!parseKbnc(entry, parts)) continue;
            if (parts.size() == 1 || parts.size() == 3 || parts.size() == 5) {
                if (!parsePartsFactors(entry, parts)) CONTINUE_WITH_REASON("Bad PRP line (bad known factors part)");
                if (entry.isMersenne()) {
                    if (!math::Cofactor::validateFactors(entry.exponent, entry.knownFactors))
                        CONTINUE_WITH_REASON("Skip PRP line: invalid known factors for exponent");
                    entry.options.primality.residueType = 5; // Mersenne cofactor
                }
            }
            if (!entry.isMersenne() && !entry.isWagstaff())
                CONTINUE_WITH_REASON("Skip unsupported PRP line (only Mersenne and Wagstaff supported)");
            if (parts.size >= 2) {
                parts.erase(parts.begin(), parts.begin() + 2); // how_far_factored,tests_saved
            }

            try {
                if (parts.size() >= 2) {
                    uint32_t base = static_cast<uint32_t>(std::stoul(parts[0]));
                    uint32_t rt   = static_cast<uint32_t>(std::stoul(parts[1]));
                    if (base < 2) CONTINUE_WITH_REASON("Skip PRP line: invalid base < 2");
                    if (base != 3) CONTINUE_WITH_REASON("Skip PRP line: only base 3 implemented");
                    if (rt != entry.options.primality.residueType)
                        std::cout << "Warning: PRP line residue type " << rt
                                  << " does not match expected " << entry.options.primality.residueType << std::endl;
                    parts.erase(parts.begin(), parts.begin() + 2);
                }
            } catch (...) {
                continue;
            }
        }

        std::cout << "Loaded " << entry.toString() << std::endl;
        return entry;
    }

    std::cerr << "No valid entry found in " << filename_ << "\n";
    return std::nullopt;
}


bool WorktodoParser::removeFirstProcessed() {
    std::ifstream inFile(filename_);
    std::ofstream tempFile(filename_ + ".tmp");
    std::ofstream saveFile("worktodo_save.txt", std::ios::app);

    if (!inFile || !tempFile || !saveFile)
        return false;

    std::string line;
    bool skipped = false;

    while (std::getline(inFile, line)) {
        if (!skipped && !line.empty()) {
            skipped = true;
            saveFile << line << "\n";
            continue;
        }
        tempFile << line << "\n";
    }

    inFile.close();
    tempFile.close();
    saveFile.close();

    std::remove(filename_.c_str());
    std::rename((filename_ + ".tmp").c_str(), filename_.c_str());

    return skipped;
}



} // namespace io
