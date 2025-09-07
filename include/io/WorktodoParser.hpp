// io/WorktodoParser.hpp
#pragma once
#include <optional>
#include <string>
#include <cstdint>
#include <vector>

namespace io {

enum TestType {
    TESTTYPE_UNSUPPORTED = 0,
    TESTTYPE_PRP = 1,
    TESTTYPE_LL  = 2,
    TESTTYPE_PM1 = 3
};

struct FactoringOptions {
    uint64_t B1 = 0;  // Bound 1
    uint64_t B2 = 0;  // Bound 2
};

// Written to but not actually read from (only needed for JsonBuilder, but that one just directly tests knownFactors)
struct PrimalityOptions {
    uint32_t residueType = 1;
}

struct WorktodoEntry {
    TestType testType = TESTTYPE_UNSUPPORTED;
    uint32_t exponent = 0;
    uint32_t k = 0, b = 0;
    int32_t c = 0;
    std::string aid;
    std::string rawLine;  
    std::vector<std::string> knownFactors;  
    union {
        FactoringOptions factoring;
        PrimalityOptions primality;
    } options;

    bool isMersenne() const { return k == 1 && b == 2 && c == -1; }
    bool isWagstaff() const { return k == 1 && b == 2 && c == 1 && knownfactors.length() > 0 && knownfactors[0] == "3"; }

    std::string toString() const;
};

class WorktodoParser {
public:
    explicit WorktodoParser(const std::string& filename);
    std::optional<WorktodoEntry> parse();
    bool removeFirstProcessed();  // supprime la 1ʳᵉ entrée non vide et la sauvegarde

private:
    std::string filename_;
};

} // namespace io
