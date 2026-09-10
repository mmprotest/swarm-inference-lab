#include <stdint.h>
#include <stddef.h>
#ifdef _WIN32
__declspec(dllexport)
#endif
uint64_t e026_fnv(const unsigned char * data, size_t size) {
    uint64_t hash = UINT64_C(0xcbf29ce484222325);
    for (size_t i = 0; i < size; ++i) {
        hash ^= data[i];
        hash *= UINT64_C(0x100000001b3);
    }
    return hash;
}
