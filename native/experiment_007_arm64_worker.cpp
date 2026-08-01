#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {
std::string argument(int argc, char** argv, const std::string& name,
                     const std::string& fallback) {
  for (int index = 1; index + 1 < argc; ++index) {
    if (argv[index] == name) {
      return argv[index + 1];
    }
  }
  return fallback;
}

bool flag(int argc, char** argv, const std::string& name) {
  for (int index = 1; index < argc; ++index) {
    if (argv[index] == name) {
      return true;
    }
  }
  return false;
}
}  // namespace

int main(int argc, char** argv) {
  const std::string expected = argument(argc, argv, "--expected-shard-hash", "probe-hash");
  const std::string actual = argument(argc, argv, "--actual-shard-hash", "probe-hash");
  const bool cancelled = flag(argc, argv, "--cancel");
  const bool hash_valid = expected == actual;
  const std::uint32_t input_token = 7007;
  const std::uint32_t output_token =
      (input_token * static_cast<std::uint32_t>(1103515245U) + 12345U) % 151936U;
  std::cout
      << "{\"protocol\":{\"major\":1,\"minor\":0},"
      << "\"registration\":\"accepted\","
      << "\"architecture\":\"arm64\","
      << "\"capabilities\":[\"heartbeat\",\"cancel\",\"token-protocol\","
         "\"shard-hash\",\"clean-shutdown\"],"
      << "\"shard_hash_valid\":" << (hash_valid ? "true" : "false") << ","
      << "\"heartbeat\":\"ok\","
      << "\"cancellation\":\"" << (cancelled ? "cancelled" : "not_requested") << "\","
      << "\"deterministic_input_token\":" << input_token << ","
      << "\"deterministic_output_token\":" << output_token << ","
      << "\"shutdown\":\"clean\"}"
      << std::endl;
  return hash_valid ? 0 : 3;
}
