#include "minitest.h"
#include <cstdarg>
#include <iostream>
#include <unordered_map>

#ifndef _WIN32
#include <unistd.h>
#else
#endif

using namespace std;

static unordered_map<string_view, int> __arg_counts
  = {
      { "--list_tests", 0 },
      { "--also_run_disabled_tests", 0 },
      { "--filter", 1 },
      { "--color", 1 },
      { "--print_time", 1 }
    };

int main(int argc, const char** argv)
{
  unordered_map<string_view, string_view> args;
  for (int i=1; i<argc; ++i) {
    auto it = __arg_counts.find(argv[i]);
    if (it != __arg_counts.end()) {
      if (it->second) {
        if (argc - i < 2) {
          cerr << "[minitest] insufficient argument for `" << argv[i] << "'."
               << endl;
          exit(-1);
        }
        args[string_view{argv[i]}.substr(2)] = argv[i + 1];
        ++i;
      } else {
        args[string_view{argv[i]}.substr(2)] = "";
      }
    } else {
      cerr << "[minitest] unknown argument `" << argv[i] << "'." << endl;
      exit(-1);
    }
  }
  minitest::TestPlanConfig cfg;
  if (args.count("list_tests")) {
    cfg.list_only = true;
  }
  if (args.count("color")) {
    string_view val = args["color"];
    if (val == "0" || val == "off" || val == "false" || val == "False") {
      cfg.colored_output = false;
    } else if (val == "1" || val == "on" || val == "true" || val == "True") {
      cfg.colored_output = true;
    } else if (val == "2" || val == "auto" || val == "Auto") {
      cfg.colored_output = isatty(STDOUT_FILENO);
    } else {
      cerr << "[minitest] unknown argument `" << val << "' "
           << "for `--color'." << endl;
      exit(-1);
    }
  } else {
    cfg.colored_output = isatty(STDOUT_FILENO);
  }
  if (args.count("print_time")) {
    string_view val = args["print_time"];
    if (val == "0" || val == "off" || val == "false" || val == "False") {
      cfg.print_time = 0;
    } else if (val == "1" || val == "on" || val == "true" || val == "True") {
      cfg.print_time = 1;
    } else {
      cerr << "[minitest] unknown argument `" << val << "' "
           << "for `--print_time'." << endl;
      exit(-1);
    }
  }
  if (args.count("also_run_disabled_tests")) {
    cfg.also_run_disabled = true;
  }
  if (args.count("filter")) {
    cfg.filter = args["filter"];
  }

  auto test_mgr = minitest::TestManager::get_instance();
  test_mgr->run(cfg);

  return 0;
}
