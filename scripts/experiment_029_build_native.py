from pathlib import Path
root=Path(__file__).resolve().parents[1]
source=(root/'native/experiment_028/e028_stage_server.cpp').read_text()
prefix=source[:source.index('class stage_runtime {')]
prefix=prefix.replace('#include <algorithm>', '#include <algorithm>\n#include <memory>\n#include <functional>')
prefix=prefix.replace('bool mtp = false;', 'bool mtp = false; bool dflash = false; std::string draft_path;')
prefix=prefix.replace('} else if (arg == "--mtp") {','} else if (arg == "--dflash") {\n            result.dflash = true;\n        } else if (arg == \"--draft-model\") {\n            result.draft_path = next();\n        } else if (arg == "--mtp") {')
suffix=source[source.index('socket_owner listen_socket'):]
start=suffix.index('                case operation::infer:')
end=suffix.index('                case operation::stats:',start)
suffix=suffix[:start]+'''                case operation::infer:
                case static_cast<operation>(11):
                case static_cast<operation>(12):
                case static_cast<operation>(13):
                case static_cast<operation>(14):
                case static_cast<operation>(15):
                    result = runtime.handle(request, payload, response);
                    break;
'''+suffix[end:]
suffix=suffix.replace('runtime.fingerprint_json();','runtime.fingerprint_json(request.arg);')
dest=root/'native/experiment_029';dest.mkdir(exist_ok=True)
(dest/'e029_stage_server.cpp').write_text(prefix+'#include "tree_runtime.inc"\n\n'+suffix)
cm=(root/'native/experiment_028/CMakeLists.txt').read_text().replace('e028','e029')
(dest/'CMakeLists.txt').write_text(cm)
cmd=(root/'scripts/experiment_028_build.cmd').read_text().replace('experiment_028','experiment_029').replace('experiment-028','experiment-029')
(root/'scripts/experiment_029_build.cmd').write_text(cmd)
